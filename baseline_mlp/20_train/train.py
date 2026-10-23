"""Train the baseline MLP on Sem-Lex phonological features.

Two head sets, run in parallel as separate jobs:
  --head-set all16    the SL-GCN / SHuBERT Table-9 head list
  --head-set heads6   Handshape + the 5 attributes the LSC benchmark scores

Optimiser settings follow the Handshape-GNN baseline's defaults (Adam,
lr 1e-3, StepLR step 30 gamma 0.1, summed/averaged cross-entropy, batch 32),
with two additions the multi-head setting needs:
  * early stopping on val macro Recall@1 (no fixed epoch budget is defensible
    across two head sets of very different size),
  * drop_last=True -- the trunk ends in BatchNorm1d, which throws on a
    batch of 1 in train mode.

Model selection is on val macro Recall@1 over the head set being trained;
the test set is never read here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import SemLexFrames, majority_floors  # noqa: E402
from model import build_model  # noqa: E402
from data_config import RUNS_DIR, UNOBSERVABLE_HEADS  # noqa: E402


def recall_at_1(logits_list, y) -> list:
    """Per-head top-1 accuracy (= Recall@1 on a single-label head)."""
    out = []
    for h, logits in enumerate(logits_list):
        pred = logits.argmax(dim=1)
        out.append(float((pred == y[:, h]).float().mean()))
    return out


@torch.no_grad()
def evaluate(model, loader, device, n_heads):
    model.eval()
    correct = np.zeros(n_heads, dtype=np.int64)
    scored = np.zeros(n_heads, dtype=np.int64)   # per-head, NA rows excluded
    total = 0
    loss_sum = np.zeros(n_heads)
    # ignore_index=-1: PopSign masks an absent feature value with -1,
    # where Sem-Lex encodes NA as an explicit class. Sem-Lex ids are all
    # >= 0, so this is a no-op there and correct for both.
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        for h, lg in enumerate(logits):
            # Score each head only where it HAS a label. Counting masked
            # rows as misses would silently cap PopSign's heavily-NA heads
            # (Nondominant Handshape is 66% NA) at their labelled fraction.
            m = y[:, h] >= 0
            k = int(m.sum())
            if k:
                correct[h] += int((lg.argmax(dim=1)[m] == y[m, h]).sum())
                scored[h] += k
                loss_sum[h] += float(criterion(lg, y[:, h])) * k
        total += len(X)
    recalls = [c / s if s else 0.0 for c, s in zip(correct, scored)]
    losses = [l / s if s else 0.0 for l, s in zip(loss_sum, scored)]
    return recalls, losses, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-set", choices=["all16", "heads6"], required=True)
    ap.add_argument("--corpus", choices=["semlex", "popsign"],
                    default="semlex",
                    help="training corpus. PopSign uses its OWN 16-feature "
                         "label space (11 of 16 heads differ in class count "
                         "from Sem-Lex), so runs on the two are not "
                         "head-for-head comparable")
    ap.add_argument("--arch", choices=["multi", "hierarchical"],
                    default="multi",
                    help="hierarchical is only meaningful for heads6")
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop after N epochs with no val improvement")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SemLexFrames("train", args.head_set, corpus=args.corpus)
    val_ds = SemLexFrames("val", args.head_set, corpus=args.corpus)
    names = train_ds.head_names
    counts = train_ds.num_classes
    n_heads = len(names)

    print(f"[data] train {train_ds.stats} \n[data] val {val_ds.stats}",
          flush=True)
    print(f"[heads] {list(zip(names, counts))}", flush=True)

    floors = majority_floors(train_ds, val_ds)
    print(f"[floors] val majority-class Recall@1: "
          f"{json.dumps({k: round(v, 4) for k, v in floors.items()})}",
          flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(
        args.head_set, counts, arch=args.arch,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)
    print(f"[model] {args.arch} {sum(p.numel() for p in model.parameters())} params",
          flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)   # see evaluate()

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best = {"macro": -1.0, "epoch": -1}
    history = []
    since_best = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(X)
            # Mean over heads, so the loss scale is comparable between the
            # 6-head and 16-head runs (the Handshape-GNN baseline averages
            # the same way).
            loss = sum(criterion(lg, y[:, h]) for h, lg in enumerate(logits))
            loss = loss / n_heads
            loss.backward()
            optimizer.step()
            run_loss += float(loss) * len(X)
            seen += len(X)
        scheduler.step()

        recalls, val_losses, n_val = evaluate(model, val_loader, device, n_heads)
        macro = float(np.mean(recalls))
        history.append({
            "epoch": epoch + 1,
            "train_loss": run_loss / max(seen, 1),
            "val_macro_r1": macro,
            "val_r1": dict(zip(names, [round(r, 4) for r in recalls])),
        })

        improved = macro > best["macro"]
        if improved:
            best = {"macro": macro, "epoch": epoch + 1,
                    "per_head": dict(zip(names, recalls))}
            since_best = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_macro_r1": macro,
                "val_per_head": dict(zip(names, recalls)),
                "config": {
                    "head_set": args.head_set, "arch": args.arch,
                    # Corpus is part of the model's identity: the two label
                    # spaces differ in 11 of 16 heads, so an eval that loaded
                    # these weights against the other corpus's maps would
                    # decode every one of those heads wrongly.
                    "corpus": args.corpus,
                    # Recorded so evals pick the same label build; legacy
                    # checkpoints lack the key and default to the NA-masked
                    # maps.
                    "na_policy": "class",
                    "head_names": names, "num_classes": counts,
                    "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                    "lr": args.lr, "batch_size": args.batch_size,
                    "seed": args.seed,
                },
            }, run_dir / "best.pt")
        else:
            since_best += 1

        print(f"epoch {epoch + 1:3d}  train_loss {run_loss / max(seen, 1):.4f}  "
              f"val_macro_R@1 {macro:.4f}{'  *' if improved else ''}  "
              f"({time.time() - t0:.1f}s)", flush=True)

        if since_best >= args.patience:
            print(f"[early-stop] no improvement in {args.patience} epochs",
                  flush=True)
            break

    summary = {
        "run_name": args.run_name,
        "head_set": args.head_set,
        "arch": args.arch,
        "best_epoch": best["epoch"],
        "best_val_macro_r1": best["macro"],
        "best_val_per_head": best.get("per_head", {}),
        "val_majority_floors": floors,
        "unobservable_heads": [n for n in names if n in UNOBSERVABLE_HEADS],
        "train_stats": train_ds.stats,
        "val_stats": val_ds.stats,
        "args": vars(args),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
