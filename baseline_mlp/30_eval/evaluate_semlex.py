"""Score a trained baseline-MLP checkpoint on the Sem-Lex test split.

Reports per-head Recall@1 next to two reference columns:
  * the majority-class floor computed on TRAIN and applied to the eval split
    -- a head at or below its floor learned nothing,
  * the SHuBERT paper's Table 9 Sem-Lex numbers, for the 16-head run.

The paper comparison is context, NOT a like-for-like target: SHuBERT is a
fine-tuned video transformer over face+hands+body across time, this is a
2-layer MLP over one still frame of one hand. The floor column is the honest
baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "20_train"))

from dataset import SemLexFrames, majority_floors  # noqa: E402
from model import build_model  # noqa: E402
from data_config import (  # noqa: E402
    RESULTS_DIR, RUNS_DIR, SHUBERT_REPO, UNOBSERVABLE_HEADS,
)

# Load the sibling leaf's data_config under a distinct module name: a plain
# `from data_config import ...` after a sys.path insert would hit Python's
# module cache and silently return OUR data_config (which has no
# PAPER_TABLE9_SEMLEX), dropping the paper column from every report.
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "shubert_data_config", SHUBERT_REPO / "data_config.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PAPER_TABLE9_SEMLEX = _mod.PAPER_TABLE9_SEMLEX
except Exception:      # keep the eval runnable if that leaf moves
    PAPER_TABLE9_SEMLEX = {}


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(
        cfg["head_set"], cfg["num_classes"], arch=cfg["arch"],
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg, ckpt


@torch.no_grad()
def score(model, ds, device, batch_size=512):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    n_heads = len(ds.head_names)
    correct = np.zeros(n_heads, dtype=np.int64)
    preds_all = [[] for _ in range(n_heads)]
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        for h, lg in enumerate(model(X)):
            pred = lg.argmax(dim=1)
            correct[h] += int((pred == y[:, h]).sum())
            preds_all[h].append(pred.cpu().numpy())
        total += len(X)
    recalls = correct / max(total, 1)
    preds = [np.concatenate(p) if p else np.zeros(0, int) for p in preds_all]
    return recalls, preds, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        RUNS_DIR / args.run_name / "best.pt"
    model, cfg, ckpt = load_checkpoint(ckpt_path, device)
    head_set = cfg["head_set"]

    # Follow the checkpoint, not a flag: loading PopSign weights against the
    # Sem-Lex label space would decode 11 of the 16 heads through the wrong
    # table. Older checkpoints predate the field and are Sem-Lex.
    corpus = cfg.get("corpus", "semlex")
    # Follow the checkpoint's own label build: new runs record
    # na_policy="class" (mandatory), legacy ones lack the key.
    na_policy = cfg.get("na_policy", "ignore")
    eval_ds = SemLexFrames(args.split, head_set, corpus=corpus,
                           na_policy=na_policy)
    train_ds = SemLexFrames("train", head_set, corpus=corpus,
                            na_policy=na_policy)
    floors = majority_floors(train_ds, eval_ds)

    recalls, preds, total = score(model, eval_ds, device)
    names = eval_ds.head_names

    rows = []
    for h, name in enumerate(names):
        n_distinct = int(len(np.unique(preds[h])))
        rows.append({
            "head": name,
            "recall_at_1": float(recalls[h]),
            "majority_floor": floors[name],
            "above_floor": float(recalls[h] - floors[name]),
            "n_classes": eval_ds.num_classes[h],
            "distinct_predictions": n_distinct,
            "collapsed": n_distinct == 1,
            "observable": name not in UNOBSERVABLE_HEADS,
            # SHuBERT Table 9 is a Sem-Lex result; no PopSign counterpart.
            "shubert_paper": (PAPER_TABLE9_SEMLEX.get(name)
                              if corpus == "semlex" else None),
        })

    observable = [r for r in rows if r["observable"]]
    out = {
        "run_name": args.run_name,
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("epoch"),
        "split": args.split,
        "n_clips": total,
        "macro_recall_at_1": float(np.mean(recalls)),
        "macro_majority_floor": float(np.mean(list(floors.values()))),
        "macro_recall_observable_heads": (
            float(np.mean([r["recall_at_1"] for r in observable]))
            if observable else None
        ),
        "heads_at_or_below_floor": [
            r["head"] for r in rows if r["above_floor"] <= 0
        ],
        "per_head": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Name the file after the corpus actually scored, not the script. A
    # PopSign run written as "semlex_test_*" would read as a Sem-Lex result
    # and is not comparable with one (different label space entirely).
    dest = RESULTS_DIR / f"{corpus}_{args.split}_{args.run_name}.json"
    dest.write_text(json.dumps(out, indent=2))

    width = max(len(n) for n in names)
    print(f"\n{'head'.ljust(width)}  R@1     floor   delta   paper   preds")
    for r in rows:
        paper = f"{r['shubert_paper']:.4f}" if r["shubert_paper"] else "   -  "
        flag = "" if r["observable"] else "  (unobservable)"
        print(f"{r['head'].ljust(width)}  {r['recall_at_1']:.4f}  "
              f"{r['majority_floor']:.4f}  {r['above_floor']:+.4f}  {paper}  "
              f"{r['distinct_predictions']:>3d}{flag}")
    print(f"\nmacro R@1 {out['macro_recall_at_1']:.4f}  "
          f"(floor {out['macro_majority_floor']:.4f})  n={total}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
