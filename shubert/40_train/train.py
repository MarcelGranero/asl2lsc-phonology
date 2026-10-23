"""Fine-tune SHuBERT on Sem-Lex phonological features (paper Table 9 setup).

Full fine-tuning of the pretrained backbone + BatchNorm + 16 linear heads on
time-averaged representations. Adam lr 1e-4, NO weight decay (paper), batch
128, up to 125 epochs, early stopping on val macro Recall@1.

Resumes automatically from <run_dir>/checkpoints/last.pt unless --fresh.
Single GPU. Per-epoch JSONL log in <run_dir>/log.jsonl.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
from data_config import RUNS_DIR  # noqa: E402
from dataset import (  # noqa: E402
    SemLexNpyDataset,
    SemLexPackedDataset,
    collate,
    load_n_classes,
)
from ablation import ALL_CHANNELS, resolve_channels  # noqa: E402
from engine import autocast_dtype, run_eval, to_device  # noqa: E402
from model import PhonologyClassifier  # noqa: E402
from model_loading import load_backbone  # noqa: E402


# Supervision ablation presets (--keep-heads). "handshape" = the 6
# handshape-family features — exactly the set PopSign ships labels for, so a
# run trained with it matches the PopSign models' supervision. All 16 head
# modules are still built (architecture unchanged, checkpoints stay
# load-compatible); the dropped heads simply get no loss, so their linear
# layers keep their random init and must never be read at eval time.
HEAD_PRESETS = {
    "handshape": ("Handshape", "Selected Fingers", "Flexion", "Spread",
                  "Thumb Position", "Thumb Contact"),
}


def resolve_heads(spec: str, feature_order: list) -> tuple:
    names = HEAD_PRESETS.get(spec) or tuple(
        s.strip() for s in spec.split(",") if s.strip())
    unknown = [n for n in names if n not in feature_order]
    if unknown:
        raise SystemExit(f"--keep-heads: unknown feature(s) {unknown}; "
                         f"valid: {feature_order}")
    # keep feature_order's ordering, not the spec's
    return tuple(n for n in feature_order if n in names)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--no-pack", action="store_true",
                        help="read per-clip npys via the manifest instead of "
                        "the packed blobs")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore an existing last.pt")
    parser.add_argument("--keep-channels", default=None,
                        help="input-modality ablation: keep only these "
                        "backbone channels, filling the rest with the "
                        "pretrained mask_emb. A preset ('hands', "
                        "'hands_body', ...) or a comma list "
                        "('left_hand,right_hand'). Default: all 4 (paper).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override cfg batch_size. Needed for PopSign: its "
                        "clips reach 1280 frames (Sem-Lex maxes at 179) and "
                        "batches pad to the longest clip, so the paper's 128 "
                        "OOMs a 16 GB T4 in backward. NOTE this is a real "
                        "deviation -- the head's BatchNorm is batch-size "
                        "dependent, so PopSign numbers are not comparable to "
                        "Sem-Lex ones; only scratch-vs-pretrained WITHIN a "
                        "corpus is (both arms use the same value).")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="override the backbone LR (cfg backbone_lr, or "
                        "the head LR under --random-init). Pass this only to "
                        "deviate deliberately -- it wins over both defaults.")
    parser.add_argument("--random-init", action="store_true",
                        help="pretraining ablation: build the SAME backbone "
                        "architecture but skip loading checkpoint weights, so "
                        "the only difference from the default run is that the "
                        "encoder was never SSL-pretrained. Also raises the "
                        "backbone LR to the head LR (the 1/10 rate exists to "
                        "protect pretrained features).")
    parser.add_argument("--keep-heads", default=None,
                        help="supervision ablation: train (loss + early stop) "
                        "only these feature heads. A preset ('handshape') or "
                        "a comma list of feature names. The other heads stay "
                        "at random init. Default: all 16 (paper).")
    parser.add_argument("--seed", type=int, default=None,
                        help="override cfg seed for seed-variance runs; the "
                        "resolved value is stored in the checkpoint config")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="DIAGNOSTIC ONLY: accumulate N micro-batches "
                        "before each optimizer step (effective batch = "
                        "batch_size x N). NOT the paper protocol: the head's "
                        "BatchNorm normalizes over the MICRO-batch, so this "
                        "is a documented deviation (4x32 measurably collapsed "
                        "to majority-class predictions; 2x64 is the arm this "
                        "flag exists to test). Default 1 = protocol behavior, "
                        "code path identical. Recorded in the checkpoint "
                        "config as grad_accum.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for key in ("train_split", "val_split", "epochs", "min_epochs"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    grad_accum = max(1, args.grad_accum)
    cfg["grad_accum"] = grad_accum
    if grad_accum > 1:
        print(f"GRAD ACCUM x{grad_accum}: DIAGNOSTIC arm, BatchNorm sees the "
              f"micro-batch -- not protocol-comparable")
    if args.keep_channels is not None:
        cfg["keep_channels"] = args.keep_channels
    if args.random_init:
        cfg["random_init"] = True
    # An explicit --backbone-lr is a deliberate deviation and wins over both
    # the cfg value and the random-init raise below; record that it was
    # user-set so the raise cannot silently override it.
    if args.backbone_lr is not None:
        cfg["backbone_lr"] = args.backbone_lr
        cfg["backbone_lr_explicit"] = True
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
        print(f"BATCH OVERRIDE: {args.batch_size} (cfg default was 128) — "
              f"BatchNorm is batch-size dependent, so this run is comparable "
              f"only to runs using the same value")
    if args.keep_heads is not None:
        cfg["keep_heads"] = args.keep_heads
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(cfg["seed"])

    run_dir = RUNS_DIR / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    # Corpus is inferred from the split name: popsign{6,16}[nac]_train ->
    # that corpus's label maps. Sem-Lex splits keep the default. Longest
    # prefix first: popsign16nac must not fall through to popsign16 (the
    # NA-as-class label space has different head sizes).
    split_name = cfg["train_split"]
    maps_name = "label_maps.json"
    for pre in ("popsign16nac", "popsign16", "popsign6"):
        if split_name.startswith(pre):
            maps_name = f"{pre}_label_maps.json"
            break
    if maps_name != "label_maps.json":
        print(f"CORPUS: {split_name} -> {maps_name}")
    feature_order, n_classes = load_n_classes(maps_name)
    train_heads = None  # indices whose heads receive loss / drive early stop
    if cfg.get("keep_heads"):
        kept = resolve_heads(cfg["keep_heads"], feature_order)
        train_heads = [feature_order.index(n) for n in kept]
        print(f"HEAD ABLATION: training {list(kept)}, leaving "
              f"{[n for n in feature_order if n not in kept]} at random init")
    ds_cls = SemLexNpyDataset if args.no_pack else SemLexPackedDataset
    train_ds = ds_cls(cfg["train_split"])
    val_ds = ds_cls(cfg["val_split"])
    print(f"train={cfg['train_split']} ({len(train_ds)}) "
          f"val={cfg['val_split']} ({len(val_ds)})")

    # NOTE: protocol runs use NO gradient accumulation. The classification
    # head's BatchNorm1d normalizes over the batch, so accumulating 4x32 is
    # NOT equivalent to a true 128 batch (measured: identical loss in eval
    # mode, but training diverged from step ~50 and collapsed to
    # majority-class predictions). Memory is handled by
    # checkpoint_activations in load_backbone() instead. --grad-accum exists
    # solely for explicitly-marked diagnostic arms (2x64 test).
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True,  # BatchNorm1d crashes on B=1 in train mode
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate,
    )

    device = torch.device("cuda")
    keep_channels = None
    if cfg.get("keep_channels"):
        keep_channels = resolve_channels(cfg["keep_channels"])
        dropped = [c for c in ALL_CHANNELS if c not in keep_channels]
        print(f"ABLATION: keeping {list(keep_channels)}, "
              f"dropping {dropped} (slots filled with pretrained mask_emb)")
    model = PhonologyClassifier(
        load_backbone(
            checkpoint_activations=cfg.get("checkpoint_activations", False),
            random_init=cfg.get("random_init", False)),
        n_classes, keep_channels=keep_channels).to(device)
    # Discriminative LRs: head at cfg["lr"], pretrained backbone at 1/10 of it
    # (paper's LoRA-vs-head ratio; a flat 1e-4 collapses the encoder — see
    # config.yaml).
    backbone_lr = cfg.get("backbone_lr", cfg["lr"])
    # That 1/10 exists to keep PRETRAINED features from being destroyed. A
    # randomly initialised encoder has no features to protect and would simply
    # be crippled by the reduced rate, which would bias the ablation toward
    # "pretraining helps". Train it at the full head LR unless overridden.
    #
    # The guard must test the explicit FLAG, never cfg: config.yaml always
    # sets backbone_lr, so a cfg test never fires and a scratch arm would
    # train its random encoder at 1e-5 -- exactly the bias this branch
    # exists to avoid. Only an EXPLICIT --backbone-lr suppresses the raise.
    if cfg.get("random_init") and not cfg.get("backbone_lr_explicit"):
        backbone_lr = cfg["lr"]
        # Write it back so the checkpoint records the rate actually used --
        # otherwise cfg["backbone_lr"] keeps the unused 1e-5 and the run's
        # provenance contradicts its own training log.
        cfg["backbone_lr"] = backbone_lr
        print(f"RANDOM INIT: backbone_lr raised to head lr {backbone_lr} "
              f"(the 1/10 discriminative rate protects pretrained features, "
              f"which a random encoder does not have)")
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    assert len(backbone_params) + len(head_params) == len(list(model.parameters()))
    optimizer = torch.optim.Adam(
        [{"params": backbone_params, "lr": backbone_lr},
         {"params": head_params, "lr": cfg["lr"]}],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print(f"optimizer: backbone lr {backbone_lr:g} "
          f"({len(backbone_params)} tensors), head lr {cfg['lr']:g} "
          f"({len(head_params)} tensors)")
    amp_dtype = autocast_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    print(f"amp dtype: {amp_dtype}, "
          f"gpu: {torch.cuda.get_device_name(0)}")

    start_epoch, best_macro, best_epoch, patience_left = 0, -1.0, -1, cfg["patience"]
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if last_path.exists() and not args.fresh:
        state = torch.load(last_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_macro = state["best_macro"]
        best_epoch = state["best_epoch"]
        patience_left = state["patience_left"]
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        print(f"resumed from {last_path} at epoch {start_epoch} "
              f"(best {best_macro:.4f} @ {best_epoch})")
    elif log_path.exists():
        # --fresh (or a deleted checkpoint) restarts at epoch 0; the old log
        # must not be appended to, or the history reads as a non-monotonic
        # [0,1,2,0,1,2] and every downstream check misreads it.
        log_path.rename(log_path.with_suffix(".jsonl.bak"))
        print(f"fresh start: previous log moved to {log_path.name}.bak")

    n_feat = len(n_classes)
    loss_heads = train_heads if train_heads is not None else list(range(n_feat))
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.time()
        loss_sum, n_steps = 0.0, 0
        pending = 0  # micro-batches accumulated since the last optimizer step
        for step, batch in enumerate(train_loader):
            if args.limit_train_batches and step >= args.limit_train_batches:
                break
            batch = to_device(batch, device)
            if pending == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(batch["face"], batch["hand1"], batch["hand2"],
                               batch["body"], batch["padding_mask"],
                               batch["lengths"])
                # ignore_index=-1: PopSign clips can lack supervision for a
                # given feature (gloss absent from ASL-LEX, or a structurally
                # absent value). Sem-Lex never emits -1, so this is a no-op
                # there. A head whose entire batch is ignored returns NaN --
                # dropping those terms keeps one unsupervised head from
                # poisoning the whole summed loss.
                terms = [
                    F.cross_entropy(logits[i].float(), batch["labels"][:, i],
                                    ignore_index=-1)
                    for i in loss_heads
                ]
                terms = [t for t in terms if torch.isfinite(t)]
                if not terms:
                    continue
                loss = sum(terms) / len(terms)
            scaler.scale(loss / grad_accum).backward()
            pending += 1
            if pending == grad_accum:
                scaler.step(optimizer)
                scaler.update()
                pending = 0
            loss_sum += loss.item()
            n_steps += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}: loss {loss.item():.4f} "
                      f"({(step + 1) / (time.time() - t0):.2f} it/s)",
                      flush=True)
        train_loss = loss_sum / max(n_steps, 1)

        val = run_eval(model, val_loader, device, feature_order)
        # With --keep-heads, selection/early stop use the macro over the
        # TRAINED heads only; the full 16-head macro would be diluted by the
        # 10 random-init heads and never move.
        if train_heads is None:
            sel_macro = val["macro_recall1"]
        else:
            sel_macro = round(sum(
                val["per_feature_recall1"][feature_order[i]]
                for i in train_heads) / len(train_heads), 4)
        is_best = sel_macro > best_macro
        if is_best:
            best_macro, best_epoch = sel_macro, epoch
            patience_left = cfg["patience"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": val, "config": cfg,
                        "feature_order": feature_order,
                        "n_classes": n_classes}, best_path)
        else:
            patience_left -= 1

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch, "best_macro": best_macro,
            "best_epoch": best_epoch, "patience_left": patience_left,
            "config": cfg,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()},
        }, last_path)

        rec = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_macro_recall1": val["macro_recall1"],
               # with --keep-heads this (not the full macro) drives early stop
               **({"val_macro_kept": sel_macro} if train_heads is not None
                  else {}),
               # per-feature kept so check_smoke.py / analysis can detect
               # majority-class collapse without re-running eval
               "val_per_feature_recall1": val["per_feature_recall1"],
               "val_loss": val["loss"], "is_best": is_best,
               "best_macro": best_macro, "best_epoch": best_epoch,
               "seconds": round(time.time() - t0, 1)}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch}: train_loss {train_loss:.4f} "
              f"val macro R@1 {sel_macro:.4f} "
              f"(best {best_macro:.4f} @ {best_epoch}, "
              f"patience {patience_left}) [{rec['seconds']}s]", flush=True)

        if epoch + 1 >= cfg["min_epochs"] and patience_left <= 0:
            print(f"early stop at epoch {epoch}")
            break

    print(f"DONE: best val macro Recall@1 {best_macro:.4f} at epoch "
          f"{best_epoch} -> {best_path}")


if __name__ == "__main__":
    main()
