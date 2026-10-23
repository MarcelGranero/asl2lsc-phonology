"""Phase-2 training: gloss + phonological-feature heads on SL-GCN.

A SEPARATE training loop, never an edit to the upstream clone.  Upstream's
`main.py` assumes one head and a scalar label, so it cannot express multi-head
batches; rather than patch it (which would violate the repo's least-invasive
rule) this script reimplements just the loop, reusing upstream's feeder,
model, graph and optimiser settings unchanged.

Two regimes, matching the paper:

  --regime pfr    §5.2.  Phonological feature recognition.  "All models are
                  SL-GCNs pre-trained to predict gloss and then trained to
                  predict phonological feature types" — so this REQUIRES
                  --init-from <gloss checkpoint>.  No gloss head.
                  Reproduces Table 5's "Multitask" column (all 16 at once).

  --regime isr_pfr §5.3.  Gloss + all 16 features trained jointly, from
                  scratch by default.  Reproduces the ISR+PFR row of Table 4
                  (68.6% top-1 test).

The LR schedule, optimiser, warmup and augmentation mirror
02_train/config/train_joint_paper.yaml (upstream's WLASL recipe), so phase-2
runs stay comparable to the phase-1 gloss run.
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_sha, utcnow, write_json, write_run_card  # noqa: E402
from data_config import (FEATURE_LABELS_PKL, FEATURE_VOCAB_JSON,  # noqa: E402
                         GENDATA_DIR, PROCESSED_DIR, RESULTS_DIR, VOCAB_JSON)

WORK_DIR = PROCESSED_DIR / "work_dir"
SAVE_DIR = PROCESSED_DIR / "save_models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg, feature_label_path, batch_size, test_batch_size, workers,
                  debug=False):
    from feeder_multihead import FeederMultihead

    train_args = dict(cfg["train_feeder_args"])
    test_args = dict(cfg["test_feeder_args"])
    if debug:
        train_args["debug"] = True
        test_args["debug"] = True

    train_set = FeederMultihead(feature_label_path=feature_label_path, **train_args)
    val_set = FeederMultihead(feature_label_path=feature_label_path, **test_args)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=test_batch_size, shuffle=False, num_workers=workers,
        pin_memory=True,
    )
    return train_set, val_set, train_loader, val_loader


def load_gloss_checkpoint(model, path, with_gloss):
    """Load a phase-1 gloss checkpoint into the multi-head model.

    Shared weights (data_bn, l1-l10) must match exactly — a silent partial
    load here would produce a randomly-initialised encoder and a plausible
    but meaningless result, so anything unexpected is a hard error.
    """
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith(("l1.", "data_bn.")) for k in state
    ):
        state = state["model"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    own = model.state_dict()
    encoder_keys = [k for k in own if not k.startswith("feature_heads.")
                    and not k.startswith("fc.")]
    missing = [k for k in encoder_keys if k not in state]
    if missing:
        raise SystemExit(
            f"checkpoint {path} is missing {len(missing)} encoder weights, "
            f"e.g. {missing[:3]} — wrong checkpoint or wrong architecture"
        )
    mismatched = [k for k in encoder_keys if state[k].shape != own[k].shape]
    if mismatched:
        raise SystemExit(f"shape mismatch on {mismatched[:3]} — check num_point/groups")

    to_load = {k: state[k] for k in encoder_keys}
    n_gloss = 0
    if with_gloss and "fc.weight" in state:
        if state["fc.weight"].shape == own["fc.weight"].shape:
            to_load["fc.weight"] = state["fc.weight"]
            to_load["fc.bias"] = state["fc.bias"]
            n_gloss = 2
        else:
            print("[init] gloss head shape differs — leaving fc randomly initialised")
    model.load_state_dict(to_load, strict=False)
    print(f"[init] loaded {len(encoder_keys)} encoder tensors "
          f"+ {n_gloss} gloss-head tensors from {Path(path).name}")


def lr_at(epoch, base_lr, steps, warm_up_epoch):
    if epoch < warm_up_epoch:
        return base_lr * (epoch + 1) / warm_up_epoch
    decay = sum(1 for s in steps if epoch >= s)
    return base_lr * (0.1 ** decay)


def evaluate(model, loader, feature_names, device, with_gloss, topk=(1, 3, 5)):
    """Val pass: gloss top-k plus per-feature top-1, ignoring masked targets."""
    model.eval()
    gloss_hits = {k: 0 for k in topk}
    gloss_total = 0
    feat_hits = defaultdict(int)
    feat_total = defaultdict(int)

    with torch.no_grad():
        for data, label, feat_label, _ in loader:
            data = data.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            feat_label = feat_label.long().to(device, non_blocking=True)

            gloss_logits, feature_logits = model(data)

            if with_gloss and gloss_logits is not None:
                gloss_total += label.numel()
                maxk = max(topk)
                ranked = gloss_logits.topk(maxk, dim=1).indices
                correct = ranked.eq(label.view(-1, 1))
                for k in topk:
                    gloss_hits[k] += int(correct[:, :k].any(dim=1).sum())

            for i, logits in enumerate(feature_logits):
                target = feat_label[:, i]
                valid = target != -1
                if not bool(valid.any()):
                    continue
                pred = logits.argmax(dim=1)
                feat_hits[i] += int((pred[valid] == target[valid]).sum())
                feat_total[i] += int(valid.sum())

    metrics = {}
    if gloss_total:
        for k in topk:
            metrics[f"gloss_top{k}"] = gloss_hits[k] / gloss_total
    per_feature = {
        feature_names[i]: feat_hits[i] / feat_total[i]
        for i in sorted(feat_total)
        if feat_total[i]
    }
    metrics["feature_top1"] = per_feature
    if per_feature:
        metrics["feature_top1_avg"] = sum(per_feature.values()) / len(per_feature)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=["pfr", "isr_pfr"], required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "02_train/config/train_joint_paper.yaml",
                    help="phase-1 config; feeder/model/optimiser args are reused")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="phase-1 gloss checkpoint (REQUIRED for --regime pfr)")
    ap.add_argument("--feature-labels", type=Path, default=FEATURE_LABELS_PKL)
    ap.add_argument("--feature-vocab", type=Path, default=FEATURE_VOCAB_JSON)
    ap.add_argument("--num-epoch", type=int, default=None,
                    help="epochs to run IN THIS JOB (a resume passes the "
                         "remaining count, not the total)")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="absolute 0-based epoch this job starts at. REQUIRED "
                         "on a resume: the LR schedule, checkpoint names and "
                         "logs are all keyed on absolute epoch, so leaving it "
                         "at 0 restarts the warmup and re-measures "
                         "step:[150,200] from the resume.")
    ap.add_argument("--base-lr", type=float, default=None,
                    help="default: config base_lr (0.1) for isr_pfr; "
                         "base_lr/10 for pfr, which fine-tunes a pretrained "
                         "encoder. Pass explicitly to override either.")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--warm-up-epoch", type=int, default=None,
                    help="default: config value (20). The config's ramp is "
                         "sized for a 250-epoch from-scratch run; shorten it "
                         "for short fine-tune runs.")
    ap.add_argument("--gloss-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-interval", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 100 samples, results under results/_smoke/")
    args = ap.parse_args()

    if args.regime == "pfr" and args.init_from is None:
        ap.error(
            "--regime pfr reproduces the paper's §5.2 setup, where feature heads "
            "are trained on top of a gloss-pretrained encoder; pass --init-from "
            "<checkpoint>. Use --regime isr_pfr to train jointly from scratch."
        )

    set_seed(args.seed)
    cfg = yaml.safe_load(os.path.expandvars(args.config.read_text()))

    exp_name = args.exp_name or f"semlex_{args.regime}"
    if args.smoke:
        exp_name = f"{exp_name}_smoke"

    feature_vocab = json.loads(args.feature_vocab.read_text())
    feature_names = feature_vocab["features"]
    feature_sizes = [feature_vocab["per_feature"][f]["n_classes"] for f in feature_names]

    batch_size = args.batch_size or cfg["batch_size"]
    test_batch_size = cfg.get("test_batch_size", batch_size)
    num_epoch = args.num_epoch or cfg["num_epoch"]
    start_epoch = args.start_epoch
    if args.base_lr is not None:
        base_lr = args.base_lr
    elif args.regime == "pfr":
        # §5.2 trains feature heads on top of a gloss-pretrained encoder,
        # "fine-tuning the entire model" — the encoder is NOT frozen. The
        # paper gives no LR (it gives no hyperparameters at all), but the
        # phase-1 recipe's 0.1 is a from-scratch LR: applied to a converged
        # encoder it destroys the pretrained representation in the first few
        # steps, which is the very thing this regime is meant to exploit.
        # base_lr/10 is the conventional fine-tuning default; recorded in the
        # run card and overridable with --base-lr.
        base_lr = cfg["base_lr"] / 10
    else:
        base_lr = cfg["base_lr"]
    steps = cfg["step"]
    warm_up_epoch = (args.warm_up_epoch if args.warm_up_epoch is not None
                     else cfg.get("warm_up_epoch", 0))
    if args.smoke:
        num_epoch, warm_up_epoch, start_epoch = 2, 1, 0

    train_set, val_set, train_loader, val_loader = build_loaders(
        cfg, str(args.feature_labels), batch_size, test_batch_size,
        args.workers, debug=args.smoke,
    )

    from model_multihead import MultiheadModel, multihead_loss

    with_gloss = args.regime == "isr_pfr"
    model_args = dict(cfg["model_args"])
    n_classes = json.loads(VOCAB_JSON.read_text())["n_classes"]
    if model_args["num_class"] != n_classes:
        raise SystemExit(
            f"config num_class={model_args['num_class']} != vocab n_classes={n_classes}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiheadModel(
        feature_sizes=feature_sizes, with_gloss=with_gloss, **model_args
    ).to(device)

    if args.init_from:
        load_gloss_checkpoint(model, args.init_from, with_gloss)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9,
        nesterov=cfg.get("nesterov", True), weight_decay=cfg["weight_decay"],
    )

    work = WORK_DIR / exp_name
    work.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = work / "log.txt"

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"regime={args.regime} exp={exp_name} device={device}")
    log(f"{len(feature_names)} feature heads, sizes {feature_sizes}")
    log(f"epochs {start_epoch}..{start_epoch + num_epoch - 1} absolute "
        f"(lr schedule keyed on absolute epoch)")
    log(f"train={len(train_set)} val={len(val_set)} "
        f"batch={batch_size} epochs={num_epoch} base_lr={base_lr}")
    if train_set.n_missing_feature_labels:
        log(f"WARNING {train_set.n_missing_feature_labels} train samples lack "
            f"feature labels")

    best_key = "gloss_top1" if with_gloss else "feature_top1_avg"
    best_value, best_epoch, history = -1.0, -1, []

    # `epoch` is ABSOLUTE (0-based over the whole run), not the index within
    # this job. It drives three things that must all agree across a resume:
    # the LR schedule, the checkpoint filename, and the logged epoch number.
    # Restarting it at 0 on resume would re-run the warmup, measure
    # step:[150,200] from the resume, and overwrite earlier checkpoints.
    for epoch in range(start_epoch, start_epoch + num_epoch):
        model.train()
        lr = lr_at(epoch, base_lr, steps, warm_up_epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        running, n_batches = 0.0, 0
        for data, label, feat_label, _ in train_loader:
            data = data.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            feat_label = feat_label.long().to(device, non_blocking=True)

            gloss_logits, feature_logits = model(
                data, keep_prob=cfg.get("keep_rate", 0.9)
            )
            loss, _ = multihead_loss(
                gloss_logits, feature_logits,
                label if with_gloss else None, feat_label,
                gloss_weight=args.gloss_weight,
                feature_weight=args.feature_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)
            n_batches += 1

        log(f"Epoch {epoch + 1}/{num_epoch} lr={lr:.5f} "
            f"train_loss={running / max(n_batches, 1):.4f}")

        if (epoch + 1) % args.eval_interval == 0 or epoch == num_epoch - 1:
            metrics = evaluate(model, val_loader, feature_names, device, with_gloss)
            entry = {"epoch": epoch + 1, "lr": lr,
                     "train_loss": running / max(n_batches, 1), **metrics}
            history.append(entry)
            summary = (f"feat_avg={metrics.get('feature_top1_avg', 0):.4f}")
            if with_gloss:
                summary = f"Top1={metrics.get('gloss_top1', 0):.4f} " + summary
            log(f"Eval epoch: {epoch + 1} {summary}")

            value = metrics.get(best_key, -1.0)
            if value > best_value:
                best_value, best_epoch = value, epoch + 1
                torch.save(model.state_dict(), SAVE_DIR / f"{exp_name}-best.pt")
                log(f"  new best {best_key}={value:.4f} -> {exp_name}-best.pt")

        torch.save(model.state_dict(), SAVE_DIR / f"{exp_name}-{epoch}.pt")

    out_dir = (RESULTS_DIR / "_smoke" / exp_name) if args.smoke else (
        RESULTS_DIR / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = history[-1] if history else {}
    write_json(out_dir / "metrics.json", {
        "regime": args.regime,
        "best_epoch": best_epoch,
        "best_metric": best_key,
        "best_value": best_value,
        "final_eval": final,
        "history": history,
    })
    write_run_card(out_dir, {
        "git_sha": git_sha(),
        "vocab_sha": json.loads(VOCAB_JSON.read_text())["vocab_sha256"],
        "feature_vocab": {
            "n_features": len(feature_names),
            "null_policy": feature_vocab["null_policy"],
            "sizes": dict(zip(feature_names, feature_sizes)),
        },
        "config": {
            "regime": args.regime, "config_file": str(args.config),
            "init_from": str(args.init_from) if args.init_from else None,
            "num_epoch": num_epoch, "start_epoch": start_epoch,
            "end_epoch_exclusive": start_epoch + num_epoch,
            "base_lr": base_lr, "step": steps,
            "batch_size": batch_size, "warm_up_epoch": warm_up_epoch,
            "gloss_weight": args.gloss_weight,
            "feature_weight": args.feature_weight,
            "seed": args.seed, "smoke": args.smoke,
        },
        "metrics": {"best_value": best_value, "best_epoch": best_epoch, **final},
        "started_at": history[0]["epoch"] if history else None,
        "finished_at": utcnow(),
    })
    log(f"done. best {best_key}={best_value:.4f} @ epoch {best_epoch}; wrote {out_dir}")


if __name__ == "__main__":
    main()
