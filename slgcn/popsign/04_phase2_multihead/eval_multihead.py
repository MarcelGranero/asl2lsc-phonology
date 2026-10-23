"""Score a phase-2 checkpoint on the test split; reproduce the paper's Table 5.

Runs the multi-head model over the PopSign test split and reports:
  * per-feature top-1 accuracy, laid out like Table 5 with the paper's own
    numbers alongside ours (its "Multitask" column is the comparable one —
    all 16 features learned at once, which is what --regime pfr does);
  * the macro average over features (paper: 0.850 multitask / 0.858 fine-tune);
  * gloss top-1/3/5 when the checkpoint has a gloss head (--regime isr_pfr),
    comparable to the ISR+PFR row of Table 4 (68.6% top-1).

Masked targets (-1) are excluded per feature, so a feature's denominator is
its number of labelled test items, not the split size.
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_sha, utcnow, write_json, write_run_card  # noqa: E402
from data_config import (FEATURE_LABELS_PKL, FEATURE_VOCAB_JSON,  # noqa: E402
                         GENDATA_DIR, RESULTS_DIR, VOCAB_JSON)

# Table 5, "Multitask" and "Fine-Tune" columns (Kezar et al., ASSETS'23).
PAPER_TABLE5 = {
    "Major Location": (0.877, 0.875),
    "Minor Location": (0.792, 0.781),
    "Second Minor Location": (0.787, 0.772),
    "Contact": (0.893, 0.886),
    "Thumb Contact": (0.917, 0.911),
    "Sign Type": (0.889, 0.879),
    "Repeated Movement": (0.855, 0.854),
    "Path Movement": (0.756, 0.754),
    "Wrist Twist": (0.924, 0.926),
    "Selected Fingers": (0.911, 0.902),
    "Thumb Position": (0.915, 0.915),
    "Flexion": (0.812, 0.810),
    "Spread": (0.884, 0.880),
    "Spread Change": (0.903, 0.895),
    "Nondominant Handshape": (0.835, 0.817),
    "Handshape": (0.774, 0.747),
}
PAPER_AVG = {"fine_tune": 0.858, "multitask": 0.850}


def majority_baseline(feature_names, feature_label_path):
    """Per-feature accuracy of always predicting the most frequent TRAIN value.

    Fitted on train and scored on test, exactly like the model — so the two
    numbers are directly comparable. Masked (-1) targets are excluded on both
    sides, matching the model's denominator.
    """
    # Two sidecar schemas exist, same as in feeder_multihead.py:
    # build_feature_labels.py (6 features) wraps the map as
    # {"features": [...], "labels": {clip: vec}}; build_feature_labels16.py
    # writes the bare {clip: vec} map. Accept both, as the feeder does.
    with open(feature_label_path, "rb") as f:
        payload = pickle.load(f)
    lookup = (payload["labels"] if isinstance(payload.get("labels"), dict)
              else payload)
    with open(GENDATA_DIR / "train_label.pkl", "rb") as f:
        train_names, _ = pickle.load(f)
    with open(GENDATA_DIR / "test_label.pkl", "rb") as f:
        test_names, _ = pickle.load(f)

    train = np.array([lookup[n] for n in train_names if n in lookup])
    test = np.array([lookup[n] for n in test_names if n in lookup])
    if train.size == 0 or test.size == 0:
        return {}

    out = {}
    for i, name in enumerate(feature_names):
        col = train[:, i]
        col = col[col != -1]
        if col.size == 0:
            continue
        values, counts = np.unique(col, return_counts=True)
        majority = values[counts.argmax()]
        target = test[:, i]
        valid = target != -1
        if not valid.any():
            continue
        out[name] = float((target[valid] == majority).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--regime", choices=["pfr", "isr_pfr"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "02_train/config/train_joint_popsign.yaml")
    ap.add_argument("--feature-labels", type=Path, default=FEATURE_LABELS_PKL)
    ap.add_argument("--feature-vocab", type=Path, default=FEATURE_VOCAB_JSON)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(os.path.expandvars(args.config.read_text()))

    feature_vocab = json.loads(args.feature_vocab.read_text())
    feature_names = feature_vocab["features"]
    # per_feature is the 6-set schema; the 16-set vocab only carries the raw
    # maps (same fallback as train_multihead.py).
    if "per_feature" in feature_vocab:
        feature_sizes = [feature_vocab["per_feature"][f]["n_classes"]
                         for f in feature_names]
    else:
        feature_sizes = [len(feature_vocab["vocab"][f])
                         for f in feature_names]

    from feeder_multihead import FeederMultihead
    from model_multihead import MultiheadModel

    test_args = dict(cfg["test_feeder_args"])
    test_args["data_path"] = str(GENDATA_DIR / "test_data_joint.npy")
    test_args["label_path"] = str(GENDATA_DIR / "test_label.pkl")
    test_set = FeederMultihead(
        feature_label_path=str(args.feature_labels), **test_args
    )
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    with_gloss = args.regime == "isr_pfr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiheadModel(
        feature_sizes=feature_sizes, with_gloss=with_gloss, **cfg["model_args"]
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    gloss_hits = {1: 0, 3: 0, 5: 0}
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
                ranked = gloss_logits.topk(5, dim=1).indices
                correct = ranked.eq(label.view(-1, 1))
                for k in gloss_hits:
                    gloss_hits[k] += int(correct[:, :k].any(dim=1).sum())

            for i, logits in enumerate(feature_logits):
                target = feat_label[:, i]
                valid = target != -1
                if not bool(valid.any()):
                    continue
                pred = logits.argmax(dim=1)
                feat_hits[i] += int((pred[valid] == target[valid]).sum())
                feat_total[i] += int(valid.sum())

    # Majority-class baseline, per feature: the accuracy of always predicting
    # the most frequent TRAIN value. Several features are heavily skewed
    # (Wrist Twist is 82% one value), so a head can look respectable inside
    # the macro average while having learned nothing. Reporting the margin
    # over this baseline is what separates learning from class priors.
    baseline = majority_baseline(feature_names, args.feature_labels)

    per_feature = {}
    for i in sorted(feat_total):
        name = feature_names[i]
        top1 = feat_hits[i] / feat_total[i]
        base = baseline.get(name)
        per_feature[name] = {
            "top1": top1,
            "majority_baseline": base,
            "margin_over_baseline": (top1 - base) if base is not None else None,
            "n": feat_total[i],
            "n_classes": feature_sizes[i],
            "paper_fine_tune": PAPER_TABLE5.get(name, (None, None))[0],
            "paper_multitask": PAPER_TABLE5.get(name, (None, None))[1],
        }
    macro = float(np.mean([v["top1"] for v in per_feature.values()]))
    macro_baseline = float(np.mean(
        [v["majority_baseline"] for v in per_feature.values()
         if v["majority_baseline"] is not None]
    ))

    metrics = {
        "regime": args.regime,
        "checkpoint": str(args.checkpoint),
        "n_test": len(test_set),
        "feature_top1": per_feature,
        "feature_top1_avg": macro,
        "feature_majority_baseline_avg": macro_baseline,
        "feature_margin_over_baseline_avg": macro - macro_baseline,
        "paper_feature_avg": PAPER_AVG,
    }
    if gloss_total:
        for k in gloss_hits:
            metrics[f"gloss_top{k}"] = gloss_hits[k] / gloss_total

    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_run_card(out_dir, {
        "git_sha": git_sha(),
        "vocab_sha": json.loads(VOCAB_JSON.read_text())["vocab_sha256"],
        "config": {"regime": args.regime, "checkpoint": str(args.checkpoint),
                   "n_features": len(feature_names)},
        "metrics": {"feature_top1_avg": macro,
                    **{k: v for k, v in metrics.items() if k.startswith("gloss_")}},
        "finished_at": utcnow(),
    })

    lines = [
        f"# Phase 2 ({args.regime}) vs Kezar et al. Table 5",
        "",
        f"Checkpoint: `{args.checkpoint.name}`  ·  test n={len(test_set)}",
        "",
        "`majority` = always predict the most frequent train value; `Δ` = ours "
        "− majority. A head at or below its majority baseline has learned "
        "nothing, however high its raw accuracy looks.",
        "",
        "| Phonological Feature | ours | majority | Δ | paper (multitask) | paper (fine-tune) | classes | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in feature_names:
        v = per_feature.get(name)
        if v is None:
            continue
        mt = f"{v['paper_multitask']:.3f}" if v["paper_multitask"] else "—"
        ft = f"{v['paper_fine_tune']:.3f}" if v["paper_fine_tune"] else "—"
        base = v["majority_baseline"]
        base_s = f"{base:.3f}" if base is not None else "—"
        margin = v["margin_over_baseline"]
        margin_s = f"{margin:+.3f}" if margin is not None else "—"
        lines.append(
            f"| {name} | {v['top1']:.3f} | {base_s} | {margin_s} | {mt} | {ft} | "
            f"{v['n_classes']} | {v['n']} |"
        )
    lines.append(
        f"| **Average** | **{macro:.3f}** | {macro_baseline:.3f} | "
        f"{macro - macro_baseline:+.3f} | {PAPER_AVG['multitask']:.3f} | "
        f"{PAPER_AVG['fine_tune']:.3f} | | |"
    )
    if gloss_total:
        lines += ["", "## Gloss (ISR+PFR, cf. Table 4: 68.6% top-1 / 82.0% top-3)", ""]
        lines.append(f"- top-1 {metrics['gloss_top1']:.4f}")
        lines.append(f"- top-3 {metrics['gloss_top3']:.4f}")
        lines.append(f"- top-5 {metrics['gloss_top5']:.4f}")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
