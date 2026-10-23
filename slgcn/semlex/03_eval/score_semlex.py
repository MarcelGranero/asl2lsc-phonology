"""Post-hoc scoring around upstream main.py — no upstream code involved.

Subcommands:
  find-best   Parse the training log (work_dir/<exp>/log.txt) for per-epoch
              val Top1 and print the checkpoint path of the best epoch
              (last stdout line, consumed by run_eval.sbatch).
  score       Load the score pickle upstream's test phase wrote
              (work_dir/<test_exp>/eval_results/*.pkl), compute top-1/3/5,
              per-signer breakdown, run_card.json, and the paper-comparison
              table under results/<run_name>/.
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common import git_sha, sha256_file, utcnow, write_json, write_run_card
from data_config import (GENDATA_DIR, METADATA_DEDUPED_CSV, PAPER,
                         PROCESSED_DIR, RESULTS_DIR, SAM_SLR_V2_DIR,
                         VOCAB_JSON)

WORK_DIR = PROCESSED_DIR / "work_dir"


TRAIN_CONFIG = REPO / "02_train" / "config" / "train_joint_semlex.yaml"


def default_train_exp() -> str:
    """Experiment_name from the training config (fallback: semlex_joint_paper)."""
    try:
        import yaml
        return yaml.safe_load(TRAIN_CONFIG.read_text())["Experiment_name"]
    except Exception:
        return "semlex_joint_paper"


def find_best(args):
    log = WORK_DIR / args.exp / "log.txt"
    text = log.read_text()
    pairs = re.findall(r"Eval epoch: (\d+).*?Top1: ([\d.]+)%", text, re.S)
    if not pairs:
        sys.exit(f"no eval lines found in {log}")
    # log 'Eval epoch: N' is 1-based; checkpoint files are 0-based (N-1)
    best_epoch1, best_acc = max(((int(e), float(a)) for e, a in pairs),
                                key=lambda x: x[1])
    ckpt = PROCESSED_DIR / "save_models" / f"{args.exp}-{best_epoch1 - 1}.pt"
    assert ckpt.exists(), ckpt
    print(f"best val Top1 {best_acc:.2f}% at eval epoch {best_epoch1} "
          f"({len(pairs)} epochs evaluated)", file=sys.stderr)
    print(ckpt)


def topk_acc(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def score(args):
    eval_dir = WORK_DIR / args.test_exp / "eval_results"
    pkls = sorted(eval_dir.glob("epoch_*.pkl"),
                  key=lambda p: p.stat().st_mtime)
    assert pkls, f"no score pickles under {eval_dir}"
    score_pkl = pkls[-1]
    with open(score_pkl, "rb") as f:
        score_dict = pickle.load(f)

    with open(GENDATA_DIR / "test_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    labels = np.array(labels)
    scores = np.stack([score_dict[n] for n in names])
    assert scores.shape[0] == len(labels)

    metrics = {f"top{k}": topk_acc(scores, labels, k) for k in (1, 3, 5)}
    metrics["n_test"] = int(len(labels))

    # per-signer breakdown (test split is the demographic-bias split)
    meta = pd.read_csv(METADATA_DEDUPED_CSV)[["video_id", "signer_id"]]
    signer = meta.set_index("video_id").signer_id.reindex(names).values
    pred1 = scores.argmax(axis=1)
    per_signer = (pd.DataFrame({
        "signer_id": signer, "correct": pred1 == labels,
        "correct3": (np.argsort(-scores, axis=1)[:, :3]
                     == labels[:, None]).any(axis=1)})
        .groupby("signer_id")
        .agg(n=("correct", "size"), top1=("correct", "mean"),
             top3=("correct3", "mean"))
        .reset_index())
    metrics["per_signer_top1_std"] = float(per_signer.top1.std())

    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_signer.to_csv(out_dir / "per_signer_metrics.csv", index=False)
    write_json(out_dir / "metrics.json", metrics)

    import json
    vocab = json.loads(VOCAB_JSON.read_text())
    upstream_sha = "unknown"
    try:
        import subprocess
        upstream_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SAM_SLR_V2_DIR, text=True
        ).strip()
    except Exception:
        pass
    write_run_card(out_dir, {
        "task": "SEM-LEX SL-GCN gloss-only ISR reproduction",
        "paper": "Kezar et al., ASSETS'23, arXiv 2310.00196",
        "git_sha": git_sha(),
        "upstream_repo": "https://github.com/jackyjsy/SAM-SLR-v2",
        "upstream_sha": upstream_sha,
        "weights": str(args.weights) if args.weights else None,
        "weights_sha256": sha256_file(Path(args.weights)) if args.weights else None,
        "score_pickle": str(score_pkl),
        "vocab_sha256": vocab["vocab_sha256"],
        "n_classes": vocab["n_classes"],
        "metrics": metrics,
        "finished_at": utcnow(),
    })

    rows = [
        ("Top-1", PAPER["gloss_only_top1"], metrics["top1"]),
        ("Top-3", PAPER["gloss_only_top3"], metrics["top3"]),
    ]
    lines = ["| Metric | Paper (gloss-only) | This reproduction | Δ (pts) |",
             "|---|---|---|---|"]
    for name, paper_v, ours in rows:
        lines.append(f"| {name} | {paper_v * 100:.1f}% | {ours * 100:.2f}% "
                     f"| {(ours - paper_v) * 100:+.2f} |")
    lines.append(f"\nTop-5: {metrics['top5'] * 100:.2f}% | test clips: "
                 f"{metrics['n_test']} | per-signer top-1 std: "
                 f"{metrics['per_signer_top1_std'] * 100:.2f} pts")
    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n")
    print(table)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("find-best")
    # default read from the training config, never hardcoded: a renamed
    # run otherwise sends find-best at a work_dir that no longer exists.
    p1.add_argument("--exp", default=default_train_exp())
    p2 = sub.add_parser("score")
    p2.add_argument("--test-exp", default="semlex_joint_test")
    p2.add_argument("--run-name", default="slgcn_joint")
    p2.add_argument("--weights", default=None,
                    help="checkpoint evaluated (recorded in the run card)")
    args = parser.parse_args()
    {"find-best": find_best, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
