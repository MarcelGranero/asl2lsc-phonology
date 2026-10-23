"""Post-hoc scoring around upstream main.py — no upstream code involved.

Subcommands:
  find-best   Parse the training log (work_dir/<exp>/log.txt) for per-epoch
              val Top1 and print the checkpoint path of the best epoch
              (last stdout line, consumed by run_eval.sbatch).
  score       Load the score pickle upstream's test phase wrote
              (work_dir/<test_exp>/eval_results/*.pkl), compute top-1/3/5,
              per-signer and per-class breakdowns, and run_card.json under
              results/<run_name>/.

Differences from the SL-GCN Sem-Lex scorer:
  * signer comes from the CLIP ID (`<signer>-<class>-<timestamp>-<take>`),
    not from a metadata CSV — PopSign has no metadata file.
  * no paper-comparison table: there is no published SL-GCN PopSign number
    to compare against, so we report absolute metrics only. Inventing a
    baseline to compare to would be worse than reporting none.
  * per-signer spread is reported for BOTH val and test signers being
    unseen — every PopSign split is signer-disjoint from train.
"""
import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common import git_sha, sha256_file, utcnow, write_json, write_run_card
from data_config import (DATASET, GENDATA_DIR, PROCESSED_DIR, RESULTS_DIR,
                         SAM_SLR_V2_DIR, VOCAB_JSON)

WORK_DIR = PROCESSED_DIR / "work_dir"


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


def signer_of(clip_id: str) -> str:
    return clip_id.split("-")[0]


def score(args):
    eval_dir = WORK_DIR / args.test_exp / "eval_results"
    pkls = sorted(eval_dir.glob("epoch_*.pkl"),
                  key=lambda p: p.stat().st_mtime)
    assert pkls, f"no score pickles under {eval_dir}"
    score_pkl = pkls[-1]
    with open(score_pkl, "rb") as f:
        score_dict = pickle.load(f)

    with open(GENDATA_DIR / f"{args.split}_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    labels = np.array(labels)
    scores = np.stack([score_dict[n] for n in names])
    assert scores.shape[0] == len(labels), (scores.shape, len(labels))

    metrics = {f"top{k}": topk_acc(scores, labels, k) for k in (1, 3, 5)}
    metrics["n_eval"] = int(len(labels))
    metrics["split"] = args.split

    pred1 = scores.argmax(axis=1)
    correct1 = pred1 == labels
    correct3 = (np.argsort(-scores, axis=1)[:, :3] == labels[:, None]).any(1)

    # --- per-signer (every PopSign split is signer-disjoint from train) ---
    signers = np.array([signer_of(n) for n in names])
    rows = []
    for s in sorted(set(signers)):
        m = signers == s
        rows.append({"signer_id": s, "n": int(m.sum()),
                     "top1": float(correct1[m].mean()),
                     "top3": float(correct3[m].mean())})
    metrics["n_signers"] = len(rows)
    metrics["per_signer_top1_std"] = float(np.std([r["top1"] for r in rows]))
    metrics["per_signer_top1_min"] = float(min(r["top1"] for r in rows))
    metrics["per_signer_top1_max"] = float(max(r["top1"] for r in rows))

    # --- per-class (250 classes; surfaces the systematically hard signs) ---
    vocab = __import__("json").loads(VOCAB_JSON.read_text())
    id_to_label = {v: k for k, v in vocab["label_to_id"].items()}
    cls_rows = []
    for c in sorted(set(labels.tolist())):
        m = labels == c
        cls_rows.append({"label": id_to_label.get(c, str(c)), "label_id": int(c),
                         "n": int(m.sum()),
                         "top1": float(correct1[m].mean()),
                         "top3": float(correct3[m].mean())})
    cls_rows.sort(key=lambda r: r["top1"])
    metrics["n_classes_evaluated"] = len(cls_rows)
    metrics["worst_classes"] = [
        {k: r[k] for k in ("label", "n", "top1")} for r in cls_rows[:10]]
    metrics["best_classes"] = [
        {k: r[k] for k in ("label", "n", "top1")} for r in cls_rows[-10:]][::-1]

    # A model collapsed onto a few classes scores badly but LOOKS like a
    # normal bad model in top-1 alone; prediction entropy makes it obvious.
    pred_counts = Counter(pred1.tolist())
    metrics["n_distinct_predicted_classes"] = len(pred_counts)
    metrics["most_predicted_class_frac"] = float(
        max(pred_counts.values()) / len(pred1))

    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_signer_metrics.csv", rows,
               ["signer_id", "n", "top1", "top3"])
    _write_csv(out_dir / "per_class_metrics.csv", cls_rows,
               ["label", "label_id", "n", "top1", "top3"])
    write_json(out_dir / "metrics.json", metrics)

    upstream_sha = "unknown"
    try:
        import subprocess
        upstream_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SAM_SLR_V2_DIR, text=True
        ).strip()
    except Exception:
        pass
    write_run_card(out_dir, {
        "task": f"PopSign SL-GCN isolated-sign recognition ({args.split})",
        "dataset": DATASET["name"],
        "upstream_repo": "https://github.com/jackyjsy/SAM-SLR-v2",
        "upstream_sha": upstream_sha,
        "git_sha": git_sha(),
        "weights": str(args.weights) if args.weights else None,
        "weights_sha256": (sha256_file(Path(args.weights))
                           if args.weights else None),
        "score_pickle": str(score_pkl),
        "vocab_sha256": vocab["vocab_sha256"],
        "n_classes": vocab["n_classes"],
        "metrics": metrics,
        "finished_at": utcnow(),
    })

    lines = [
        f"# PopSign SL-GCN — {args.split} split",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Top-1 | {metrics['top1'] * 100:.2f}% |",
        f"| Top-3 | {metrics['top3'] * 100:.2f}% |",
        f"| Top-5 | {metrics['top5'] * 100:.2f}% |",
        f"| Clips | {metrics['n_eval']} |",
        f"| Classes | {metrics['n_classes_evaluated']} |",
        f"| Signers (all unseen in train) | {metrics['n_signers']} |",
        f"| Per-signer top-1 spread | "
        f"{metrics['per_signer_top1_min'] * 100:.1f}%–"
        f"{metrics['per_signer_top1_max'] * 100:.1f}% "
        f"(sd {metrics['per_signer_top1_std'] * 100:.2f} pts) |",
        f"| Distinct predicted classes | "
        f"{metrics['n_distinct_predicted_classes']}/250 |",
        "",
        "No published SL-GCN PopSign baseline exists, so these are absolute "
        "numbers with no paper comparison.",
        "",
        "Hardest classes (top-1): " + ", ".join(
            f"{r['label']} {r['top1'] * 100:.0f}%"
            for r in metrics["worst_classes"][:5]),
    ]
    table = "\n".join(lines)
    (out_dir / "results.md").write_text(table + "\n")
    print(table)


def _write_csv(path: Path, rows: list, cols: list):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("find-best")
    p1.add_argument("--exp", default="popsign_joint")
    p2 = sub.add_parser("score")
    p2.add_argument("--test-exp", default="popsign_joint_test")
    p2.add_argument("--run-name", default="slgcn_joint")
    p2.add_argument("--split", default="test", choices=["val", "test"])
    p2.add_argument("--weights", default=None,
                    help="checkpoint evaluated (recorded in the run card)")
    args = parser.parse_args()
    {"find-best": find_best, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
