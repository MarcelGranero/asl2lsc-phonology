"""Materialise train/val/test splits + upstream-format label CSVs.

PopSign ships its splits in the directory tree, so this script does not
partition anything — it restricts the manifest to clips that have BOTH a
frozen-vocab label and a landmark parquet, ASSERTS the signer-independence
that makes the released splits meaningful, and writes:
  - results/splits.json                {split: [clip_id, ...]}
  - <LABELS_DIR>/{split}_labels.csv    "clip_id,label_int" no header — the
                                       exact file sign_gendata.py reads.

Signer independence is an assert, not a warning (SEM-LEX's equivalent check
warns because the paper's claim was being VERIFIED; here it is a documented
property of the release, so a violation means our parsing broke).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import (CONVERT_FAILURES_JSON, LABELS_DIR, MANIFEST_CSV,
                         SPLITS_JSON, VOCAB_JSON)

SPLITS = ["train", "val", "test"]


def read_manifest():
    rows = []
    with open(MANIFEST_CSV) as f:
        cols = f.readline().strip().split(",")
        for line in f:
            rows.append(dict(zip(cols, line.rstrip("\n").split(","))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-failures", action="store_true",
        help="also drop clips listed in results/convert_failures.json "
             "(unreadable source parquet). Run this AFTER convert_poses.py, "
             "then re-run run_gendata.py.")
    args = parser.parse_args()

    rows = read_manifest()
    vocab = json.loads(VOCAB_JSON.read_text())
    label_to_id = vocab["label_to_id"]

    unreadable = set()
    if args.exclude_failures and CONVERT_FAILURES_JSON.exists():
        fail = json.loads(CONVERT_FAILURES_JSON.read_text())
        unreadable = {c for s in SPLITS for c in fail.get(s, [])}
        print(f"excluding {len(unreadable)} clip(s) with unreadable parquet")

    kept = [r for r in rows
            if r["has_landmarks"] == "1" and r["label"] in label_to_id
            and r["clip_id"] not in unreadable]
    dropped = len(rows) - len(kept)
    report = {"created_at": utcnow(), "vocab_sha256": vocab["vocab_sha256"],
              "n_manifest": len(rows), "n_selected": len(kept),
              "n_dropped_no_landmarks_or_oov": dropped,
              "n_dropped_unreadable_parquet": len(unreadable)}

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    splits, signers = {}, {}
    for split in SPLITS:
        part = [r for r in kept if r["split"] == split]
        splits[split] = [r["clip_id"] for r in part]
        signers[split] = {r["signer_id"] for r in part}
        report[f"n_{split}"] = len(part)
        report[f"n_classes_{split}"] = len({r["label"] for r in part})
        report[f"n_signers_{split}"] = len(signers[split])
        with open(LABELS_DIR / f"{split}_labels.csv", "w") as f:
            for r in part:
                f.write(f"{r['clip_id']},{label_to_id[r['label']]}\n")

    overlap_val = sorted(signers["train"] & signers["val"])
    overlap_test = sorted(signers["train"] & signers["test"])
    report["train_val_signer_overlap"] = overlap_val
    report["train_test_signer_overlap"] = overlap_test
    if overlap_val or overlap_test:
        print(f"FATAL: signer overlap — val {overlap_val}, test {overlap_test}")
        sys.exit(1)
    print(f"splits signer-independent (train {len(signers['train'])} / "
          f"val {len(signers['val'])} / test {len(signers['test'])} signers)")

    train_labels = {r["label"] for r in kept if r["split"] == "train"}
    report["n_classes_without_train_examples"] = len(
        [l for l in label_to_id if l not in train_labels])

    write_json(SPLITS_JSON, {"created_at": report["created_at"],
                             "vocab_sha256": vocab["vocab_sha256"],
                             **{s: splits[s] for s in SPLITS}})
    write_json(SPLITS_JSON.with_name("splits_report.json"), report)
    for k, v in report.items():
        if not k.endswith("_overlap") or v:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
