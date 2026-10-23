"""Materialise train/val/test splits + upstream-format label CSVs.

Restricts the deduped metadata to the frozen vocabulary and to clips whose
pose file exists, checks the paper's signer-independence claim for the test
split (report, don't crash), and writes:
  - results/splits.json                {split: [video_id, ...]}
  - <LABELS_DIR>/{split}_labels.csv    "video_id,label_int" no header — the
                                       exact file sign_gendata.py reads.
"""
import sys
from pathlib import Path

import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import LABELS_DIR, METADATA_DEDUPED_CSV, SPLITS_JSON, VOCAB_JSON

SPLITS = ["train", "val", "test"]


def main():
    df = pd.read_csv(METADATA_DEDUPED_CSV)
    vocab = json.loads(VOCAB_JSON.read_text())
    label_to_id = vocab["label_to_id"]

    # same row base as the frozen vocab rule (label types recorded in vocab.json)
    sel = df[df.label_type.isin(vocab["rule_def"]["label_types"]) & df.pose_ok
             & df.label.isin(label_to_id)]
    report = {"created_at": utcnow(), "vocab_sha256": vocab["vocab_sha256"],
              "n_selected": len(sel)}

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in SPLITS:
        part = sel[sel.split == split]
        splits[split] = part.video_id.tolist()
        report[f"n_{split}"] = len(part)
        report[f"n_classes_{split}"] = part.label.nunique()
        with open(LABELS_DIR / f"{split}_labels.csv", "w") as f:
            for r in part.itertuples():
                f.write(f"{r.video_id},{label_to_id[r.label]}\n")

    # signer independence of the test split (paper: 10 held-out signers)
    train_signers = set(sel[sel.split == "train"].signer_id)
    val_signers = set(sel[sel.split == "val"].signer_id)
    test_signers = set(sel[sel.split == "test"].signer_id)
    overlap = test_signers & (train_signers | val_signers)
    report["n_test_signers"] = len(test_signers)
    report["test_signer_overlap_with_train_val"] = sorted(overlap)
    if overlap:
        print(f"WARNING: test signers also in train/val: {sorted(overlap)}")
    else:
        print(f"test split signer-independent ({len(test_signers)} signers)")

    # classes with zero train examples can never be predicted — report them
    train_labels = set(sel[sel.split == "train"].label)
    unseen = [l for l in label_to_id if l not in train_labels]
    report["n_classes_without_train_examples"] = len(unseen)

    write_json(SPLITS_JSON, {"created_at": report["created_at"],
                             "vocab_sha256": vocab["vocab_sha256"],
                             **{s: splits[s] for s in SPLITS}})
    write_json(SPLITS_JSON.with_name("splits_report.json"), report)
    for k, v in report.items():
        if k != "test_signer_overlap_with_train_val" or v:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
