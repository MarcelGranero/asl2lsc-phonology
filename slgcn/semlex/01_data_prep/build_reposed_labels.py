"""Label CSVs for the re-posed run, restricted to clips that actually re-posed.

2,401 train clips (6.4%) in results/splits.json have NO source .webm on disk,
so they cannot be re-extracted. Upstream `sign_gendata.py` np.loads every path
in the label CSV and dies on the first missing file, so the re-posed run needs
its own label CSVs listing only available clips.

VAL AND TEST ARE COMPLETE — the eval protocol is identical to the released-pose
run, and only the training set shrinks. That keeps the two runs comparable on
the axis that matters (same test clips, same labels, same vocabulary); the
training-set delta is the confound to report.

The vocabulary is NOT re-derived: it stays frozen at the 1,914 classes from
results/vocab.json so class ids match phase 1 and the trained heads stay
comparable. One class loses all its train examples (it keeps its id and simply
cannot be predicted) — recorded in the report.

Writes <LABELS_DIR>_reposed/{split}_labels.csv + results/reposed_labels_report.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import (LABELS_DIR, METADATA_DEDUPED_CSV, PROCESSED_DIR,
                         RESULTS_DIR, SPLITS_JSON, VOCAB_JSON)

REPOSED_POSES = PROCESSED_DIR / "reposed"
REPOSED_LABELS = Path(str(LABELS_DIR) + "_reposed")


def main():
    splits = json.loads(SPLITS_JSON.read_text())
    vocab = json.loads(VOCAB_JSON.read_text())
    label_to_id = vocab["label_to_id"]
    meta = pd.read_csv(METADATA_DEDUPED_CSV).set_index("video_id").label

    REPOSED_LABELS.mkdir(parents=True, exist_ok=True)
    report = {"created_at": utcnow(),
              "vocab_sha256": vocab["vocab_sha256"],
              "n_classes": vocab["n_classes"],
              "note": "vocabulary frozen from the released-pose run so class "
                      "ids stay comparable; only clip availability differs"}

    for split in ["train", "val", "test"]:
        wanted = splits[split]
        have = {p.stem for p in (REPOSED_POSES / split).glob("*.npy")}
        kept = [v for v in wanted if v in have]
        dropped = len(wanted) - len(kept)
        with open(REPOSED_LABELS / f"{split}_labels.csv", "w") as f:
            for v in kept:
                f.write(f"{v},{label_to_id[meta[v]]}\n")
        report[f"n_{split}_wanted"] = len(wanted)
        report[f"n_{split}_kept"] = len(kept)
        report[f"n_{split}_dropped"] = dropped
        report[f"pct_{split}_dropped"] = round(100 * dropped / len(wanted), 2)
        if split == "train":
            classes_before = {meta[v] for v in wanted}
            classes_after = {meta[v] for v in kept}
            report["n_train_classes_with_no_examples"] = len(
                classes_before - classes_after)
        print(f"{split}: {len(kept)}/{len(wanted)} kept ({dropped} dropped)")

    if report["n_val_dropped"] or report["n_test_dropped"]:
        print("\nWARNING: val/test lost clips — the re-posed run is NO LONGER "
              "scored on the same eval set as the released-pose run. Stop and "
              "reconcile before reporting any comparison.")
    else:
        print("\nval/test complete: eval protocol identical to the "
              "released-pose run; only train shrinks.")

    write_json(RESULTS_DIR / "reposed_labels_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
