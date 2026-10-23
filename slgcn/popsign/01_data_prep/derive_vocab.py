"""Freeze the 250-class PopSign label space.

Unlike SEM-LEX — where the paper's vocabulary rule had to be reverse-engineered
and every candidate rule documented — PopSign's class set is given twice and we
only have to check the two agree:

  1. the split directory names in the video tree (the data we actually train on)
  2. `sign_to_prediction_index_map.json` from the official Kaggle release
     (the canonical 0..249 ordering)

We adopt the Kaggle ordering when the sets match, so label ints are comparable
to any published PopSign/Kaggle numbers. If they DISAGREE the script fails
loudly rather than inventing a merge — a silent vocabulary mismatch is the one
error that would poison every downstream number invisibly.

Writes results/vocab.json (SHA-stamped; frozen once built).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import KAGGLE_SIGN_MAP_JSON, MANIFEST_CSV, VOCAB_JSON


def read_manifest_labels():
    labels = {}
    with open(MANIFEST_CSV) as f:
        header = f.readline().strip().split(",")
        i_label, i_split = header.index("label"), header.index("split")
        for line in f:
            p = line.rstrip("\n").split(",")
            labels.setdefault(p[i_label], set()).add(p[i_split])
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen vocab.json")
    args = parser.parse_args()

    if VOCAB_JSON.exists() and not args.force:
        existing = json.loads(VOCAB_JSON.read_text())
        print(f"vocab.json already frozen: {existing['n_classes']} classes, "
              f"sha {existing['vocab_sha256'][:12]}… (use --force to rebuild)")
        return

    from_dirs = read_manifest_labels()
    kaggle = json.loads(KAGGLE_SIGN_MAP_JSON.read_text())

    only_dirs = sorted(set(from_dirs) - set(kaggle))
    only_kaggle = sorted(set(kaggle) - set(from_dirs))
    if only_dirs or only_kaggle:
        print(f"FATAL: class sets disagree.\n"
              f"  only in split dirs ({len(only_dirs)}): {only_dirs[:20]}\n"
              f"  only in kaggle map ({len(only_kaggle)}): {only_kaggle[:20]}")
        sys.exit(1)

    label_to_id = {k: int(v) for k, v in sorted(kaggle.items(),
                                                key=lambda kv: kv[1])}
    ids = sorted(label_to_id.values())
    assert ids == list(range(len(ids))), "kaggle ids are not contiguous 0..N-1"

    # classes absent from a split can never be scored there — report, don't fail
    missing_by_split = {
        s: sorted(l for l, ss in from_dirs.items() if s not in ss)
        for s in ("train", "val", "test")
    }

    payload = {
        "created_at": utcnow(),
        "source": "PopSign v1.0 game split dirs, ordering from "
                  "sign_to_prediction_index_map.json",
        "n_classes": len(label_to_id),
        "label_to_id": label_to_id,
        "n_classes_missing_per_split": {k: len(v)
                                        for k, v in missing_by_split.items()},
        "classes_missing_per_split": missing_by_split,
    }
    blob = json.dumps(payload["label_to_id"], sort_keys=True).encode()
    payload["vocab_sha256"] = hashlib.sha256(blob).hexdigest()

    write_json(VOCAB_JSON, payload)
    print(f"froze {payload['n_classes']} classes -> {VOCAB_JSON}")
    print(f"vocab_sha256 {payload['vocab_sha256']}")
    for s, v in missing_by_split.items():
        if v:
            print(f"  WARN {len(v)} classes absent from {s}: {v[:10]}")


if __name__ == "__main__":
    main()
