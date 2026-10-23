"""Enumerate PopSign clips and join videos <-> landmark parquet.

PopSign carries its split, class and signer in the DIRECTORY TREE and the
FILENAME, so there is no metadata CSV to dedupe (unlike SEM-LEX). This script
materialises the equivalent:

    <split>/<class>/<signer>-<class>-<timestamp>-<take>.mp4
                    ^^^^^^^^ clip_id (stem) — also the parquet basename

Writes results/manifest.csv with one row per clip:
    clip_id, split, label, signer_id, has_video, has_landmarks

Clips WITHOUT a landmark parquet are kept in the manifest (flagged) so the
join loss is visible and counted, and dropped later in build_splits.py — the
same "report, don't silently shrink" rule the SEM-LEX prep follows.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import (LANDMARKS_DIR, MANIFEST_CSV, MANIFEST_REPORT_JSON,
                         SPLIT_DIRS)

SPLITS = ["train", "val", "test"]


def parse_clip_id(stem: str):
    """'4a.6001-TV-2022_12_14_14_59_01.480-0' -> (signer, label_from_name)."""
    parts = stem.split("-")
    if len(parts) < 3:
        return None, None
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="only the first 5 classes per split")
    args = parser.parse_args()

    have_landmarks = {p.stem for p in LANDMARKS_DIR.glob("*.parquet")}
    print(f"landmark parquet on disk: {len(have_landmarks)}")

    rows = []
    report = {"created_at": utcnow(), "n_landmark_files": len(have_landmarks)}
    for split in SPLITS:
        root = SPLIT_DIRS[split]
        classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        if args.smoke:
            classes = classes[:5]
        # Interrupted wget leaves stray "<class>.tar.tmp*" dirs; they are not
        # classes and contain no mp4 — excluded explicitly and reported.
        stray = [c for c in classes if ".tar" in c]
        classes = [c for c in classes if ".tar" not in c]
        n_split = 0
        for cls in classes:
            for mp4 in (root / cls).glob("*.mp4"):
                signer, label_in_name = parse_clip_id(mp4.stem)
                if signer is None:
                    print(f"  WARN unparseable filename: {mp4.name}")
                    continue
                if label_in_name != cls:
                    # directory is authoritative; disagreement is worth seeing
                    print(f"  WARN {mp4.name}: dir class {cls} != "
                          f"name class {label_in_name}")
                rows.append({
                    "clip_id": mp4.stem, "split": split, "label": cls,
                    "signer_id": signer, "has_video": 1,
                    "has_landmarks": int(mp4.stem in have_landmarks),
                })
                n_split += 1
        report[f"n_{split}"] = n_split
        report[f"n_classes_{split}"] = len(classes)
        report[f"n_signers_{split}"] = len(
            {r["signer_id"] for r in rows if r["split"] == split})
        if stray:
            report[f"stray_dirs_{split}"] = stray
        print(f"{split}: {n_split} clips, {len(classes)} classes, "
              f"{report[f'n_signers_{split}']} signers"
              + (f", stray dirs {stray}" if stray else ""))

    missing = [r for r in rows if not r["has_landmarks"]]
    report["n_total"] = len(rows)
    report["n_missing_landmarks"] = len(missing)
    report["missing_landmarks_by_split"] = dict(
        Counter(r["split"] for r in missing))
    # parquet with no corresponding mp4 (harmless, but count it)
    report["n_landmarks_without_video"] = len(
        have_landmarks - {r["clip_id"] for r in rows})

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["clip_id", "split", "label", "signer_id", "has_video",
            "has_landmarks"]
    with open(MANIFEST_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    write_json(MANIFEST_REPORT_JSON, report)
    print(f"\nwrote {MANIFEST_CSV} ({len(rows)} rows)")
    print(f"missing landmarks: {len(missing)} "
          f"({report['missing_landmarks_by_split']})")
    print(f"landmarks without video: {report['n_landmarks_without_video']}")


if __name__ == "__main__":
    main()
