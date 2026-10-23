"""Build the 5-column training manifests from completed feature dirs.

Per split, joins on video stem across the four streams and keeps stems with
all of: dinov2_face/<s>_face.npy, dinov2_hands/<s>_hand{1,2}.npy,
body_features/<s>_pose.npy. num_frames = min length over the four streams
(the crop scripts drop leading frames before the first pose detection, so
streams can disagree; readers truncate to num_frames).

Outputs (manifests/):
  {split}.tsv       headerless: face \t hand1 \t hand2 \t body \t num_frames
                    — restricted to video_ids in results/{split}_labels.csv
  {split}_all.tsv   every completed stem (future ISLR/gloss reuse)
plus results/{split}_manifest_report.json (drops + length-mismatch stats).

Stdlib + numpy only (runs in either env).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    MANIFESTS_DIR,
    RESULTS_DIR,
    SPLITS,
    stage_dir,
)


def npy_len(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


def build_split(split: str) -> None:
    dino_face = stage_dir(split, "dinov2_face")
    dino_hands = stage_dir(split, "dinov2_hands")
    body_dir = stage_dir(split, "body_features")

    stems = sorted(p.name[: -len("_face.npy")] for p in dino_face.glob("*_face.npy"))
    rows = []
    incomplete, mismatched, empty = [], [], []
    for s in stems:
        paths = [
            dino_face / f"{s}_face.npy",
            dino_hands / f"{s}_hand1.npy",
            dino_hands / f"{s}_hand2.npy",
            body_dir / f"{s}_pose.npy",
        ]
        if not all(p.exists() for p in paths):
            incomplete.append(s)
            continue
        lengths = [npy_len(p) for p in paths]
        n = min(lengths)
        if n <= 0:
            empty.append(s)
            continue
        if max(lengths) - n > 2:
            mismatched.append({"stem": s, "lengths": lengths})
        rows.append((s, paths, n))

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    with (MANIFESTS_DIR / f"{split}_all.tsv").open("w") as f:
        for s, paths, n in rows:
            f.write("\t".join(str(p) for p in paths) + f"\t{n}\n")

    with (RESULTS_DIR / f"{split}_labels.csv").open(newline="") as f:
        labeled = {r["video_id"] for r in csv.DictReader(f)}
    labeled_rows = [(s, paths, n) for s, paths, n in rows if s in labeled]
    with (MANIFESTS_DIR / f"{split}.tsv").open("w") as f:
        for s, paths, n in labeled_rows:
            f.write("\t".join(str(p) for p in paths) + f"\t{n}\n")

    report = {
        "n_face_npy_stems": len(stems),
        "n_complete": len(rows),
        "n_incomplete": len(incomplete),
        "n_empty": len(empty),
        "n_length_mismatch_gt2": len(mismatched),
        "n_labeled": len(labeled),
        "n_labeled_with_manifest": len(labeled_rows),
        "labeled_missing_from_manifest": len(labeled) - len(labeled_rows),
        "incomplete_stems": incomplete[:50],
        "empty_stems": empty[:50],
        "length_mismatches": mismatched[:50],
    }
    out = RESULTS_DIR / f"{split}_manifest_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{split}: {len(labeled_rows)} labeled / {len(rows)} total rows "
          f"-> {MANIFESTS_DIR / (split + '.tsv')}  (report: {out})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=list(SPLITS),
                        required=True)
    args = parser.parse_args()
    for split in args.split:
        build_split(split)


if __name__ == "__main__":
    main()
