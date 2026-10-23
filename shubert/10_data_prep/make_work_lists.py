"""Build the video work list (upstream gzip-pickle format) from the
transcoded mp4 dir, and report labeled clips whose video is missing.

The list lives in the dedicated work_lists/ dir: kpe_mediapipe.py rewrites
<parent>/<parent-name>.list.gz at startup, which is harmless there.

Smoke mode (--limit N) writes {split}_smoke_videos.list.gz instead; all
downstream stage scripts derive their input from whichever list the SLURM
chain passes along.
"""

import argparse
import csv
import gzip
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    RESULTS_DIR,
    SPLITS,
    VIDEOS_MP4_DIR,
    WORK_LISTS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=list(SPLITS))
    parser.add_argument("--limit", type=int, default=0,
                        help="Only the first N mp4s (smoke mode); 0 = all")
    args = parser.parse_args()

    mp4s = sorted((VIDEOS_MP4_DIR / args.split).glob("*.mp4"))
    if args.limit:
        mp4s = mp4s[: args.limit]
    assert mp4s, f"no mp4s under {VIDEOS_MP4_DIR / args.split}"

    prefix = f"{args.split}_smoke" if args.limit else args.split
    WORK_LISTS_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_LISTS_DIR / f"{prefix}_videos.list.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump([str(p) for p in mp4s], f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(mp4s)} videos -> {out}")

    # labeled clips without a video (full mode only — smoke is a subset by design)
    if not args.limit:
        with (RESULTS_DIR / f"{args.split}_labels.csv").open(newline="") as f:
            labeled = {row["video_id"] for row in csv.DictReader(f)}
        have = {p.stem for p in mp4s}
        missing = sorted(labeled - have)
        report = RESULTS_DIR / f"{args.split}_missing_videos.json"
        report.write_text(json.dumps(
            {"n_labeled": len(labeled), "n_with_video": len(labeled & have),
             "n_missing": len(missing), "missing": missing}, indent=2))
        print(f"labeled clips missing a video: {len(missing)} -> {report}")


if __name__ == "__main__":
    main()
