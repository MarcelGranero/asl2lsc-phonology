#!/usr/bin/env python3
"""Convert a per-leaf LSC per-clip CSV into the standard predictions TSV.

Every leaf eval records the five raw predicted attribute streams as
pred_selected_fingers .. pred_thumb_contact (clip column "clip_stem" for
shubert/baseline_mlp, "clip" for the slgcn leaves).  This script just
extracts those six columns, preserving the model's own value spellings --
all canonicalisation happens in evaluate.py, identically for every model.

    python3 convert_legacy.py <legacy per_clip csv> -o predictions/<name>.tsv
"""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsc_map import STREAMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legacy_csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    with open(str(args.legacy_csv), newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        clip_col = next((c for c in ("clip_stem", "clip") if c in fields), None)
        missing = [c for c in ["pred_" + s for s in STREAMS] if c not in fields]
        if clip_col is None or missing:
            raise SystemExit("%s: not a recognised per-clip CSV (clip column: "
                             "%s, missing: %s)"
                             % (args.legacy_csv, clip_col, missing))
        rows = list(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(("clip_id",) + STREAMS)
        for r in rows:
            w.writerow([r[clip_col]] + [r["pred_" + s] for s in STREAMS])
    print("%s: %d rows -> %s" % (args.legacy_csv, len(rows), args.output))


if __name__ == "__main__":
    main()
