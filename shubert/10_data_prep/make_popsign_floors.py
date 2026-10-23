"""Majority-class floors for the popsign16 splits.

Mirrors the report block of make_labels.py so check_smoke.py can gate a
PopSign smoke run the same way it gates Sem-Lex ones
(`--report popsign16_labels_report.json`).

One PopSign-only wrinkle: label -1 means "no supervision for this feature on
this clip" (gloss absent from ASL-LEX, or a structurally absent value under
the ignore policy). engine.run_eval excludes those rows from each feature's
denominator, so the floor must too — a floor computed over ALL rows would be
artificially low and let a majority-collapsed head pass the smoke gate.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_config import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="popsign16",
                    help="split-name prefix (popsign16nac for the NA-as-class "
                         "build, mandatory for new runs)")
    args = ap.parse_args()
    prefix = args.prefix
    SPLITS = tuple(f"{prefix}_{s}" for s in ("train", "val", "test"))
    maps = json.loads((RESULTS_DIR / f"{prefix}_label_maps.json").read_text())
    feature_names = maps["feature_order"]
    id2str = {
        name: {v: k for k, v in maps["maps"][name].items()}
        for name in feature_names
    }

    report = {"metadata": {"source": "popsign16_{split}_labels.csv",
                           "floor_denominator": "valid (label != -1) rows"},
              "splits": {}}
    for split in SPLITS:
        with (RESULTS_DIR / f"{split}_labels.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        hist = {
            name: Counter(
                id2str[name][int(r[f"f{i:02d}"])]
                for r in rows if int(r[f"f{i:02d}"]) != -1
            )
            for i, name in enumerate(feature_names)
        }
        n_valid = {name: sum(hist[name].values()) for name in feature_names}
        report["splits"][split] = {
            "n": len(rows),
            "n_signers": len({r["signer_id"] for r in rows}),
            "n_valid": n_valid,
            "majority_class_floor": {
                name: (round(hist[name].most_common(1)[0][1] / n_valid[name], 4)
                       if n_valid[name] else None)
                for name in feature_names
            },
            "class_histogram": {
                name: dict(sorted(hist[name].items()))
                for name in feature_names
            },
        }
        print(f"{split}: {len(rows)} rows, min valid "
              f"{min(n_valid.values())}, floors "
              f"{sorted(report['splits'][split]['majority_class_floor'].values())[:3]}…")

    out = RESULTS_DIR / f"{prefix}_labels_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
