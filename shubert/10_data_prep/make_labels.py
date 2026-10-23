"""Build the 16 phonological-feature class maps and per-split label CSVs.

Reads the deduped Sem-Lex metadata (reused verbatim from the SL-GCN Sem-Lex leaf, sha
pinned) — or the raw metadata with --raw-metadata for ablation — keeps only
label_type=="asllex" rows (the only rows carrying features), and derives one
class map per feature: sorted unique canonical string values, plus "__NA__"
appended iff the column has blank cells.

Label-integrity gate: each map's size must equal the paper's Table 10 class
count (data_config.FEATURES_TABLE10). Any mismatch aborts.

Outputs (committed, in results/):
  label_maps.json          {"feature_order": [...], "maps": {feature: {value: id}}}
  {split}_labels.csv       video_id,signer_id,f00..f15  (int class ids)
  labels_report.json       split sizes, per-feature class histograms,
                           majority-class Recall@1 floors, signer overlap info

Stdlib only — must run under shubert_train_env (no pandas).
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    FEATURES_TABLE10,
    METADATA_DEDUPED,
    METADATA_DEDUPED_SHA256,
    NA_CLASS,
    RESULTS_DIR,
    SEMLEX_METADATA,
    SPLITS,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(value: str) -> str:
    """Canonical string form of a feature cell; blank -> NA_CLASS."""
    value = value.strip()
    return value if value else NA_CLASS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-metadata",
        action="store_true",
        help="Ablation: derive labels from the raw semlex_metadata.csv "
        "(no dedupe) instead of the pinned deduped CSV.",
    )
    args = parser.parse_args()

    if args.raw_metadata:
        metadata_path = SEMLEX_METADATA
        print(f"ABLATION: using raw metadata {metadata_path}")
    else:
        metadata_path = METADATA_DEDUPED
        digest = sha256_file(metadata_path)
        assert digest == METADATA_DEDUPED_SHA256, (
            f"{metadata_path} sha256 {digest} != pinned {METADATA_DEDUPED_SHA256}; "
            "the SL-GCN Sem-Lex leaf's dedupe output changed — re-verify before trusting it."
        )

    feature_names = [name for name, _ in FEATURES_TABLE10]

    rows = []
    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in feature_names if c not in reader.fieldnames]
        assert not missing_cols, f"metadata lacks feature columns: {missing_cols}"
        for row in reader:
            if row["label_type"] != "asllex":
                continue
            assert row["split"] in SPLITS, f"unexpected split {row['split']!r}"
            rows.append(row)
    print(f"asllex rows: {len(rows)}")

    # ---------------------------------------------------------- class maps
    maps: dict[str, dict[str, int]] = {}
    for name, expected in FEATURES_TABLE10:
        values = {canon(r[name]) for r in rows}
        has_na = NA_CLASS in values
        ordered = sorted(values - {NA_CLASS}) + ([NA_CLASS] if has_na else [])
        assert len(ordered) == expected, (
            f"{name}: derived {len(ordered)} classes "
            f"({ordered[:8]}{'...' if len(ordered) > 8 else ''}), "
            f"paper Table 10 says {expected}. Label derivation is wrong — abort."
        )
        maps[name] = {v: i for i, v in enumerate(ordered)}
    print(f"class maps OK: all {len(maps)} feature sizes match paper Table 10")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "label_maps.json").write_text(
        json.dumps({"feature_order": feature_names, "maps": maps}, indent=2)
    )

    # ------------------------------------------------- per-split label csvs
    report: dict = {"metadata": str(metadata_path), "splits": {}}
    for split in SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        out = RESULTS_DIR / f"{split}_labels.csv"
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["video_id", "signer_id"]
                + [f"f{i:02d}" for i in range(len(feature_names))]
            )
            for r in sorted(split_rows, key=lambda r: r["video_id"]):
                writer.writerow(
                    [r["video_id"], r["signer_id"]]
                    + [maps[name][canon(r[name])] for name in feature_names]
                )

        hist = {
            name: Counter(canon(r[name]) for r in split_rows)
            for name in feature_names
        }
        report["splits"][split] = {
            "n": len(split_rows),
            "n_signers": len({r["signer_id"] for r in split_rows}),
            "majority_class_floor": {
                name: (
                    round(hist[name].most_common(1)[0][1] / len(split_rows), 4)
                    if split_rows
                    else None
                )
                for name in feature_names
            },
            "class_histogram": {
                name: dict(sorted(hist[name].items())) for name in feature_names
            },
        }
        print(f"{split}: {len(split_rows)} rows -> {out}")

    signers = {
        split: {r["signer_id"] for r in rows if r["split"] == split}
        for split in SPLITS
    }
    report["signer_overlap"] = {
        "test&train": len(signers["test"] & signers["train"]),
        "test&val": len(signers["test"] & signers["val"]),
        "train&val": len(signers["train"] & signers["val"]),
    }
    (RESULTS_DIR / "labels_report.json").write_text(json.dumps(report, indent=2))
    print(f"signer overlap: {report['signer_overlap']}")
    print(f"report -> {RESULTS_DIR / 'labels_report.json'}")


if __name__ == "__main__":
    main()
