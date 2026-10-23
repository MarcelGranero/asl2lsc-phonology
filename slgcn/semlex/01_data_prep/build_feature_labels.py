"""Encode ASL-LEX phonological features into a sidecar label file.

Phase 2 (04_phase2_multihead) needs, per video, an integer vector of the
paper's 16 phonological feature types. This script derives that from the
deduped metadata and writes a sidecar pkl keyed by video_id, leaving the
upstream gendata output (`{split}_label.pkl`) untouched — the multi-head
feeder joins the two by name at load time.

Two judgement calls:

1. WHICH 16.  The deduped metadata carries 17 named phonological columns;
   the paper's Table 1 enumerates 16.  The extra column is `Flexion Change`,
   which appears in the ASL-LEX release but in neither Table 1 nor the
   per-feature results of Table 5.  It is EXCLUDED here so the head count
   matches the paper.  `--include-flexion-change` overrides for ablation.

2. HOW NULLS ARE TREATED.  Table 1's caption states that "n/a appears in
   some Boolean phonological feature types, resulting in three possible
   values instead of two" — i.e. the paper treats missing as a REAL class,
   not as a masked-out target.  Our value counts confirm this: with n/a
   counted, 13 of 16 features match the paper's #Values exactly (Thumb
   Contact 3, Spread 3, Spread Change 3, Flexion 8, Path Movement 8,
   Nondominant Handshape 56 ...).  So the default `--null-policy class`
   encodes null as its own class.  `--null-policy ignore` emits -1 instead,
   for use with CrossEntropyLoss(ignore_index=-1); that was the phase-1
   design note's assumption, kept as an option because it is defensible for
   structurally-absent values (Nondominant Handshape on a one-handed sign).

The label space is derived FROM THE DATA (sorted for determinism), not
hardcoded to Table 1's counts.  Three features legitimately differ from the
paper and the difference is recorded rather than forced:

    Selected Fingers        12 vs 8   release has rarer finger combinations
    Handshape               57 vs 58  one ASL-LEX handshape unattested here
    Second Minor Location   36 vs 37  one ASL-LEX location unattested here

The over-count is the same "release carries ~4% more data" effect logged in
docs/VOCAB_DERIVATION.md; the under-counts follow from the paper_protocol
vocabulary dropping sub-5-instance signs, which can remove a rare value's
last instance.  Forcing the paper's inventory would mean inventing classes
with no examples (under) or merging real ones (over) — both worse.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_config as dc  # noqa: E402
from common import git_sha, utcnow, write_json  # noqa: E402

# The paper's 16 types, in Table 1 order.  `Flexion Change` is deliberately
# absent — see the module docstring.
PAPER_FEATURES = [
    "Major Location",
    "Minor Location",
    "Second Minor Location",
    "Contact",
    "Thumb Contact",
    "Thumb Position",
    "Nondominant Handshape",
    "Handshape",
    "Selected Fingers",
    "Flexion",
    "Spread",
    "Spread Change",
    "Repeated Movement",
    "Sign Type",
    "Wrist Twist",
    "Path Movement",
]

EXTRA_FEATURE = "Flexion Change"

# Table 1 "#Values", for reporting the delta only — never to force the space.
PAPER_N_VALUES = {
    "Major Location": 5,
    "Minor Location": 37,
    "Second Minor Location": 37,
    "Contact": 2,
    "Thumb Contact": 3,
    "Thumb Position": 2,
    "Nondominant Handshape": 56,
    "Handshape": 58,
    "Selected Fingers": 8,
    "Flexion": 8,
    "Spread": 3,
    "Spread Change": 3,
    "Repeated Movement": 2,
    "Sign Type": 6,
    "Wrist Twist": 2,
    "Path Movement": 8,
}

NULL_TOKEN = "n/a"


def _normalise(value):
    """Canonicalise a raw cell to a hashable label token or None.

    The columns are a mix of dtypes: some are object (strings), others were
    read as float64 because their only values are booleans plus NaN.  Map
    both onto stable string tokens so the vocabulary is dtype-independent
    and reproducible across pandas versions.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Boolean-valued columns arrive as 0.0/1.0; keep integers integral so
        # the token is "1" and not "1.0".
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    return text if text else None


def build_vocabularies(df: pd.DataFrame, features: list[str], null_policy: str) -> dict:
    """Derive {feature: {token: index}} from the data, sorted for determinism."""
    vocabs = {}
    for feat in features:
        tokens = {t for t in (_normalise(v) for v in df[feat]) if t is not None}
        ordered = sorted(tokens)
        if null_policy == "class" and df[feat].isna().any():
            # n/a always takes the LAST index so that adding data later (which
            # can only add real values) never renumbers it.
            ordered = ordered + [NULL_TOKEN]
        vocabs[feat] = {tok: i for i, tok in enumerate(ordered)}
    return vocabs


def encode(df: pd.DataFrame, features: list[str], vocabs: dict, null_policy: str) -> dict:
    """Return {video_id: [int per feature]}, -1 where a value is masked."""
    out = {}
    columns = [df[f].tolist() for f in features]
    for row_i, video_id in enumerate(df["video_id"].tolist()):
        vector = []
        for feat_i, feat in enumerate(features):
            token = _normalise(columns[feat_i][row_i])
            if token is None:
                token = NULL_TOKEN if null_policy == "class" else None
            vector.append(vocabs[feat].get(token, -1) if token is not None else -1)
        out[video_id] = vector
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--null-policy",
        choices=["class", "ignore"],
        default="class",
        help="'class': n/a is its own label (paper's Table 1 reading, default). "
        "'ignore': emit -1 so the loss can mask it via ignore_index.",
    )
    ap.add_argument(
        "--include-flexion-change",
        action="store_true",
        help="Add the 17th column the paper omits (ablation only — makes the "
        "head count 17 and no longer comparable to Table 5).",
    )
    ap.add_argument("--out", type=Path, default=dc.FEATURE_LABELS_PKL)
    ap.add_argument("--vocab-out", type=Path, default=dc.FEATURE_VOCAB_JSON)
    args = ap.parse_args()

    features = list(PAPER_FEATURES)
    if args.include_flexion_change:
        features.append(EXTRA_FEATURE)

    df = pd.read_csv(dc.METADATA_DEDUPED_CSV, low_memory=False)
    splits = json.loads(dc.SPLITS_JSON.read_text())
    split_of = {
        video_id: name
        for name in ("train", "val", "test")
        for video_id in splits[name]
    }

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"metadata is missing expected columns: {missing}")

    df = df[df["video_id"].isin(split_of)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no metadata rows intersect results/splits.json")

    # The vocabulary is derived from TRAIN ONLY, so a value that appears only
    # in val/test cannot silently define a class the model was never taught.
    # Such values encode to -1 (unknown) and are excluded from PFR scoring.
    train_df = df[df["video_id"].map(split_of) == "train"]
    vocabs = build_vocabularies(train_df, features, args.null_policy)
    labels = encode(df, features, vocabs, args.null_policy)

    report = {}
    for feat in features:
        column = df[feat]
        n_derived = len(vocabs[feat])
        unknown = sum(1 for v in labels.values() if v[features.index(feat)] == -1)
        report[feat] = {
            "n_classes": n_derived,
            "paper_n_values": PAPER_N_VALUES.get(feat),
            "delta_vs_paper": (
                n_derived - PAPER_N_VALUES[feat] if feat in PAPER_N_VALUES else None
            ),
            "null_rate": round(float(column.isna().mean()), 6),
            "n_unlabelled": unknown,
            "classes": list(vocabs[feat]),
        }

    payload = {
        "created_at": utcnow(),
        "git_sha": git_sha(),
        "null_policy": args.null_policy,
        "features": features,
        "n_features": len(features),
        "vocab_sha256": json.loads(dc.VOCAB_JSON.read_text())["vocab_sha256"],
        "n_videos": len(labels),
        "per_feature": report,
    }
    write_json(args.vocab_out, payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)

    print(f"wrote {args.out}  ({len(labels)} videos x {len(features)} features)")
    print(f"wrote {args.vocab_out}")
    print(f"\n{'feature':24s} {'ours':>5s} {'paper':>6s} {'null%':>7s} {'unlab':>7s}")
    for feat in features:
        r = report[feat]
        paper = r["paper_n_values"]
        flag = "" if paper is None or r["n_classes"] == paper else "  <-- differs"
        print(
            f"{feat:24s} {r['n_classes']:5d} {str(paper):>6s} "
            f"{100 * r['null_rate']:6.2f}% {r['n_unlabelled']:7d}{flag}"
        )


if __name__ == "__main__":
    main()
