"""Encode PopSign's ASL-LEX phonological features into a sidecar label file.

Phase 2 (`04_phase2_multihead/`) needs, per clip, an integer vector of
phonological feature types. PopSign ships them per clip inside the
`merged_videos_dataset_batch*.json` files under
`ExternalDatasets/popsign_videos/<split>/`:

    {"path": "<clip_id>.parquet", "sign": "TV", "handshape": "baby_o",
     "subhandshapes": {"selected_fingers": "i", "flexion": "Bent",
                       "spread": "NA", "thumb_position": "Open",
                       "thumb_contact": "1"}, "landmarks": [...]}

We read ONLY the label fields — the embedded `landmarks` are the same
MediaPipe data we already took from the parquet, and re-reading them here
would be 15 GB of wasted I/O.

This writes a sidecar pkl keyed by clip_id, leaving upstream's gendata output
(`{split}_label.pkl`) untouched — the multi-head feeder joins the two by name
at load time, exactly as in the SL-GCN Sem-Lex leaf.

## How this differs from the SEM-LEX feature set

SEM-LEX carries the paper's **16** ASL-LEX feature types (handshape, movement,
location, …). PopSign carries **6**, all of them HANDSHAPE-related:

    handshape          37 values  (the ASL-LEX handshape inventory)
    selected_fingers    9         which fingers are selected
    flexion             7         incl. NA
    spread              3         0 / 1 / NA
    thumb_position      2         Open / Closed
    thumb_contact       3         0 / 1 / NA

So a PopSign phase-2 run is a **handshape-phonology** model, not a
reproduction of the paper's full 16-type Table 5. Reported as such — the two
are not comparable head-for-head.

## Judgement calls (same conventions as the SEM-LEX build)

1. `NA` IS A CLASS, not a mask. PopSign writes the literal string `"NA"`
   (not null) for structurally-absent values — e.g. `spread` is NA whenever
   fewer than two fingers are selected, and `flexion`/`thumb_contact` are NA
   on thumb-only handshapes. That matches the paper's Table 1 caption ("n/a
   appears in some Boolean phonological feature types, resulting in three
   possible values instead of two"), so NA is encoded as its own label and
   always takes the LAST index — adding data later (which can only introduce
   real values) never renumbers it. `--null-policy ignore` emits -1 instead,
   for CrossEntropyLoss(ignore_index=-1).

2. The label space is derived FROM THE TRAINING SPLIT ONLY, sorted for
   determinism. A value occurring solely in val/test would otherwise define a
   class the model was never taught; instead it encodes to -1 and is excluded
   from scoring. Any such value is reported.

3. Only ~82% of packed clips have a feature row (train 91,296/110,774; val
   24,628/29,868; test 27,340/33,599). The JSON batches simply cover fewer
   clips than the video/landmark trees. Unlabelled clips are NOT dropped:
   the feeder masks them with -1, so PFR trains on the labelled subset while
   ISR+PFR still gets the full gloss signal from every clip. The coverage is
   recorded in results/feature_vocab.json.
"""
import argparse
import glob
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import git_sha, utcnow, write_json  # noqa: E402
from data_config import (FEATURE_LABELS_PKL, FEATURE_VOCAB_JSON,  # noqa: E402
                         GENDATA_DIR, POPSIGN_JSON_ROOT)

# Order is fixed here (head order in the model + report order). `handshape`
# first, then the five subhandshape components in PopSign's own key order.
FEATURES = ["handshape", "selected_fingers", "flexion", "spread",
            "thumb_position", "thumb_contact"]
SUBHANDSHAPE = set(FEATURES[1:])
NULL_TOKEN = "NA"
SPLITS = ["train", "val", "test"]


def read_split(split: str):
    """Yield (clip_id, {feature: value}) from a split's JSON batches."""
    files = sorted(glob.glob(str(POPSIGN_JSON_ROOT / split /
                                 "merged_videos_dataset_batch*.json")))
    if not files:
        sys.exit(f"no feature JSON batches under {POPSIGN_JSON_ROOT / split}")
    for path in files:
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            clip_id = r["path"].rsplit(".parquet", 1)[0]
            sub = r.get("subhandshapes") or {}
            vals = {"handshape": r.get("handshape")}
            for k in SUBHANDSHAPE:
                vals[k] = sub.get(k)
            yield clip_id, vals


def normalise(v):
    """None/'' -> the NA token; PopSign already writes literal 'NA'."""
    if v is None or v == "":
        return NULL_TOKEN
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-policy", default="class",
                        choices=["class", "ignore"],
                        help="'class': NA is its own label (paper default). "
                             "'ignore': NA -> -1, masked in the loss.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen feature vocab")
    args = parser.parse_args()

    if FEATURE_VOCAB_JSON.exists() and not args.force:
        v = json.loads(FEATURE_VOCAB_JSON.read_text())
        print(f"feature_vocab.json already frozen "
              f"({len(v['features'])} features; --force to rebuild)")
        return

    per_split = {s: dict(read_split(s)) for s in SPLITS}
    for s in SPLITS:
        print(f"{s}: {len(per_split[s])} clips with feature labels")

    # --- label space from TRAIN only, sorted; NA last ---
    vocab, counts = {}, {}
    for feat in FEATURES:
        c = Counter(normalise(v[feat]) for v in per_split["train"].values())
        values = sorted(x for x in c if x != NULL_TOKEN)
        if NULL_TOKEN in c and args.null_policy == "class":
            values.append(NULL_TOKEN)          # NA always last
        vocab[feat] = {val: i for i, val in enumerate(values)}
        counts[feat] = dict(c.most_common())

    # --- encode every clip in every split ---
    labels, unseen = {}, defaultdict(Counter)
    for s in SPLITS:
        for clip_id, vals in per_split[s].items():
            vec = []
            for feat in FEATURES:
                val = normalise(vals[feat])
                if val == NULL_TOKEN and args.null_policy == "ignore":
                    vec.append(-1)
                    continue
                idx = vocab[feat].get(val)
                if idx is None:                # value unseen in train
                    unseen[feat][val] += 1
                    idx = -1
                vec.append(idx)
            labels[clip_id] = vec

    # --- coverage against the packed clips the model will actually see ---
    coverage = {}
    for s in SPLITS:
        with open(GENDATA_DIR / f"{s}_label.pkl", "rb") as f:
            names, _ = pickle.load(f)
        have = sum(1 for n in names if n in labels)
        coverage[s] = {"n_packed": len(names), "n_with_features": have,
                       "n_masked": len(names) - have,
                       "frac": round(have / len(names), 4)}
        print(f"{s}: {have}/{len(names)} packed clips have features "
              f"({coverage[s]['frac'] * 100:.1f}%)")

    FEATURE_LABELS_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_LABELS_PKL, "wb") as f:
        pickle.dump({"features": FEATURES, "labels": labels}, f)

    n_train = len(per_split["train"])
    payload = {
        "created_at": utcnow(),
        "git_sha": git_sha(),
        "source": "PopSign merged_videos_dataset_batch*.json "
                  "(handshape + subhandshapes fields only)",
        "null_policy": args.null_policy,
        "null_token": NULL_TOKEN,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "n_videos": len(labels),
        "feature_sizes": [len(vocab[f]) for f in FEATURES],
        # `per_feature` mirrors the SL-GCN Sem-Lex schema so 04_phase2_multihead
        # reads it unmodified. No paper_n_values/delta here: PopSign's 6
        # handshape features are not the paper's 16 types, so there is nothing
        # to compare against.
        "per_feature": {
            f: {
                "n_classes": len(vocab[f]),
                "null_rate": round(
                    counts[f].get(NULL_TOKEN, 0) / max(n_train, 1), 4),
                "classes": sorted(vocab[f], key=vocab[f].get),
            }
            for f in FEATURES
        },
        "vocab": vocab,
        "train_value_counts": counts,
        "values_unseen_in_train": {k: dict(v) for k, v in unseen.items()},
        "coverage_vs_packed_clips": coverage,
        "note": "PopSign carries 6 HANDSHAPE-related feature types, not the "
                "16 ASL-LEX types of the Sem-Lex paper's Table 1/5 — a "
                "PopSign phase-2 run is a handshape-phonology model and is "
                "NOT head-for-head comparable to that table.",
    }
    write_json(FEATURE_VOCAB_JSON, payload)

    print(f"\nwrote {FEATURE_LABELS_PKL} ({len(labels)} clips)")
    print(f"wrote {FEATURE_VOCAB_JSON}")
    for feat in FEATURES:
        print(f"  {feat:18s} {len(vocab[feat]):3d} classes")
    for feat, vals in unseen.items():
        if vals:
            print(f"  WARN {feat}: values unseen in train -> -1: {dict(vals)}")


if __name__ == "__main__":
    main()
