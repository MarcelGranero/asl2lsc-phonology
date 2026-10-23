"""Derive 6-feature (handshape-family) sidecars from the 16-feature ones.

The SL-GCN Sem-Lex heads6 ablation trains a genuine 6-head model — the same
construction as the PopSign SL-GCN heads6 arm (feature_labels.pkl there has
exactly these 6 native fields) — so the two heads6 rows of the ablation
table are like-for-like within the family. The 6 features are the
handshape-family set used everywhere else (SHuBERT's --keep-heads
"handshape" preset, the MLP heads6 grid).

Reads FEATURE_LABELS_PKL + FEATURE_VOCAB_JSON (16 features), writes:
  <PROCESSED_DIR>/feature_labels_heads6.pkl   {"features": [6], "labels": {clip: [6]}}
  results/feature_vocab_heads6.json           same schema, 6-feature subset

Pure column subset: encodings, class ids and null policy are byte-identical
to the 16-feature build for the surviving columns.

Run under any env with numpy-free stdlib (login python3 is fine):
  python3 04_phase2_multihead/make_heads6_sidecars.py
"""
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_config import (FEATURE_LABELS_PKL, FEATURE_VOCAB_JSON,  # noqa: E402
                         PROCESSED_DIR, RESULTS_DIR)

HEADS6 = ["Handshape", "Selected Fingers", "Flexion", "Spread",
          "Thumb Position", "Thumb Contact"]

OUT_PKL = PROCESSED_DIR / "feature_labels_heads6.pkl"
OUT_VOCAB = RESULTS_DIR / "feature_vocab_heads6.json"


def main() -> None:
    with open(FEATURE_LABELS_PKL, "rb") as f:
        payload = pickle.load(f)
    features16 = payload["features"]
    idx = [features16.index(name) for name in HEADS6]

    labels6 = {clip: [vec[i] for i in idx]
               for clip, vec in payload["labels"].items()}
    with open(OUT_PKL, "wb") as f:
        pickle.dump({"features": HEADS6, "labels": labels6}, f)

    vocab16 = json.loads(FEATURE_VOCAB_JSON.read_text())
    vocab6 = dict(vocab16)
    vocab6["features"] = HEADS6
    vocab6["n_features"] = len(HEADS6)
    vocab6["per_feature"] = {name: vocab16["per_feature"][name]
                             for name in HEADS6}
    vocab6["derived_from"] = {
        "vocab": str(FEATURE_VOCAB_JSON), "labels": str(FEATURE_LABELS_PKL),
        "note": "pure column subset by make_heads6_sidecars.py; "
                "class encodings unchanged"}
    OUT_VOCAB.write_text(json.dumps(vocab6, indent=2))

    sizes = [vocab6["per_feature"][n]["n_classes"] for n in HEADS6]
    print(f"wrote {OUT_PKL} ({len(labels6)} clips, 6 features)")
    print(f"wrote {OUT_VOCAB} (sizes {sizes})")


if __name__ == "__main__":
    main()
