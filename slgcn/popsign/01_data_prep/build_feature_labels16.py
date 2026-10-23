"""Encode the FULL 16 ASL-LEX feature types for PopSign clips.

PopSign ships only 6 phonological fields per clip (handshape + 5 subhandshape
components) -- see build_feature_labels.py, which this file deliberately does
NOT touch. The other 10 ASL-LEX types (location, movement, sign type, ...) are
lexical properties of the SIGN, so they can be recovered by joining PopSign's
`sign` gloss to the ASL-LEX table that ships with SEM-LEX.

Validity of that join was checked before building anything: over 133 glosses
present in both sources, PopSign's own `handshape` value agrees with ASL-LEX's
`Handshape` 133/133 times. PopSign's labels are ASL-LEX-derived, so attaching
the remaining 10 columns from the same table is consistent, not a splice of two
different annotation schemes.

Gloss matching is CONSERVATIVE by design. Fuzzy string matching was evaluated
and rejected: at cutoff 0.75 it proposes bee->beer, cloud->loud, chin->china,
food->wood, goose->moose -- unrelated signs with different phonology, which
would inject silent label noise into exactly the features being measured. Only
three rules are used, all reversible by inspection:

  1. exact match on the normalised gloss
  2. numbered lemma variants ("closet" -> "closet 2")
  3. a hand-curated synonym table (English label differences
     for the same sign concept, e.g. dad -> father)

Committed-run tallies (results/feature_vocab16.json, gloss_match_rules --
always read the numbers from there, an earlier docstring drifted): of the
201 distinct glosses present in the PopSign label JSONs, 186 matched exact,
8 via variant, 7 unmatched; the synonym rule did not fire in that run.
Classes absent from the merged_videos label JSONs (~18% of clips carry no
label entry at all) never reach the matcher.

Unmatched glosses are absent from ASL-LEX. Their 10 joined features encode
to the null policy (-1 by default so CrossEntropyLoss ignores them) while
their 6 NATIVE PopSign features stay real -- so those clips still train the
handshape heads at full strength and only abstain on the lexical ones.
Coverage is recorded in the vocab JSON; nothing is silently dropped.

Writes NEW files (feature_labels16.pkl / feature_vocab16.json) so the existing
6-feature pipeline keeps working byte-identically.
"""
import argparse
import csv
import glob
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import git_sha, utcnow, write_json  # noqa: E402
from data_config import POPSIGN_JSON_ROOT, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

# SEM-LEX ships the ASL-LEX table; reused read-only (never written to).
# Monorepo sibling (gitignored there; regenerable by its 01_data_prep).
SEMLEX_METADATA_DEDUPED = (
    Path(__file__).resolve().parents[2]
    / "semlex" / "results" / "metadata_deduped.csv")

# The canonical 16 and their order are taken from the SHuBERT-leaf build, so
# a PopSign-16 head vector is positionally identical to a SemLex-16 one and the
# two runs stay directly comparable. Do NOT hand-maintain this list.
FEATURE_ORDER_16 = [
    "Major Location", "Minor Location", "Second Minor Location", "Contact",
    "Thumb Contact", "Sign Type", "Repeated Movement", "Path Movement",
    "Wrist Twist", "Selected Fingers", "Thumb Position", "Flexion", "Spread",
    "Spread Change", "Nondominant Handshape", "Handshape",
]
# PopSign ships these 6 per clip (its own key -> the canonical name). They come
# from the clip JSON and are NEVER overwritten by the gloss join.
NATIVE_KEY = {
    "Handshape": "handshape", "Selected Fingers": "selected_fingers",
    "Flexion": "flexion", "Spread": "spread",
    "Thumb Position": "thumb_position", "Thumb Contact": "thumb_contact",
}
NATIVE = list(NATIVE_KEY)                                    # 6
JOINED = [f for f in FEATURE_ORDER_16 if f not in NATIVE_KEY]  # 10
assert len(NATIVE) == 6 and len(JOINED) == 10, (NATIVE, JOINED)

FEATURES = FEATURE_ORDER_16  # emit in canonical order, not native-then-joined
NULL_TOKEN = "NA"
SPLITS = ["train", "val", "test"]

# Unambiguous English-label differences for the same sign concept. Kept short
# and hand-checked; anything requiring judgement is left unmatched instead.
SYNONYMS = {
    "dad": "father", "mom": "mother", "grandma": "grandmother",
    "grandpa": "grandfather", "tv": "television", "kitty": "cat",
    "haveto": "must", "hesheit": "he", "callonphone": "telephone",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


def load_asllex() -> dict:
    """gloss -> row of ASL-LEX features, from the SEM-LEX metadata."""
    table = {}
    with open(SEMLEX_METADATA_DEDUPED) as f:
        for r in csv.DictReader(f):
            if r.get("label_type") != "asllex":
                continue
            table.setdefault(norm(r["label"]), r)
    if not table:
        sys.exit(f"no asllex rows in {SEMLEX_METADATA_DEDUPED}")
    return table


def resolve_gloss(gloss: str, table: dict) -> tuple[str | None, str]:
    """Return (matched_key, rule) or (None, 'unmatched')."""
    g = norm(gloss)
    if g in table:
        return g, "exact"
    for suffix in (" 2", " 1"):
        if g + suffix in table:
            return g + suffix, "variant"
    syn = SYNONYMS.get(g)
    if syn and norm(syn) in table:
        return norm(syn), "synonym"
    return None, "unmatched"


def read_split(split: str):
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
            # keyed by CANONICAL name; PopSign's own key is looked up per field
            vals = {"Handshape": r.get("handshape")}
            for canon, pkey in NATIVE_KEY.items():
                if canon == "Handshape":
                    continue
                vals[canon] = sub.get(pkey)
            yield clip_id, r.get("sign", ""), vals


def normalise(v):
    if v is None or v == "":
        return NULL_TOKEN
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--null-policy", default="class", choices=("class", "ignore"),
                    help="'class' (default) makes NA a real class as in "
                         "the 6-feature build; 'ignore' emits -1 so "
                         "CrossEntropyLoss skips NA -- kept only to reproduce "
                         "the legacy NA-masked build. "
                         "Unmatched glosses' joined features are -1 under "
                         "EITHER policy (unknown is not NA).")
    ap.add_argument("--out-pkl", type=Path,
                    default=PROCESSED_DIR / "feature_labels16.pkl")
    ap.add_argument("--out-vocab", type=Path,
                    default=RESULTS_DIR / "feature_vocab16.json")
    args = ap.parse_args()

    table = load_asllex()
    print(f"ASL-LEX glosses: {len(table)}")

    # Pass 1: collect raw values per split.
    raw = {s: {} for s in SPLITS}
    rules = Counter()
    gloss_rule = {}
    for split in SPLITS:
        for clip_id, gloss, vals in read_split(split):
            key, rule = resolve_gloss(gloss, table)
            if norm(gloss) not in gloss_rule:
                gloss_rule[norm(gloss)] = rule
                rules[rule] += 1
            row = table.get(key) if key else None
            merged = {f: normalise(vals.get(f)) for f in NATIVE}
            for f in JOINED:
                merged[f] = normalise(row[f]) if row else None  # None = unmatched
            raw[split][clip_id] = merged
    print(f"gloss match rules: {dict(rules)}")
    matched = sum(v for k, v in rules.items() if k != "unmatched")
    print(f"classes matched: {matched}/{sum(rules.values())} "
          f"= {matched/max(1,sum(rules.values())):.1%}")

    # Pass 2: label space from TRAIN ONLY, NA last (same rule as the 6-build).
    vocab = {}
    for f in FEATURES:
        vals = {v for v in (r[f] for r in raw["train"].values()) if v is not None}
        ordered = sorted(v for v in vals if v != NULL_TOKEN)
        if NULL_TOKEN in vals:
            ordered.append(NULL_TOKEN)
        vocab[f] = {v: i for i, v in enumerate(ordered)}

    # Pass 3: encode.
    out = {}
    unseen = defaultdict(set)
    n_ignored = Counter()
    for split in SPLITS:
        for clip_id, vals in raw[split].items():
            vec = []
            for f in FEATURES:
                v = vals[f]
                if v is None:  # unmatched gloss, joined feature
                    vec.append(-1)
                    n_ignored[f] += 1
                    continue
                if v not in vocab[f]:
                    unseen[f].add(v)
                    vec.append(-1)
                    continue
                idx = vocab[f][v]
                if args.null_policy == "ignore" and v == NULL_TOKEN:
                    idx = -1
                vec.append(idx)
            out[clip_id] = vec

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as f:
        pickle.dump(out, f)

    write_json(args.out_vocab, {
        "created_at": utcnow(), "git_sha": git_sha(),
        "source": "PopSign JSON (6 native) + ASL-LEX gloss join (10 lexical)",
        "asllex_table": str(SEMLEX_METADATA_DEDUPED),
        "join_validation": "handshape agrees 133/133 on overlapping glosses",
        "null_policy": args.null_policy, "null_token": NULL_TOKEN,
        "features": FEATURES, "native_features": NATIVE, "joined_features": JOINED,
        "feature_sizes": [len(vocab[f]) for f in FEATURES],
        "vocab": vocab,
        "gloss_match_rules": dict(rules),
        "synonyms_used": SYNONYMS,
        "n_clips": len(out),
        "n_ignored_per_feature": dict(n_ignored),
        "values_unseen_in_train": {k: sorted(v) for k, v in unseen.items()},
    })
    print(f"wrote {len(out)} clips -> {args.out_pkl}")
    print(f"feature sizes: {[len(vocab[f]) for f in FEATURES]}")
    print(f"-> {args.out_vocab}")


if __name__ == "__main__":
    main()
