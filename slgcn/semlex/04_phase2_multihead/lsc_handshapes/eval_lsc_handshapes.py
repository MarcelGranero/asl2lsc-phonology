"""Zero-shot cross-lingual eval: the SEM-LEX(ASL)-trained SL-GCN multi-head
model on the LSC benchmark handshape clips.

Mirrors the metric definitions of the SHuBERT harness
(shubert/50_eval/lsc_handshapes/eval_shubert_lsc.py): canonical strings, the
LSCHandshapeMapper snap, the three handshape readings. The per-clip CSV is
the output that matters: common/lsc_eval scores it with the conventions shared
by every model family. The metrics JSON written here is an in-leaf summary
(spread skipped on the spread-free handshapes) and is not comparable to those
numbers.

Why the Handshape head is unusable and the five attributes are not:
the model's Handshape head emits ASL-LEX handshape ids, a different vocabulary
from LSC's 37 `<base>-<variant>` classes (zero exact overlap). But LSC
handshapes are themselves CODED in ASL-LEX attributes (labels/README.txt), and
five of our heads — Selected Fingers, Flexion, Spread, Thumb Position, Thumb
Contact — share those id spaces. So attribute recall is measured directly, and
LSC handshape identity is recovered by snapping the five predicted attributes
onto the nearest valid LSC configuration.

Id spaces overlap numerically WITHOUT agreeing on either side, so every
comparison goes through canonical strings, never raw ints.

Metrics
  attribute Recall@1   per head, over the 5 comparable attributes
  handshape accuracy   after distance-snapping, three ways:
      strict    the deterministic name must equal the truth
      tie-any   correct if truth is among the tied candidates (the README's
                suggested reading — 57.5% of inputs are ties)
      exact     fraction whose predicted attribute set was already a valid
                LSC handshape before snapping
  mean snap distance   how far predictions sit from any real LSC handshape
"""
import argparse
import csv
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# LSC handshape benchmark (a separate release): ASL2LSC_LSC_BENCH is its root
# (labels/ holds lsc_handshape_distance.py); ASL2LSC_LSC_LABELS holds the v2
# gold labels (37 clips after two faulty recordings were excluded).
# --- site defaults (optional per-machine values) ---
_SITE_DEFAULTS = {}
# --- end site defaults ---


def _bench_dir(var):
    value = os.environ.get(var) or _SITE_DEFAULTS.get(var)
    if not value:
        raise SystemExit(
            f"{var} is not set. Point it at the LSC handshape benchmark "
            f"(a separate release; see the repository README).")
    return Path(value)


BENCH = _bench_dir("ASL2LSC_LSC_BENCH")
LABELS = BENCH / "labels"
V2_LABELS = _bench_dir("ASL2LSC_LSC_LABELS")
V2_EXCLUDED_PREFIXES = ("06_hs_1807350_1808563", "06_hs_2177280_2178425")

# Spread is not contrastive for the joint 5B handshapes, so it is not scored
# there: LSC separates 5 (spread) from B (unspread) only for the extended and
# curved-open configurations. Keyed by the gold sheet handshape name.
SPREAD_FREE_HANDSHAPES = {
    "B-flat-open", "B-flat-closed", "B-curved-closed", "B-closed",
}

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_phase2_multihead"))
sys.path.insert(0, str(LABELS))

from common import git_sha, utcnow, write_json, write_run_card  # noqa: E402
from data_config import FEATURE_VOCAB_JSON, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from lsc_handshape_distance import (  # noqa: E402
    FLEXION_CANON, SPREAD_CANON, THUMB_CONTACT_CANON, THUMB_POSITION_CANON,
    LSCHandshapeMapper, _canon, _canon_fingers)

LSC_ROOT = PROCESSED_DIR / "lsc_handshapes"

# our head name -> benchmark audit-CSV column
ATTRS = {
    "Selected Fingers": "selected_fingers",
    "Flexion": "flexion",
    "Spread": "spread",
    "Thumb Position": "thumb_position",
    "Thumb Contact": "thumb_contact",
}

_FLEX = {"fully open": "FullyOpen", "fullyopen": "FullyOpen",
         "fully closed": "FullyClosed", "fullyclosed": "FullyClosed",
         "curved": "Curved/Bent", "bent": "Curved/Bent",
         "curved/bent": "Curved/Bent", "flat": "Flat",
         "stacked": "Stacked", "crossed": "Crossed"}


def canon(head: str, v: str) -> str:
    """One canonical spelling per value, for both sides. Same rules as the
    SHuBERT harness — our vocab says "Fully Open"/"n/a", the benchmark says
    "FullyOpen"/"NA"."""
    s = str(v).strip()
    if s in ("__NA__", "", "NA", "N/A", "n/a", "nan", "None"):
        return "NA"
    if head == "Flexion":
        return _FLEX.get(s.lower(), s)
    if head in ("Spread", "Thumb Contact"):
        try:
            return str(int(float(s)))
        except ValueError:
            return s
    return s


_CHANNEL_TABLES = {"flexion": FLEXION_CANON, "spread": SPREAD_CANON,
                   "thumb_position": THUMB_POSITION_CANON,
                   "thumb_contact": THUMB_CONTACT_CANON}


def lsc_inventory_indices(id2label, mapper):
    """Per head: which class ids denote a value occurring in the LSC inventory.

    Same construction as the SHuBERT and MLP harnesses (their
    lsc_inventory_indices), kept identical so "+c" means the same treatment
    for every model family. It reads only the target LABEL SPACE (the
    43-handshape inventory), never the LSC videos or labels, so a
    constrained argmax stays zero-shot. Values with no LSC counterpart at
    all (Flexion NA/Stacked/Crossed, ASL-only finger combinations) and
    values the mapper accepts but no configuration uses are both dropped,
    so a constrained prediction can never be unmappable.
    """
    inv_vals = {c: set() for c in ATTRS.values()}
    for feats, _names in mapper.inventory:
        d = feats.as_dict()
        for c in inv_vals:
            inv_vals[c].add(d[c])

    def channel_canon(col, v):
        try:
            if col == "selected_fingers":
                return _canon_fingers(v)
            return _canon(v, _CHANNEL_TABLES[col], col)
        except ValueError:
            return None  # no LSC counterpart at all

    allowed, dropped = {}, {}
    for h, col in ATTRS.items():
        keep, drop = [], []
        for i, s in enumerate(id2label[h]):
            c = channel_canon(col, canon(h, s))
            (keep if c is not None and c in inv_vals[col] else drop).append((i, s))
        if not keep:
            raise SystemExit(f"--constrain-lsc left no valid classes for {h!r}")
        allowed[h] = torch.tensor([i for i, _ in keep])
        dropped[h] = [s for _, s in drop]
    return allowed, dropped


def stem_label(stem: str) -> str:
    """`06_hs_<start>_<end>_<label>` -> `<label>`, minus any example suffix."""
    m = re.match(r"^\d+_hs_\d+_\d+_(.+)$", stem)
    if not m:
        raise SystemExit(f"unparseable clip stem: {stem}")
    return m.group(1).removesuffix("_example_sign")


def _v2_excluded_indices():
    """Row indices of the two excluded clips, from the CONFIG split's order.

    Both splits are in recording order and pair 1:1 (each example sign follows
    its configuration), so an index established here applies to either.
    """
    p = LSC_ROOT / "gendata_27" / "test_label.pkl"
    with open(p, "rb") as f:
        names, _ = pickle.load(f)
    stems = [n.replace("_color.mp4", "") for n in names]
    idx = {i for i, s in enumerate(stems)
           if s.startswith(V2_EXCLUDED_PREFIXES)}
    if len(idx) != len(V2_EXCLUDED_PREFIXES):
        raise SystemExit(
            f"expected {len(V2_EXCLUDED_PREFIXES)} excluded clips in "
            f"{p}, found {len(idx)} — stem naming may have changed")
    return idx


def load_truth(split: str, order_stems, v1_split: bool = False):
    """Per-clip gold attributes, keyed by clip stem.

    Both splits read the SAME audit CSV. The examples split reuses the
    CONFIGURATION labels: each example sign demonstrates the configuration
    recorded immediately before it, so it carries the same handshape. Its stem
    differs (own timestamps + `_example_sign`), so the join key is the LABEL
    embedded in the stem, not the stem itself.

    This mirrors eval_shubert_lsc.py, including its order check: `3-extended`
    and `middle-flat_open` each occur twice, so label-keyed lookup alone could
    silently pair the wrong duplicate. Requiring the label SEQUENCES to match
    rules that out.
    """
    audit = (LABELS / "test_labels_audit.csv" if v1_split
             else V2_LABELS / "test_labels_audit_v2.csv")
    truth_rows = list(csv.DictReader(audit.open()))
    if len(truth_rows) != len(order_stems):
        raise SystemExit(
            f"audit has {len(truth_rows)} rows but split has {len(order_stems)}; "
            "row-order label alignment would be wrong"
        )

    truth_by_label = {stem_label(r["clip_stem"]): r for r in truth_rows}
    if [stem_label(s) for s in order_stems] != [
            stem_label(r["clip_stem"]) for r in truth_rows]:
        raise SystemExit(
            "clip label order differs from the audit CSV; row alignment "
            "cannot be trusted (duplicate labels exist)"
        )

    out = {}
    for stem in order_stems:
        label = stem_label(stem)
        if label not in truth_by_label:
            raise SystemExit(f"no label row for {stem!r} (label {label!r})")
        row = truth_by_label[label]
        out[stem] = {col: row[col] for col in ATTRS.values()}
        # Carried through for the spread-free rule: the CLIP label is the
        # filename spelling ("5-flat_open"), while the joint 5B handshapes
        # are identified by their SHEET name ("B-flat-open").
        out[stem]["sheet_handshape_name"] = row.get("sheet_handshape_name", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--regime", choices=["pfr", "isr_pfr"], required=True)
    ap.add_argument("--split", choices=["test", "examples"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "02_train/config/train_joint_paper.yaml")
    ap.add_argument("--feature-vocab", type=Path, default=FEATURE_VOCAB_JSON)
    ap.add_argument("--variant", default="",
                    help="preprocessing variant to score, e.g. '30fps' "
                         "(reads gendata_27_<variant>, writes "
                         "lsc_handshapes_<split>_<variant>.json)")
    ap.add_argument("--constrain-lsc", action="store_true",
                    help="restrict each head's argmax to values occurring in "
                         "the LSC handshape inventory (still zero-shot; "
                         "writes lsc_handshapes_<split><variant>_cons.*)")
    ap.add_argument("--v1-split", action="store_true",
                    help="score the ORIGINAL 39-clip split with spread scored "
                         "for every handshape. Default is v2: 37 clips, "
                         "spread not scored for the joint 5B handshapes.")
    args = ap.parse_args()

    cfg = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    fv = json.loads(args.feature_vocab.read_text())
    feature_names = fv["features"]
    feature_sizes = [fv["per_feature"][f]["n_classes"] for f in feature_names]
    id2label = {f: fv["per_feature"][f]["classes"] for f in feature_names}

    suffix = f"_{args.variant}" if args.variant else ""
    # Output-only tag: `suffix` also selects the gendata dir, which the
    # constraint must not redirect (it is a decoding choice, same inputs).
    out_suffix = suffix + ("_cons" if args.constrain_lsc else "")
    gendata_dir = LSC_ROOT / f"gendata_27{suffix}"
    data = np.load(gendata_dir / f"{args.split}_data_joint.npy")
    with open(gendata_dir / f"{args.split}_label.pkl", "rb") as f:
        sample_names, _ = pickle.load(f)
    stems = [n.replace("_color.mp4", "") for n in sample_names]

    if not args.v1_split:
        # v2 labels: two clips were excluded as faulty recordings.
        # data and stems are row-aligned, so both
        # must be filtered together -- dropping one would silently mis-pair
        # every prediction after the first excluded row.
        #
        # The EXAMPLES split cannot be matched on those stems: each example
        # sign carries its own timestamps, so only the configuration split
        # contains them. Both splits are in recording order and pair 1:1, so
        # the excluded ROW INDICES are established from the config split and
        # applied positionally to either.
        drop_idx = _v2_excluded_indices()
        keep = [i for i in range(len(stems)) if i not in drop_idx]
        dropped = len(stems) - len(keep)
        if dropped != len(drop_idx):
            raise SystemExit(
                f"v2 exclusion dropped {dropped} clips, expected "
                f"{len(drop_idx)} — split length or ordering may have changed")
        data = data[keep]
        stems = [stems[i] for i in keep]
        print(f"v2 labels: dropped {dropped} faulty clips, {len(stems)} remain")

    truth = load_truth(args.split, stems, v1_split=args.v1_split)

    from model_multihead import MultiheadModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiheadModel(
        feature_sizes=feature_sizes,
        with_gloss=(args.regime == "isr_pfr"),
        **cfg["model_args"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    mapper = LSCHandshapeMapper()
    allowed = None
    if args.constrain_lsc:
        allowed, dropped_cls = lsc_inventory_indices(id2label, mapper)
        for h in ATTRS:
            print(f"constrain-lsc {h}: keeping {len(allowed[h])}/"
                  f"{len(id2label[h])} classes, dropping {dropped_cls[h]}")

    head_idx = {h: feature_names.index(h) for h in ATTRS}
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), 16):
            batch = torch.from_numpy(data[i:i + 16]).float().to(device)
            _, feature_logits = model(batch)
            for j in range(batch.shape[0]):
                row = {}
                for h in ATTRS:
                    logits = feature_logits[head_idx[h]][j]
                    if allowed is None:
                        row[h] = id2label[h][int(logits.argmax())]
                    else:
                        # argmax within the allowed subset, then back to the
                        # model's own class id
                        keep = allowed[h].to(logits.device)
                        row[h] = id2label[h][int(keep[logits[keep].argmax()])]
                preds.append(row)

    per_attr = {h: 0 for h in ATTRS}
    n_strict = n_tie = n_exact = n_unmappable = 0
    snap_distances = []
    rows = []

    for stem, p in zip(stems, preds):
        t = truth[stem]
        gold_name = stem_label(stem)
        # Spread carries no contrast for the joint 5B handshapes, so any
        # prediction counts there (see SPREAD_FREE_HANDSHAPES).
        spread_free = (not args.v1_split
                       and t["sheet_handshape_name"] in SPREAD_FREE_HANDSHAPES)
        hit = {}
        for h, col in ATTRS.items():
            if h == "Spread" and spread_free:
                ok = True
            else:
                ok = canon(h, p[h]) == canon(h, t[col])
            hit[h] = ok
            per_attr[h] += int(ok)
        # The heads can emit values that are outside the mapper's ASL-LEX
        # inventory entirely — our Flexion vocabulary includes 'n/a',
        # 'Stacked' and 'Crossed', while the mapper accepts only the four
        # values that occur in real LSC configurations. Those predictions
        # cannot be snapped at all (they are not merely far from a valid
        # handshape, they are off the axis), so they count as UNMAPPABLE
        # rather than crashing the run or being silently coerced to some
        # substitute value that would fake a plausible prediction.
        try:
            r = mapper.map(
                selected_fingers=canon("Selected Fingers", p["Selected Fingers"]),
                flexion=canon("Flexion", p["Flexion"]),
                spread=canon("Spread", p["Spread"]),
                thumb_position=canon("Thumb Position", p["Thumb Position"]),
                thumb_contact=canon("Thumb Contact", p["Thumb Contact"]),
            )
        except ValueError as exc:
            n_unmappable += 1
            rows.append({
                "clip": stem,
                "gold_handshape": gold_name,
                "pred_handshape": "<unmappable>",
                "pred_handshape_all": "<unmappable>",
                "snap_distance": "",
                "is_tie": False,
                "is_exact": False,
                "strict": False,
                "tie_any": False,
                "unmappable_reason": str(exc),
                **{f"gold_{c}": t[c] for c in ATTRS.values()},
                **{f"pred_{ATTRS[h]}": p[h] for h in ATTRS},
                **{f"ok_{ATTRS[h]}": hit[h] for h in ATTRS},
            })
            continue
        snap_distances.append(r.distance)
        n_exact += int(bool(r.is_exact))

        def norm(name):
            return name.replace("-", "_").replace("__", "_").lower()

        # `handshape_names` are the names of the WINNING feature set only.
        # `alternatives` carries the other feature sets sitting at the same
        # minimum distance — the README is explicit that these are different
        # things, and tie-any is only meaningful if the alternatives are
        # included (57.5% of all possible inputs are ties).
        tied_names = list(r.handshape_names)
        for _features, names in getattr(r, "alternatives", ()):
            tied_names.extend(names)

        strict = norm(r.handshape_names[0]) == norm(gold_name)
        tie_any = any(norm(n) == norm(gold_name) for n in tied_names)
        n_strict += int(strict)
        n_tie += int(tie_any)

        rows.append({
            "clip": stem,
            "gold_handshape": gold_name,
            "pred_handshape": r.handshape_names[0],
            "pred_handshape_all": "|".join(tied_names),
            "snap_distance": r.distance,
            "is_tie": bool(r.is_tie),
            "is_exact": bool(r.is_exact),
            "strict": strict,
            "tie_any": tie_any,
            "unmappable_reason": "",
            **{f"gold_{c}": t[c] for c in ATTRS.values()},
            **{f"pred_{ATTRS[h]}": p[h] for h in ATTRS},
            **{f"ok_{ATTRS[h]}": hit[h] for h in ATTRS},
        })

    n = len(stems)
    metrics = {
        "model": "SL-GCN Sem-Lex phase-2 multihead",
        "checkpoint": str(args.checkpoint),
        "regime": args.regime,
        "split": args.split,
        "variant": args.variant or "native",
        "lsc_labels": "v1 (39 clips)" if args.v1_split else
                      "v2 (37 clips; spread unscored for the joint 5B "
                      "handshapes)",
        "n": n,
        "attribute_recall1": {h: round(per_attr[h] / n, 4) for h in ATTRS},
        "macro_attribute_recall1": round(
            sum(per_attr.values()) / (n * len(ATTRS)), 4),
        # Handshape accuracy is over ALL n clips: an unmappable prediction is
        # a failure to identify the handshape, not an excused sample.
        "handshape_acc_strict": round(n_strict / n, 4),
        "handshape_acc_tie_any": round(n_tie / n, 4),
        "pred_already_valid_lsc": round(n_exact / n, 4),
        "constrain_lsc": bool(args.constrain_lsc),
        "n_unmappable": n_unmappable,
        # Snap distance is only defined for predictions that COULD be snapped,
        # so its denominator excludes them (stated here to keep the two
        # readings from being confused).
        "mean_snap_distance": (round(float(np.mean(snap_distances)), 4)
                               if snap_distances else None),
        "n_snapped": len(snap_distances),
    }

    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"lsc_handshapes_{args.split}{out_suffix}.json", metrics)
    with open(out_dir / f"lsc_handshapes_{args.split}{out_suffix}_per_clip.csv", "w",
              newline="") as f:
        # Union of keys, not just the first row's: unmappable rows carry a
        # different key set and either kind can come first.
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    write_run_card(out_dir, {
        "git_sha": git_sha(),
        "config": {"task": "lsc_handshapes", "split": args.split,
                   "checkpoint": str(args.checkpoint), "regime": args.regime,
                   "constrain_lsc": bool(args.constrain_lsc)},
        "metrics": metrics,
        "finished_at": utcnow(),
    })

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
