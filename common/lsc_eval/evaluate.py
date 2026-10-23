#!/usr/bin/env python3
"""Score model predictions of the five ASL-LEX phonological streams on the
37-clip LSC handshape benchmark (v2 labels).

Input is the model-independent predictions TSV (see README.md): one row per
clip, six tab-separated columns

    clip_id  selected_fingers  flexion  spread  thumb_position  thumb_contact

with attribute values in the model's own spelling -- canonicalisation
happens here, identically for every model.  Whether the model constrained
its argmax to the LSC inventory or not is a prediction-time choice; both
variants are just TSVs and are scored identically.

Protocol:

  attribute Recall@1   per stream, exact match of canonical strings
  handshape metrics    the 5 predicted attributes are snapped onto the
                       nearest valid LSC handshape (lsc_map.py):
      strict           the single deterministic nearest name equals gold
      expected         credit 1/|tied nearest names| when gold is among them,
                       i.e. the accuracy of breaking every tie by a uniform
                       draw.  expected <= recall@1 per clip; strict may
                       exceed it on a tie the arbitrary alphabetical rule
                       happens to win, so strict <= expected is only an
                       aggregate tendency.  Recall@1 pays a full point for a
                       tie the model never resolved; expected pays the
                       1/|tie| a guesser would actually earn.
      recall@k         gold retrieved iff fewer than k distinct feature sets
                       lie strictly closer than gold (ties share the best
                       rank; recall@1 == "tie-any": gold among the tied names)
  Joint-5B golds       the four joint-5B gold handshapes (B-flat-open,
                       B-flat-closed, B-curved-closed, B-closed) are coded
                       unspread -- gold spread 0 (NA for B-closed, where
                       ASL-LEX codes no spread) -- and score like every
                       other clip: the SignHub illustration and the
                       benchmark clips both show an unspread hand.
  unmappable           predictions with no LSC counterpart (flexion NA/
                       Stacked/Crossed, non-imrpt fingers) count as wrong at
                       every k; mean snap distance averages the rest.
  candidate space      the default inventory is the 37 handshapes actually
                       recorded in the benchmark (inventory/
                       lsc_handshapes.csv; the 6 never-recorded classes are
                       commented out there).  Snapping and ranking never
                       nominate a class that cannot be gold.  Pass
                       --inventory inventory/map_ASL_to_LSC.csv for the full
                       43-class space.

Gold labels are joined by the configuration label embedded in the clip stem
(NN_hs_START_END_<label>[_example_sign]), never by row order, so the same
gold file serves both the test and the examples split.

Outputs <out-dir>/<stem>.metrics.json and <out-dir>/<stem>.per_clip.csv.

Stdlib only.  Example:

    python3 evaluate.py predictions/shubert_semlex_test_unconstrained.tsv
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lsc_map import (  # noqa: E402
    STREAMS, canon_loose, strict_features, load_inventory, norm_name,
    snap, gold_rank, expected_recall_at)

REPO = HERE.parent.parent
# The gold labels ship with the LSC benchmark release (ASL2LSC_LSC_LABELS).
DEFAULT_GOLD = Path(os.environ.get(
    "ASL2LSC_LSC_LABELS",
    REPO / "shubert" / "50_eval" / "lsc_labels_v2")) / "test_labels_audit_v2.csv"
DEFAULT_INVENTORY = HERE / "inventory" / "lsc_handshapes.csv"

RECALL_KS = (1, 3, 5)
EXPECTED_N = 37

_STEM_RE = re.compile(r"^\d+_hs_\d+_\d+_(.+?)(?:_example_sign)?$")


def stem_label(clip_id):
    m = _STEM_RE.match(clip_id)
    if not m:
        raise SystemExit("unparseable clip id: %r" % (clip_id,))
    # Leaves spell the embedded label differently: "index+thumb-flat_closed"
    # (slgcn/popsign) vs "indexpthumb-flat_closed" (the audit stems), and
    # case varies ("Y-extended" / "n-extended").  Normalise the join key.
    return m.group(1).lower().replace("+", "p")


def load_gold(path):
    """Audit CSV -> {embedded label: row}.  Duplicate labels must agree."""
    if not Path(path).is_file():
        raise SystemExit("no gold labels at %s -- set ASL2LSC_LSC_LABELS to the "
                         "LSC benchmark's label directory, or pass --gold"
                         % (path,))
    gold = {}
    with open(str(path), newline="") as fh:
        for r in csv.DictReader(fh):
            lab = stem_label(r["clip_stem"])
            if lab in gold and gold[lab] != r:
                raise SystemExit(
                    "%s: label %r appears twice with different gold rows"
                    % (path, lab))
            gold[lab] = r
    return gold


def load_predictions(path):
    with open(str(path), newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        need = ("clip_id",) + STREAMS
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit("%s: missing TSV column(s) %s" % (path, missing))
        return list(reader)


def _pack_exp(exp_at):
    """"k:value|..." -- one CSV cell carrying expected recall at every k."""
    return "|".join("%d:%g" % (k, round(exp_at[k], 6))
                    for k in sorted(exp_at))


def evaluate(pred_rows, gold_by_label, inventory, source="", ks=RECALL_KS):
    """Score one predictions TSV; returns (metrics dict, per-clip rows)."""
    per_attr = {s: 0 for s in STREAMS}
    strict = exact = unmappable = 0
    expected = 0.0
    rank_hits = {k: 0 for k in ks}
    exp_rank_hits = {k: 0.0 for k in ks}
    dist_sum = 0.0
    out_rows = []

    for r in pred_rows:
        clip = r["clip_id"]
        t = gold_by_label.get(stem_label(clip))
        if t is None:
            raise SystemExit("%s: no gold row for clip %r" % (source, clip))
        gold_hs = t["sheet_handshape_name"]

        # -- attribute Recall@1 in the loose canonical space ---------------
        hit = {}
        for s in STREAMS:
            hit[s] = canon_loose(s, r[s]) == canon_loose(s, t[s])
            per_attr[s] += hit[s]

        base = {"clip_id": clip, "gold_handshape": gold_hs}
        for s in STREAMS:
            base["gold_" + s] = t[s]
            base["pred_" + s] = r[s]
            base["ok_" + s] = hit[s]

        # -- snap onto the LSC inventory -----------------------------------
        try:
            q = strict_features(*[canon_loose(s, r[s]) for s in STREAMS])
        except ValueError as e:
            unmappable += 1
            out_rows.append(dict(
                base, pred_lsc_handshape="", pred_candidates="", gold_rank="",
                pred_top5="", snap_distance="", is_tie="", was_exact=False,
                strict_ok=False, expected_credit=0.0,
                expected_recall=_pack_exp({k: 0.0 for k in ks}),
                recall_at_1=False, unmappable=str(e)[:80]))
            continue

        m = snap(q, inventory)
        cand = list(m["candidates"])
        s_ok = norm_name(m["name"]) == norm_name(gold_hs)
        t_ok = any(norm_name(c) == norm_name(gold_hs) for c in cand)

        best_rank = gold_rank(m["ranked"], gold_hs)
        if best_rank is None:
            raise SystemExit(
                "gold handshape %r is not in the inventory" % (gold_hs,))

        # Internal consistency: recall@1 must equal the tie-any rule.
        if (best_rank <= 1) != t_ok:
            raise RuntimeError(
                "%s: rank/tie-any disagreement on %r" % (source, clip))

        # Expected accuracy if the tie were broken by a uniform draw over the
        # tied NAMES (not feature tuples): the two provably-conflated pairs
        # are two distinct identities a guesser must choose between, so they
        # belong in the denominator.  Degenerates to s_ok when |cand| == 1.
        credit = (1.0 / len(cand)) if t_ok else 0.0
        # Only credit <= recall@1 holds per clip.  strict can EXCEED credit on
        # a tie the alphabetical rule happens to win, so strict <= expected is
        # an aggregate tendency, never a per-clip invariant -- do not assert it.
        if credit > (1.0 if t_ok else 0.0) + 1e-12:
            raise RuntimeError(
                "%s: expected credit exceeds tie-any on %r" % (source, clip))

        # Expected recall@k under the same uniform tie-breaking.  At k=1 it
        # must reproduce `credit` exactly -- the two are independent code
        # paths (candidate set vs the full ranking), so this is a real gate.
        exp_at = {}
        for k in ks:
            v = expected_recall_at(m["ranked"], gold_hs, k)
            if v is None:
                raise SystemExit(
                    "gold handshape %r is not in the inventory" % (gold_hs,))
            exp_at[k] = v
        if 1 in exp_at and abs(exp_at[1] - credit) > 1e-9:
            raise RuntimeError(
                "%s: expected recall@1 %r != expected_credit %r on %r"
                % (source, exp_at[1], credit, clip))

        strict += s_ok
        expected += credit
        exact += m["is_exact"]
        dist_sum += m["distance"]
        for k in ks:
            rank_hits[k] += best_rank <= k
            exp_rank_hits[k] += exp_at[k]
        top5 = "|".join("%s:%d" % (names[0], d)
                        for d, _, names in m["ranked"][:5])
        out_rows.append(dict(
            base, pred_lsc_handshape=m["name"],
            pred_candidates="|".join(cand), gold_rank=best_rank,
            pred_top5=top5, snap_distance=m["distance"], is_tie=m["is_tie"],
            was_exact=m["is_exact"], strict_ok=s_ok,
            expected_credit=round(credit, 6),
            expected_recall=_pack_exp(exp_at),
            recall_at_1=best_rank <= 1, unmappable=""))

    n = len(pred_rows)
    attr_r1 = {s: round(per_attr[s] / n, 4) for s in STREAMS}
    metrics = {
        "n": n,
        "attribute_recall1": attr_r1,
        "macro_attribute_recall1": round(sum(attr_r1.values()) / len(attr_r1), 4),
        # Accuracies are over all n clips: an unmappable prediction is a
        # wrong handshape, not an excused one.  Mean distance averages only
        # over the mappable ones (the others have no defined distance).
        "handshape_acc_strict": round(strict / n, 4),
        # Uniform tie-breaking: expected <= recall@1
        # always; vs strict it can fall either way, clip by clip.
        "handshape_acc_expected": round(expected / n, 4),
        "handshape_recall_at": {str(k): round(rank_hits[k] / n, 4)
                                for k in ks},
        # Same uniform tie-breaking generalised to retrieval: a straddling
        # tie contributes room/|tie| instead of a full point.  Equals
        # handshape_acc_expected at k=1 (asserted per clip).
        "handshape_expected_recall_at": {str(k): round(exp_rank_hits[k] / n, 4)
                                         for k in ks},
        "pred_already_valid_lsc": round(exact / n, 4),
        "n_unmappable": unmappable,
        "mean_snap_distance": (round(dist_sum / (n - unmappable), 4)
                               if n - unmappable else None),
    }
    return metrics, out_rows


FIELDS = (["clip_id", "gold_handshape", "pred_lsc_handshape",
           "pred_candidates", "gold_rank", "pred_top5", "snap_distance",
           "is_tie", "was_exact", "strict_ok", "expected_credit",
           "expected_recall", "recall_at_1", "unmappable"]
          + ["gold_" + s for s in STREAMS]
          + ["pred_" + s for s in STREAMS]
          + ["ok_" + s for s in STREAMS])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", type=Path,
                    help="predictions TSV(s), see README.md for the format")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD,
                    help="benchmark gold-label CSV (default: "
                         "$ASL2LSC_LSC_LABELS/test_labels_audit_v2.csv)")
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY,
                    help="valid-LSC-handshape inventory CSV")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--expect-n", type=int, default=EXPECTED_N,
                    help="required row count (0 disables the check); the "
                         "default 37 guards against stale v1 39-clip inputs")
    ap.add_argument("--ks", default=",".join(str(k) for k in RECALL_KS),
                    help="comma-separated k values for handshape recall@k "
                         "(default: %(default)s); any k works, the per-clip "
                         "CSV's gold_rank column is the underlying ranking")
    args = ap.parse_args()

    try:
        ks = tuple(sorted({int(k) for k in args.ks.split(",") if k.strip()}))
    except ValueError:
        raise SystemExit("--ks must be comma-separated integers, got %r"
                         % (args.ks,))
    if not ks or min(ks) < 1:
        raise SystemExit("--ks needs at least one k >= 1, got %r" % (args.ks,))

    gold = load_gold(args.gold)
    inventory = load_inventory(args.inventory)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tsv in args.predictions:
        rows = load_predictions(tsv)
        if args.expect_n and len(rows) != args.expect_n:
            raise SystemExit("%s: %d rows, expected %d"
                             % (tsv, len(rows), args.expect_n))
        metrics, per_clip = evaluate(rows, gold, inventory, source=str(tsv),
                                     ks=ks)
        def rel(p):  # repo-relative when possible, so results diff cleanly
            try:
                return str(Path(p).resolve().relative_to(REPO))
            except ValueError:
                return str(p)
        metrics = dict({"predictions": tsv.name, "gold": rel(args.gold),
                        "inventory": rel(args.inventory)}, **metrics)
        stem = tsv.stem
        mpath = args.out_dir / (stem + ".metrics.json")
        cpath = args.out_dir / (stem + ".per_clip.csv")
        with open(str(mpath), "w") as fh:
            json.dump(metrics, fh, indent=2)
            fh.write("\n")
        with open(str(cpath), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(per_clip)
        print(json.dumps(metrics, indent=2))
        print("-> %s" % mpath)


if __name__ == "__main__":
    main()
