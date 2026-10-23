"""Zero-shot cross-lingual eval: a SemLex(ASL)-trained SHuBERT phonology model
on the 37 LSC benchmark handshape clips.

The per-clip CSV is the output that matters: common/lsc_eval scores it
(convert_legacy.py -> evaluate.py) with the conventions shared by every model
family, and those are the reported numbers. The metrics JSON written here is a
quick in-leaf summary under the older conventions below (e.g. spread skipped
on the spread-free handshapes) and is not comparable to them.

The model's 16 heads are trained on ASL-LEX. Five of them -- Selected Fingers,
Flexion, Spread, Thumb Position, Thumb Contact -- are directly comparable with
the LSC labels (labels/README.txt: "these five attribute heads ARE directly
comparable"). The Handshape head is NOT: its classes are ASL handshape ids and
the LSC ids come from a different vocabulary. LSC handshape identity is instead
recovered by snapping the five predicted attributes onto the nearest valid LSC
configuration with lsc_handshape_distance.LSCHandshapeMapper.

Id spaces differ on BOTH sides and overlap numerically without agreeing, so
everything is matched through canonical STRINGS, never raw ints:
    selected_fingers id 2 = "imp" (SemLex) but "imr" (benchmark)
    thumb_contact    id 1 = "1.0" (SemLex) but "1"   (benchmark)
Spellings differ too ("Fully Open" vs "FullyOpen", "__NA__" vs "NA").

In-leaf summary metrics
  attribute Recall@1   per head, over the 5 comparable attributes
  handshape accuracy   after distance-snapping, three ways:
      strict   the single deterministic name must equal the truth
      tie-any  correct if truth is among the tied candidates (the README's
               suggested reading -- 57.5% of inputs are ties)
      oracle-  exact-match rate BEFORE snapping (is_exact)
  mean snap distance   how far predictions sit from any real LSC handshape

Run under shubert_train_env.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SEMLEX_REPO = Path(__file__).resolve().parent.parent.parent

# The LSC handshape benchmark is a separate release. ASL2LSC_LSC_BENCH is its
# root (labels/ holds lsc_handshape_distance.py); ASL2LSC_LSC_LABELS holds the
# 37-clip gold labels and the per-split feature manifests.
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
LSC_LABELS = _bench_dir("ASL2LSC_LSC_LABELS")

sys.path.insert(0, str(LABELS))
sys.path.insert(1, str(SEMLEX_REPO / "40_train"))
sys.path.insert(2, str(SEMLEX_REPO))

from lsc_handshape_distance import (  # noqa: E402
    FLEXION_CANON, SPREAD_CANON, THUMB_CONTACT_CANON, THUMB_POSITION_CANON,
    LSCHandshapeMapper, _canon, _canon_fingers)
from ablation import resolve_channels  # noqa: E402
from dataset import collate  # noqa: E402
from model import PhonologyClassifier  # noqa: E402
from model_loading import load_backbone  # noqa: E402

# SemLex head name -> benchmark audit-CSV column
ATTRS = {
    "Selected Fingers": "selected_fingers",
    "Flexion": "flexion",
    "Spread": "spread",
    "Thumb Position": "thumb_position",
    "Thumb Contact": "thumb_contact",
}

# Handshapes for which SPREAD IS NOT CONTRASTIVE, so it must not be scored.
#
# In LSC, spread distinguishes 5 (spread) from B (unspread) for the extended
# and curved-open configurations only. For flat-open, flat-closed,
# curved-closed and closed the two collapse into a single handshape: SignHub
# 1.1.1.2 prints ONE image spanning both rows (rowspan=2), and sheet_map.csv
# has no 5-* row for them at all. The benchmark files them under B-* with
# Spread=0, which would penalise a model that predicts spread on a handshape
# where spread carries no information. Confirmed with the signer.
#
# Keyed by the sheet handshape name (the audit CSV's gold identity).
SPREAD_FREE_HANDSHAPES = {
    "B-flat-open", "B-flat-closed", "B-curved-closed", "B-closed",
}

# Canonicalisation to a shared string space. Applied to BOTH sides, so the
# comparison never depends on either side's integer ids or spelling.
_FLEX = {"fully open": "FullyOpen", "fullyopen": "FullyOpen",
         "fully closed": "FullyClosed", "fullyclosed": "FullyClosed",
         "curved": "Curved/Bent", "bent": "Curved/Bent",
         "curved/bent": "Curved/Bent", "flat": "Flat",
         "stacked": "Stacked", "crossed": "Crossed"}


def canon(head: str, v: str) -> str:
    """One canonical spelling per value, for both SemLex and benchmark sides."""
    s = str(v).strip()
    if s in ("__NA__", "", "NA", "N/A", "nan", "None"):
        return "NA"
    if head == "Flexion":
        return _FLEX.get(s.lower(), s)
    if head in ("Spread", "Thumb Contact"):
        # "1.0"/"1" -> "1"; tolerate floats from either vocabulary
        try:
            return str(int(float(s)))
        except ValueError:
            return s
    return s  # Selected Fingers, Thumb Position are already identical


_CHANNEL_TABLES = {"flexion": FLEXION_CANON, "spread": SPREAD_CANON,
                   "thumb_position": THUMB_POSITION_CANON,
                   "thumb_contact": THUMB_CONTACT_CANON}


def lsc_inventory_indices(int2str, mapper):
    """Per head: which of the model's class ids denote a value that occurs in
    the LSC handshape inventory.

    Zero-shot-legal constraint: it uses only the target label space (the
    43-handshape inventory), never the LSC videos. Values the inventory
    rejects outright (Flexion NA/Stacked/Crossed, ASL-only finger combos) and
    values it accepts but never uses are both excluded, so a constrained
    argmax can never produce an unmappable prediction and never wastes
    probability mass on values no LSC handshape has.
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
        for i, s in sorted(int2str[h].items()):
            c = channel_canon(col, canon(h, s))
            (keep if c is not None and c in inv_vals[col] else drop).append((i, s))
        if not keep:
            raise SystemExit(f"--constrain-lsc left no valid classes for {h!r}")
        allowed[h] = torch.tensor([i for i, _ in keep])
        dropped[h] = [s for _, s in drop]
    return allowed, dropped


class BenchDataset(Dataset):
    """The benchmark clips, read from the 5-column SHuBERT input manifest.

    Same layout the trainer's SemLexNpyDataset expects -- (T,384) face/hand1/
    hand2 + (T,14) body -- but that class hardcodes SemLex paths and label CSVs,
    so the manifest is read directly here. Labels are attached by ROW ORDER:
    test.txt is documented as "one row per manifest line, in manifest order".
    """

    def __init__(self, manifest: Path):
        self.rows = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            face, h1, h2, body, n = line.split("\t")
            stem = Path(face).name[: -len("_face.npy")]
            self.rows.append((face, h1, h2, body, int(n), stem))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        face, h1, h2, body, n, stem = self.rows[i]
        streams = [
            torch.from_numpy(np.array(np.load(p, mmap_mode="r")[:n], dtype=np.float32))
            for p in (face, h1, h2, body)
        ]
        # collate() needs a label tensor; unused (we read logits directly).
        return (*streams, torch.zeros(16, dtype=torch.int64), stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("test", "examples"), default="test",
                    help="test = the 37 demonstrated configurations; examples = "
                         "the 37 example SIGNS that follow each one")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: $ASL2LSC_LSC_LABELS/"
                         "shubert_inference_handshape_<split>_v2.csv")
    ap.add_argument("--audit", type=Path,
                    default=LSC_LABELS / "test_labels_audit_v2.csv")
    ap.add_argument("--run-label", default="main")
    ap.add_argument("--out-dir", type=Path,
                    default=SEMLEX_REPO / "results" / "lsc_eval_v2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--constrain-lsc", action="store_true",
                    help="restrict each head's argmax to values that occur in "
                         "the LSC handshape inventory (uses only the target "
                         "label space, no LSC videos -- still zero-shot)")
    ap.add_argument("--label-maps", default="label_maps.json",
                    help="label-maps file in results/ matching the CORPUS the "
                         "checkpoint was trained on (popsign runs need "
                         "popsign16_label_maps.json). Mismatch would decode "
                         "head ids through the wrong string table -- the "
                         "n_classes cross-check below catches it.")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    feature_order = state["feature_order"]
    spec = (state.get("config") or {}).get("keep_channels")
    keep = resolve_channels(spec) if spec else None
    if keep:
        print(f"ABLATION (from ckpt config): keeping {list(keep)}")
    model = PhonologyClassifier(load_backbone(), state["n_classes"], keep_channels=keep)
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # id -> string for the model's own heads
    maps_data = json.loads(
        (SEMLEX_REPO / "results" / args.label_maps).read_text())
    maps = maps_data["maps"]
    # Guard against decoding a checkpoint through the wrong corpus's table:
    # per-head class counts must match the checkpoint exactly.
    map_counts = [len(maps[name]) for name in feature_order]
    if map_counts != list(state["n_classes"]):
        raise SystemExit(
            f"--label-maps {args.label_maps} does not match this checkpoint: "
            f"per-head class counts {map_counts} vs ckpt {state['n_classes']}. "
            f"PopSign checkpoints need popsign16_label_maps.json.")
    int2str = {h: {v: k for k, v in maps[h].items()} for h in ATTRS}

    manifest = args.manifest or (
        LSC_LABELS / f"shubert_inference_handshape_{args.split}_v2.csv")
    truth = list(csv.DictReader(args.audit.open()))
    ds = BenchDataset(manifest)
    if len(ds) != len(truth):
        raise SystemExit(f"manifest has {len(ds)} rows but audit has {len(truth)}; "
                         "row-order label alignment would be wrong")
    order_stems = [r[5] for r in ds.rows]

    # The examples split reuses the CONFIGURATION labels: each example sign is
    # the sign demonstrating the configuration recorded immediately before it,
    # so it carries the same handshape. Its stem differs from the config's
    # (own timestamps + "_example_sign"), so stems cannot be the join key --
    # the label embedded in the stem is. Verified: the labels agree in order
    # between the two manifests, and each example starts where its config ends
    # (2 clips differ by 12 ms, i.e. one frame).
    def stem_label(stem: str) -> str:
        m = re.match(r"^\d+_hs_\d+_\d+_(.+)$", stem)
        if not m:
            raise SystemExit(f"unparseable clip stem: {stem}")
        return m.group(1).removesuffix("_example_sign")

    stem2truth = {}
    truth_by_label = {stem_label(r["clip_stem"]): r for r in truth}
    for stem in order_stems:
        lab = stem_label(stem)
        if lab not in truth_by_label:
            raise SystemExit(f"no label row for {stem!r} (label {lab!r})")
        stem2truth[stem] = truth_by_label[lab]
    # Order must agree too, or a duplicated label (3-extended and
    # middle-flat_open each occur twice) could silently pair the wrong rows.
    if [stem_label(s) for s in order_stems] != [
            stem_label(r["clip_stem"]) for r in truth]:
        raise SystemExit("manifest label order differs from the audit CSV; "
                         "row alignment cannot be trusted")

    mapper = LSCHandshapeMapper()
    allowed = None
    if args.constrain_lsc:
        allowed, dropped_cls = lsc_inventory_indices(int2str, mapper)
        for h in ATTRS:
            print(f"constrain-lsc {h}: keeping {len(allowed[h])}/"
                  f"{len(int2str[h])} classes, dropping {dropped_cls[h]}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate)
    head_idx = {h: feature_order.index(h) for h in ATTRS}
    preds = {}
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(b["face"], b["hand1"], b["hand2"], b["body"],
                           b["padding_mask"], b["lengths"])
            for j, stem in enumerate(batch["ids"]):
                preds[stem] = {}
                for h in ATTRS:
                    vec = logits[head_idx[h]][j].detach().cpu()
                    if allowed is not None:
                        ids = allowed[h]
                        k = int(ids[int(vec[ids].argmax())])
                    else:
                        k = int(vec.argmax())
                    preds[stem][h] = int2str[h][k]
    per_attr = {h: 0 for h in ATTRS}
    strict = tie_any = exact = unmappable = 0
    dist_sum = 0.0
    rows = []
    for stem in order_stems:
        t, p = stem2truth[stem], preds[stem]
        hit = {}
        spread_free = t["sheet_handshape_name"] in SPREAD_FREE_HANDSHAPES
        for h, col in ATTRS.items():
            if h == "Spread" and spread_free:
                # Spread is not contrastive for this handshape: any prediction
                # (0, 1 or NA) is accepted. Scoring it would measure a
                # distinction the LSC inventory does not make.
                ok = True
            else:
                ok = canon(h, p[h]) == canon(h, t[col])
            per_attr[h] += ok
            hit[h] = ok
        # Feed the mapper CANONICAL strings: the model emits "0.0"/"1.0"/"__NA__"
        # where the LSC inventory writes "0"/"1"/"NA". Same values, different
        # notation -- passing the raw ones makes HandshapeFeatures reject every
        # clip. Flexion NA/Stacked/Crossed have no LSC counterpart at all (the
        # module rejects them rather than guessing); those clips cannot be
        # snapped and are counted as handshape-incorrect, tracked separately.
        gold = t["sheet_handshape_name"]
        try:
            m = mapper.map(
                selected_fingers=canon("Selected Fingers", p["Selected Fingers"]),
                flexion=canon("Flexion", p["Flexion"]),
                spread=canon("Spread", p["Spread"]),
                thumb_position=canon("Thumb Position", p["Thumb Position"]),
                thumb_contact=canon("Thumb Contact", p["Thumb Contact"]))
        except ValueError as e:
            unmappable += 1
            rows.append({
                "clip_stem": stem, "gold_handshape": gold,
                "pred_handshape": "", "pred_candidates": "",
                "snap_distance": "", "is_tie": "", "was_exact": False,
                "strict_ok": False, "tie_any_ok": False, "unmappable": str(e)[:80],
                **{f"gold_{c}": t[c] for c in ATTRS.values()},
                **{f"pred_{ATTRS[h]}": p[h] for h in ATTRS},
                **{f"ok_{ATTRS[h]}": hit[h] for h in ATTRS},
            })
            continue
        cand = set(m.handshape_names)
        for feats, names in m.alternatives:
            cand.update(names)
        s_ok, t_ok = (m.handshape_name == gold), (gold in cand)
        if spread_free and not s_ok:
            # Same free pass on identity: re-snap with the opposite spread and
            # with spread unset. If any of those lands on the gold handshape,
            # the model only "missed" on a feature this handshape does not
            # encode. Cheap -- at most two extra snaps, on <=4 clips per split.
            for alt_spread in ("0", "1", "NA"):
                if alt_spread == canon("Spread", p["Spread"]):
                    continue
                try:
                    m_alt = mapper.map(
                        selected_fingers=canon("Selected Fingers", p["Selected Fingers"]),
                        flexion=canon("Flexion", p["Flexion"]),
                        spread=alt_spread,
                        thumb_position=canon("Thumb Position", p["Thumb Position"]),
                        thumb_contact=canon("Thumb Contact", p["Thumb Contact"]))
                except ValueError:
                    continue
                alt_cand = set(m_alt.handshape_names)
                for feats, names in m_alt.alternatives:
                    alt_cand.update(names)
                if m_alt.handshape_name == gold:
                    m, cand, s_ok, t_ok = m_alt, alt_cand, True, True
                    break
                if gold in alt_cand:
                    cand, t_ok = cand | alt_cand, True
        strict += s_ok
        tie_any += t_ok
        exact += m.is_exact
        dist_sum += m.distance
        rows.append({
            "clip_stem": stem, "gold_handshape": gold,
            "pred_handshape": m.handshape_name,
            "pred_candidates": "|".join(sorted(cand)),
            "snap_distance": m.distance, "is_tie": m.is_tie,
            "was_exact": m.is_exact, "strict_ok": s_ok, "tie_any_ok": t_ok,
            "unmappable": "",
            **{f"gold_{c}": t[c] for c in ATTRS.values()},
            **{f"pred_{ATTRS[h]}": p[h] for h in ATTRS},
            **{f"ok_{ATTRS[h]}": hit[h] for h in ATTRS},
        })

    n = len(order_stems)
    attr_r1 = {h: round(per_attr[h] / n, 4) for h in ATTRS}
    metrics = {
        "run_label": args.run_label, "split": args.split,
        "manifest": str(manifest), "ckpt": str(args.ckpt),
        "ckpt_epoch": state["epoch"], "n": n,
        "keep_channels": list(keep) if keep else "all",
        "constrain_lsc": bool(args.constrain_lsc),
        "attribute_recall1": attr_r1,
        "macro_attribute_recall1": round(sum(attr_r1.values()) / len(attr_r1), 4),
        # Accuracies are over all n clips: an unmappable prediction is a wrong
        # handshape, not an excused one. Mean distance averages only over the
        # mappable ones (the others have no defined distance).
        "handshape_acc_strict": round(strict / n, 4),
        "handshape_acc_tie_any": round(tie_any / n, 4),
        "pred_already_valid_lsc": round(exact / n, 4),
        "n_unmappable": unmappable,
        "mean_snap_distance": (round(dist_sum / (n - unmappable), 4)
                               if n - unmappable else None),
    }
    tag = f"{args.run_label}_{args.split}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"lsc_eval_{tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    with (args.out_dir / f"lsc_eval_{tag}_per_clip.csv").open(
            "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(metrics, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
