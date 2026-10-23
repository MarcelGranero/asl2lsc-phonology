"""Zero-shot cross-lingual eval: a baseline MLP checkpoint on the 37 LSC
benchmark handshape clips, either split.

Mirrors shubert/50_eval/lsc_handshapes/eval_shubert_lsc.py: the same five
comparable attribute heads (the Handshape head is NOT comparable -- ASL ids
vs a 37-class LSC vocabulary with zero overlap), the same canonicalisation to
shared STRINGS, and the same handshape recovery by snapping the predicted
attributes onto the nearest valid LSC configuration (LSCHandshapeMapper).
The per-clip CSV is the output that matters: common/lsc_eval scores it with
the conventions shared by every model family. The metrics JSON written here
is an in-leaf summary (spread skipped on the spread-free handshapes) and is
not comparable to those numbers.

Inputs differ, because the MLP eats hand landmarks rather than DINOv2
features: we read the (T,553,3) Sem-Lex-layout npys that the SL-GCN Sem-Lex
leaf converts from the benchmark's `.pose` files
(04_phase2_multihead/lsc_handshapes/pose_to_553.py), and apply the SAME
low-motion frame selection and wrist normalisation used to build the
training set.

Hand block: `[532:553]` is the person's RIGHT hand = screen-left, the active
articulator in every LSC clip. It also has near-perfect detection here
(98.7% of frames vs 83.5% for the other hand).

--xy-only zeroes the depth channel on BOTH the training statistics and this
eval. It is a diagnostic: LSC poses carry a z-spread about 3.6x flatter than
Sem-Lex's (median z/xy 0.060 vs 0.215) because the two came from different
MediaPipe pipelines, so it separates a real cross-lingual gap from a
coordinate-convention artefact.

--variant reads pose553_<variant>/ instead of pose553/ (an alternative
pose tree for the same clips, e.g. re-cropped video).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "20_train"))
sys.path.insert(0, str(REPO / "10_data_prep"))

from data_config import (  # noqa: E402
    LABEL_MAPS_JSON, LSC_LABELS_DIR, LSC_V2_DIR, POPSIGN_VOCAB16,
    POPSIGN_VOCAB16_NACLASS,
    POSE553_DIR, RESULTS_DIR, RUNS_DIR,
)
from extract_frames import normalize  # noqa: E402
from model import build_model  # noqa: E402

sys.path.insert(0, str(LSC_LABELS_DIR))
from lsc_handshape_distance import (  # noqa: E402
    FLEXION_CANON, SPREAD_CANON, THUMB_CONTACT_CANON, THUMB_POSITION_CANON,
    LSCHandshapeMapper, _canon, _canon_fingers)

RIGHT_HAND = slice(532, 553)      # person's right = screen-left = active

# Model head name -> benchmark audit-CSV column.
ATTRS = {
    "Selected Fingers": "selected_fingers",
    "Flexion": "flexion",
    "Spread": "spread",
    "Thumb Position": "thumb_position",
    "Thumb Contact": "thumb_contact",
}

# Handshapes for which SPREAD IS NOT CONTRASTIVE, so it must not be scored.
# In LSC, spread distinguishes 5 (spread) from B (unspread) for the extended
# and curved-open configurations only; for flat-open, flat-closed,
# curved-closed and closed the two collapse into one handshape (SignHub
# 1.1.1.2 spans a single image across both rows, and sheet_map.csv has no 5-*
# row for them). The benchmark files them under B-* with Spread=0, which would
# penalise a model predicting spread on a handshape that does not encode it.
# Confirmed with the signer. Keyed by sheet handshape name.
SPREAD_FREE_HANDSHAPES = {
    "B-flat-open", "B-flat-closed", "B-curved-closed", "B-closed",
}

_FLEX = {"fully open": "FullyOpen", "fullyopen": "FullyOpen",
         "fully closed": "FullyClosed", "fullyclosed": "FullyClosed",
         "curved": "Curved/Bent", "bent": "Curved/Bent",
         "curved/bent": "Curved/Bent", "flat": "Flat",
         "stacked": "Stacked", "crossed": "Crossed"}

_CHANNEL_TABLES = {"flexion": FLEXION_CANON, "spread": SPREAD_CANON,
                   "thumb_position": THUMB_POSITION_CANON,
                   "thumb_contact": THUMB_CONTACT_CANON}


def canon(head: str, v) -> str:
    """One canonical spelling per value, applied to BOTH sides."""
    s = str(v).strip()
    if s in ("__NA__", "", "NA", "N/A", "nan", "None"):
        return "NA"
    if head == "Flexion":
        return _FLEX.get(s.lower(), s)
    if head in ("Spread", "Thumb Contact"):
        try:
            return str(int(float(s)))
        except ValueError:
            return s
    return s


def clip_vector(npy_path: Path, xy_only=False):
    """Low-motion static hand frame for one LSC clip -> 63-d vector."""
    arr = np.load(npy_path)
    hand = arr[:, RIGHT_HAND, :].astype(np.float64)
    ok = np.isfinite(hand).all(axis=(1, 2))
    frames = hand[ok]
    if len(frames) == 0:
        return None, 0
    normed = [normalize(f) for f in frames]
    if len(normed) == 1:
        v = normed[0]
    else:
        scores = [float(np.linalg.norm(normed[i] - normed[i + 1]))
                  for i in range(len(normed) - 1)] + [float("inf")]
        v = normed[int(np.argmin(scores))]
    v = v.copy()
    if xy_only:
        v[:, 2] = 0.0
    return v.reshape(-1).astype(np.float32), int(ok.sum())


def stem_label(stem: str) -> str:
    m = re.match(r"^\d+_hs_\d+_\d+_(.+)$", stem)
    if not m:
        raise SystemExit(f"unparseable clip stem: {stem}")
    return m.group(1).removesuffix("_example_sign")


def lsc_inventory_indices(int2str, mapper):
    """Per head, the class ids whose value occurs in the LSC inventory.

    Uses only the target LABEL SPACE (the 43-handshape inventory), never the
    LSC videos, so a constrained argmax stays zero-shot.
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
            return None

    allowed, dropped = {}, {}
    for h, col in ATTRS.items():
        keep, drop = [], []
        for i, s in sorted(int2str[h].items()):
            c = channel_canon(col, canon(h, s))
            (keep if c is not None and c in inv_vals[col] else drop).append((i, s))
        if not keep:
            raise SystemExit(f"--constrain-lsc left no classes for {h!r}")
        allowed[h] = torch.tensor([i for i, _ in keep])
        dropped[h] = [s for _, s in drop]
    return allowed, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=("test", "examples"), default="test")
    ap.add_argument("--audit", type=Path,
                    default=LSC_V2_DIR / "test_labels_audit_v2.csv")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: $ASL2LSC_LSC_LABELS/"
                         "shubert_inference_handshape_<split>_v2.csv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/")
    ap.add_argument("--constrain-lsc", action="store_true",
                    help="restrict each head's argmax to values occurring in "
                         "the LSC handshape inventory (still zero-shot)")
    ap.add_argument("--xy-only", action="store_true",
                    help="diagnostic: zero the depth channel (see docstring)")
    ap.add_argument("--variant", default="",
                    help="read pose553_<variant> instead of pose553 (an "
                         "alternative pose tree for the same clips). The "
                         "manifest still supplies stem ORDER only, so the "
                         "37-clip selection is unaffected.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        RUNS_DIR / args.run_name / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["head_set"], cfg["num_classes"], arch=cfg["arch"],
                        hidden_dim=cfg.get("hidden_dim", 256),
                        dropout=cfg.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    names = cfg["head_names"]
    missing = [h for h in ATTRS if h not in names]
    if missing:
        raise SystemExit(f"checkpoint has no head(s) {missing}; "
                         f"it carries {names}")
    head_idx = {h: names.index(h) for h in ATTRS}

    # Head ids are corpus-relative: PopSign's vocabulary is derived from its
    # own 250-gloss subset, so 11 of the 16 heads have a different class
    # count than Sem-Lex's (data_config.py). Decoding a PopSign checkpoint
    # through Sem-Lex's label_maps.json would silently produce plausible
    # nonsense, so the map follows the checkpoint's own corpus -- the same
    # contract shubert/50_eval/run_eval_popsign.sbatch enforces with
    # --label-maps popsign16_label_maps.json.
    #
    # The bridge itself needs no per-corpus work: canon() above already
    # folds both spellings ("FullyClosed"/"Fully Closed", "NA"/"__NA__",
    # "0"/"0.0") onto one canonical form, and PopSign's values for the five
    # comparable attributes are a subset of Sem-Lex's.
    corpus = cfg.get("corpus", "semlex")   # pre-PopSign ckpts carry no key
    if corpus == "popsign":
        # Follow the checkpoint's label build: new runs record
        # na_policy="class"; legacy checkpoints lack the key and decode
        # against the NA-masked vocab.
        vocab_path = (POPSIGN_VOCAB16_NACLASS
                      if cfg.get("na_policy") == "class" else POPSIGN_VOCAB16)
        maps = json.loads(vocab_path.read_text())["vocab"]
    elif corpus == "semlex":
        maps = json.loads(LABEL_MAPS_JSON.read_text())["maps"]
    else:
        raise SystemExit(f"checkpoint declares unknown corpus {corpus!r}; "
                         f"no label map is wired for it")

    # Hard-fail on a map/checkpoint mismatch rather than decoding through the
    # wrong table: a wrong map yields in-range ids and therefore plausible,
    # silently wrong attributes.
    for h in ATTRS:
        n_ckpt = cfg["num_classes"][names.index(h)]
        if len(maps[h]) != n_ckpt:
            raise SystemExit(
                f"label-map mismatch for {h!r}: checkpoint has {n_ckpt} "
                f"classes, {corpus} map has {len(maps[h])}. Refusing to "
                f"decode (wrong corpus map?).")

    int2str = {h: {v: k for k, v in maps[h].items()} for h in ATTRS}

    # --- inputs, in manifest order so row-order labels stay aligned
    manifest = args.manifest or (
        LSC_V2_DIR / f"shubert_inference_handshape_{args.split}_v2.csv")
    stems = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            face = line.split("\t")[0]
            stems.append(Path(face).name[: -len("_face.npy")])

    truth_rows = list(csv.DictReader(args.audit.open()))
    truth_by_label = {stem_label(r["clip_stem"]): r for r in truth_rows}
    if [stem_label(s) for s in stems] != [
            stem_label(r["clip_stem"]) for r in truth_rows]:
        raise SystemExit("manifest label order differs from the audit CSV; "
                         "row alignment cannot be trusted")

    X, kept_stems, n_hand, no_hand = [], [], [], []
    pose_dir = (POSE553_DIR.with_name(f"pose553_{args.variant}")
                if args.variant else POSE553_DIR)
    for stem in stems:
        npy = pose_dir / args.split / f"{stem}.npy"
        if not npy.exists():
            raise SystemExit(f"missing converted pose {npy}")
        vec, nh = clip_vector(npy, xy_only=args.xy_only)
        if vec is None:
            # No usable frame at all. On the shipped poses this cannot
            # happen (detection is 98.7%) and it stays a hard error; under a
            # crop variant it CAN, because the treatment's whole point is to
            # make the hand leave the frame. Scoring it as a forced miss is
            # the only honest option: dropping the clip would shrink the
            # denominator and inflate every rate on exactly the arms the
            # crop hurt most.
            if not args.variant:
                raise SystemExit(f"no detected hand in {stem}")
            no_hand.append(stem)
            continue
        X.append(vec)
        kept_stems.append(stem)
        n_hand.append(nh)
    if no_hand:
        print(f"NO USABLE HAND FRAME in {len(no_hand)} clip(s): "
              f"{', '.join(no_hand)} — scored as forced misses "
              f"(denominator stays {len(stems)})")

    mapper = LSCHandshapeMapper()
    allowed = None
    if args.constrain_lsc:
        allowed, dropped = lsc_inventory_indices(int2str, mapper)
        for h in ATTRS:
            print(f"constrain-lsc {h}: keeping {len(allowed[h])}/"
                  f"{len(int2str[h])} classes, dropping {dropped[h]}")

    with torch.no_grad():
        logits = model(torch.from_numpy(np.stack(X)).to(device))

    preds = {}
    for j, stem in enumerate(kept_stems):
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
    # Clips the crop left with no detectable hand: every attribute wrong,
    # no handshape, counted in n. They are NOT "unmappable" — that code
    # means the mapper rejected a prediction, and here there is no
    # prediction to reject.
    for stem in no_hand:
        t = truth_by_label[stem_label(stem)]
        rows.append({"clip_stem": stem,
                     "gold_handshape": t["sheet_handshape_name"],
                     "pred_handshape": "", "snap_distance": "",
                     "strict_ok": False, "tie_any_ok": False,
                     "unmappable": "", "no_hand_detected": True,
                     **{f"pred_{ATTRS[h]}": "" for h in ATTRS},
                     **{f"ok_{ATTRS[h]}": False for h in ATTRS}})
    for stem in kept_stems:
        t, p = truth_by_label[stem_label(stem)], preds[stem]
        gold = t["sheet_handshape_name"]
        # Spread is not contrastive for the joint 5B handshapes, so it is not
        # scored there -- any prediction counts. See SPREAD_FREE_HANDSHAPES.
        spread_free = gold in SPREAD_FREE_HANDSHAPES
        hit = {h: (True if (h == "Spread" and spread_free)
                   else canon(h, p[h]) == canon(h, t[col]))
               for h, col in ATTRS.items()}
        for h in ATTRS:
            per_attr[h] += hit[h]
        try:
            m = mapper.map(
                selected_fingers=canon("Selected Fingers", p["Selected Fingers"]),
                flexion=canon("Flexion", p["Flexion"]),
                spread=canon("Spread", p["Spread"]),
                thumb_position=canon("Thumb Position", p["Thumb Position"]),
                thumb_contact=canon("Thumb Contact", p["Thumb Contact"]))
        except ValueError as exc:
            unmappable += 1
            rows.append({"clip_stem": stem, "gold_handshape": gold,
                         "pred_handshape": "", "snap_distance": "",
                         "strict_ok": False, "tie_any_ok": False,
                         "unmappable": str(exc)[:80],
                         **{f"pred_{ATTRS[h]}": p[h] for h in ATTRS},
                         **{f"ok_{ATTRS[h]}": hit[h] for h in ATTRS}})
            continue
        cand = set(m.handshape_names)
        for _feats, nms in m.alternatives:
            cand.update(nms)
        s_ok, t_ok = (m.handshape_name == gold), (gold in cand)
        if spread_free and not s_ok:
            # Same free pass on identity: re-snap with the other spread values.
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
                for _feats, nms in m_alt.alternatives:
                    alt_cand.update(nms)
                if m_alt.handshape_name == gold:
                    m, cand, s_ok, t_ok = m_alt, alt_cand, True, True
                    break
                if gold in alt_cand:
                    cand, t_ok = cand | alt_cand, True
        strict += s_ok
        tie_any += t_ok
        exact += m.is_exact
        dist_sum += m.distance
        rows.append({"clip_stem": stem, "gold_handshape": gold,
                     "pred_handshape": m.handshape_name,
                     "pred_candidates": "|".join(sorted(cand)),
                     "snap_distance": m.distance, "is_tie": m.is_tie,
                     "was_exact": m.is_exact, "strict_ok": s_ok,
                     "tie_any_ok": t_ok, "unmappable": "",
                     **{f"gold_{c}": t[c] for c in ATTRS.values()},
                     **{f"pred_{ATTRS[h]}": p[h] for h in ATTRS},
                     **{f"ok_{ATTRS[h]}": hit[h] for h in ATTRS}})

    # Denominator is every manifest clip, including the no-hand forced
    # misses — see the loop above.
    n = len(kept_stems) + len(no_hand)
    attr_r1 = {h: round(per_attr[h] / n, 4) for h in ATTRS}
    macro = round(float(np.mean(list(attr_r1.values()))), 4)
    n_mapped = n - unmappable - len(no_hand)
    distinct = len({r["pred_handshape"] for r in rows if r["pred_handshape"]})

    out = {
        "model": "baseline-MLP",
        "run_name": args.run_name,
        "checkpoint": str(ckpt_path),
        "head_set": cfg["head_set"],
        "arch": cfg["arch"],
        "split": args.split,
        "constrain_lsc": args.constrain_lsc,
        "xy_only": args.xy_only,
        "variant": args.variant or "native",
        "pose_dir": str(pose_dir),
        "n_clips": n,
        "n_no_hand_detected": len(no_hand),
        "attribute_recall_at_1": attr_r1,
        "macro_attribute_recall": macro,
        "handshape_strict": round(strict / n, 4),
        "handshape_tie_any": round(tie_any / n, 4),
        # Denominators follow eval_shubert_lsc.py exactly, or the columns are
        # not comparable: already-valid is over ALL n (an unmappable
        # prediction is not a valid LSC configuration, it is a wrong one),
        # while mean snap distance averages only over the mappable clips
        # (the rest have no defined distance).
        "pred_already_valid_lsc": round(exact / n, 4) if n else 0.0,
        "mean_snap_distance": round(dist_sum / n_mapped, 4) if n_mapped else None,
        "unmappable": unmappable,
        "distinct_handshape_predictions": distinct,
        "mean_hand_frames": float(np.mean(n_hand)) if n_hand else 0.0,
    }

    out_dir = args.out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.run_name + ("_cons" if args.constrain_lsc else "") + \
        ("_xyonly" if args.xy_only else "") + \
        (f"_{args.variant}" if args.variant else "")
    (out_dir / f"lsc_{args.split}_{tag}.json").write_text(
        json.dumps(out, indent=2))
    with (out_dir / f"lsc_{args.split}_{tag}_per_clip.csv").open(
            "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== LSC {args.split}  ({args.run_name}"
          f"{', constrained' if args.constrain_lsc else ''}"
          f"{', xy-only' if args.xy_only else ''}) ===")
    for h in ATTRS:
        print(f"  {h:<18} {attr_r1[h]:.3f}")
    print(f"  {'macro':<18} {macro:.3f}")
    print(f"  handshape strict   {out['handshape_strict']:.3f}   "
          f"tie-any {out['handshape_tie_any']:.3f}")
    print(f"  distinct handshape predictions: {distinct} / {n} clips")


if __name__ == "__main__":
    main()
