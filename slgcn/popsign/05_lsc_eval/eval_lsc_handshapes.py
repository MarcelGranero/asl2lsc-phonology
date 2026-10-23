"""Zero-shot cross-lingual eval: a PopSign phase-2 model on LSC handshapes.

Pipeline (no training, no LSC data ever seen by the model):

  LSC .pose (586)  ─ map_lsc_pose_to_wholebody133 ─→ (T,133,3)
                   ─ upstream selected_joints['27'] + 150-frame crop/pad ─→ (3,150,27,1)
                   ─ MultiheadModel ─→ 6 ASL-LEX feature predictions
                   ─ predictions.csv  (raw attribute streams; STOPS HERE)

PREDICTIONS ONLY -- THIS SCRIPT DOES NOT SCORE. It writes the
five raw predicted attribute streams and stops. Handshape identity,
attribute recall and every accuracy are produced by common/lsc_eval, which
is THE scoring path for all four model families:

    cd common/lsc_eval
    python3 convert_legacy.py <out_dir>/predictions.csv -o predictions/<arm>.tsv
    python3 evaluate.py predictions/<arm>.tsv

This script emits predictions only; it does no ASL-LEX->LSC mapping of its
own. asllex_to_lsc.py supplies the LSC inventory behind the --constrain-lsc
mask.

The frame packing REUSES upstream's own logic rather than reimplementing it:
`sign_gendata.py` crops >150 frames and cyclically pads <150, and any deviation
here would mean the model sees a differently-shaped input than it trained on.

WHAT THIS MEASURES: whether ASL handshape phonology learned from PopSign
transfers to Catalan Sign Language handshapes -- the two label spaces are
bridged downstream by a FROZEN linguistic lookup, never by anything fitted
to LSC.

WHAT IT DOES NOT MEASURE: LSC handshape recognition as such. The eval set is
~1 example per class, so per-class accuracy is a single Bernoulli trial and
the aggregate has a very wide confidence interval. Report it as a transfer
probe, never as a benchmark number.

SPLIT: emits the v2 split by default -- 37 clips / 37 classes, after two
faulty recordings were excluded. Pass --v1-split for the original 39
clips. The spread rule for the joint 5B handshapes is applied by the
scorer, not here.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "04_phase2_multihead"))

from asllex_to_lsc import CONFIGS, FLEXION_BRIDGE  # noqa: E402
from common import git_sha, sha256_file, utcnow, write_json, write_run_card  # noqa: E402
from data_config import FEATURE_VOCAB_JSON, RESULTS_DIR, SL_GCN_DIR, VOCAB_JSON  # noqa: E402
from lsc_domain_adapt import reframe_to_popsign, resample_fps  # noqa: E402
from map_lsc_pose_to_wholebody133 import convert, read_lsc_pose  # noqa: E402
from select_hold_frames import select_hold  # noqa: E402

sys.path.insert(0, str(SL_GCN_DIR / "data_gen"))
from sign_gendata import selected_joints  # noqa: E402

SEL27 = selected_joints["27"]
MAX_FRAME = 150            # upstream sign_gendata constant

# SEM-LEX 553 layout offsets (face 0:478, pose 478:511, LH 511:532,
# RH 532:553), the layout of --pose553-dir trees.
P553_POSE_OFF, P553_LH_OFF, P553_RH_OFF = 478, 511, 532
# The LSC handshape benchmark is a separate release; ASL2LSC_LSC_BENCH is its
# root (manifests at the top level, raw .pose files under handshape_poses/
# and handshape_example_poses/).
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


LSC_MANIFEST = _bench_dir("ASL2LSC_LSC_BENCH") / (
    "handshape_manifest.csv")
# The examples split: one example sign recorded right after each
# configuration, 1:1 with the configurations by recording order; label_raw
# is "<config>_example_sign" and base/variant are inherited from the paired
# configuration (the signs' lexical identities are not annotated).
LSC_EXAMPLES_MANIFEST = LSC_MANIFEST.with_name(
    "handshape_examples_manifest.csv")

# --- v2 labels ------------------------------------------------------------
# Two clips were excluded as faulty recordings, leaving 37 clips / 37 classes
# (each label once). Keyed by <start_ms>_<end_ms>, which is what identifies a
# manifest row here -- the label alone is ambiguous, since both excluded clips
# shared their label with a surviving clip.
V2_EXCLUDED_SPANS = {
    ("1807350", "1808563"),   # 06_hs_..._middle-flat_open
    ("2177280", "2178425"),   # 06_hs_..._3-extended
}

LSC_POSE_DIR = _bench_dir("ASL2LSC_LSC_BENCH") / "handshape_poses"
LSC_EXAMPLE_POSE_DIR = LSC_POSE_DIR.with_name("handshape_example_poses")

# ASL-LEX value order per feature must match the model's head order, which is
# feature_vocab.json's `features` + per-feature `classes`.
PREDICT_ORDER = ["handshape", "selected_fingers", "flexion", "spread",
                 "thumb_position", "thumb_contact"]


# CONFIGS column carrying each predicted channel's ASL-LEX value.
_INV_COLUMNS = {"selected_fingers": "asllex_SelectedFingers",
                "flexion": "asllex_Flexion",
                "spread": "asllex_Spread",
                "thumb_position": "asllex_ThumbPosition_derived",
                "thumb_contact": "asllex_ThumbContact"}


def lsc_inventory_indices(idx_to_value):
    """Per head: which class ids denote a value occurring in the LSC inventory.

    Same PER-CHANNEL MEMBERSHIP rule as the SHuBERT / MLP / SL-GCN-Sem-Lex
    harnesses, so "+c" is one treatment across every model family: a value
    survives iff it appears in that channel's column of the 43-configuration
    table, independently of the other four channels. Reads only the target
    LABEL SPACE, never the LSC videos or labels, so the constrained argmax
    stays zero-shot.

    Flexion NA survives here but not in the other leaves, and that is a
    TABLE divergence rather than a disagreement about phonology: this
    table codes the three thumb-only configurations (6-extended,
    6-curved-open, 6-closed) with asllex_Flexion='NA' as a wildcard and
    carries the real contrast in its separate lsc_flexion column, while
    the shared inventory codes their flexion outright as FullyOpen /
    Curved-Bent / FullyClosed. Consequence: a thumb-only prediction here
    resolves three-ways AMBIGUOUS rather than to one configuration.
    """
    inv_vals = {c: set() for c in _INV_COLUMNS}
    for cfg in CONFIGS:
        for c, col in _INV_COLUMNS.items():
            inv_vals[c].add(str(cfg[col]))
    # The table's flexion column is the ASL-LEX side, so it never spells
    # 'Bent' -- Bent bridges to curved (FLEXION_BRIDGE)
    # and must survive the mask. Membership is therefore tested on the
    # bridged value, and 't' is normalised the way map_prediction does.
    inv_flex = {v for v in inv_vals["flexion"]}
    inv_flex |= {k for k, v in FLEXION_BRIDGE.items()
                 if v in {FLEXION_BRIDGE.get(x) for x in inv_vals["flexion"]}}

    def channel_value(c, s):
        s = str(s)
        if c == "selected_fingers" and s == "t":
            return "thumb"
        return s

    allowed, dropped = {}, {}
    for c in _INV_COLUMNS:
        keep, drop = [], []
        valid = inv_flex if c == "flexion" else inv_vals[c]
        for i, s in enumerate(idx_to_value[c]):
            (keep if channel_value(c, s) in valid else drop).append((i, s))
        if not keep:
            raise SystemExit(f"--constrain-lsc left no valid classes for {c!r}")
        allowed[c] = torch.tensor([i for i, _ in keep])
        dropped[c] = [s for _, s in drop]
    return allowed, dropped



def pose553_to_wholebody(arr):
    """(T,553,3) crop-normalised npy -> (T,133,3) in the training pixel box.

    For the video-crop arms the domain adaptation happened in PIXELS
    (the re-cut video put the signer in PopSign's portrait framing, and
    MediaPipe re-extraction normalised to the CROPPED frame), so the only
    lift left is the training one: x/y x512 both axes, exactly what
    map_543_to_wholebody133 did to the PopSign parquet — no scale_mode logic
    applies. Presence comes from the NaN convention (channel 3 of the npy is
    MediaPipe z, not confidence). Same joint choice as every converter in the
    family; only the offsets differ.
    """
    arr = np.asarray(arr, dtype=np.float32)
    assert arr.ndim == 3 and arr.shape[1] == 553, arr.shape
    body = [(w, P553_POSE_OFF + m) for w, m in
            [(0, 0), (1, 2), (2, 5), (3, 7), (4, 8), (5, 11), (6, 12),
             (7, 13), (8, 14), (9, 15), (10, 16), (11, 23), (12, 24),
             (13, 25), (14, 26), (15, 27), (16, 28)]]
    full = body + [(91 + i, P553_LH_OFF + i) for i in range(21)] \
                + [(112 + i, P553_RH_OFF + i) for i in range(21)]
    wb_idx = np.array([w for w, _ in full])
    src_idx = np.array([s for _, s in full])

    src = arr[:, src_idx, :]                      # (T, 59, 3)
    missing = np.isnan(src[:, :, :2]).any(axis=2)
    out = np.zeros((arr.shape[0], 133, 3), dtype=np.float32)
    xy = np.nan_to_num(src[:, :, :2], nan=0.0) * 512.0
    xy[missing] = 0.0
    out[:, wb_idx, :2] = xy
    out[:, wb_idx, 2] = (~missing).astype(np.float32)
    return out


def pack_clip(wb, normalization=True):
    """(T,133,3) -> (1,3,150,27,1): upstream's crop/cyclic-pad AND the
    feeder's `normalization` step.

    The normalisation is NOT optional decoration. `feeders/feeder.py`
    __getitem__ applies it to every sample when the config sets
    `normalization: True` (both PopSign configs do):

        data[0] -= data[0,:,0,0].mean(axis=0)   # subtract joint-0 (nose) x
        data[1] -= data[1,:,0,0].mean(axis=0)   # subtract joint-0 (nose) y

    i.e. a per-clip translation putting the nose at the origin. Omitting it
    feeds the model absolute coordinates it never saw in training (on PopSign
    val that drops feat_avg from 0.95 to ~0.30 for the same checkpoint).

    Because the model sees nose-centred input it is already invariant to
    absolute POSITION — which is why the hand/person/fit modes matter only
    for SCALE.
    """
    skel = wb[:, SEL27, :]                       # (T, 27, 3)
    fp = np.zeros((MAX_FRAME, 27, 3), dtype=np.float32)
    T = skel.shape[0]
    if T < MAX_FRAME:
        fp[:T] = skel
        rest = MAX_FRAME - T
        num = int(np.ceil(rest / T))
        fp[T:] = np.concatenate([skel] * num, axis=0)[:rest]
    else:
        fp[:] = skel[:MAX_FRAME]                 # upstream crops the tail
    # (T,V,C) -> (C,T,V,M)
    out = torch.from_numpy(fp).permute(2, 0, 1).unsqueeze(-1)   # (3,150,27,1)
    if normalization:
        # verbatim port of feeder.py lines 96-104 (non-vector branch)
        out[0] = out[0] - out[0, :, 0, 0].mean(axis=0)
        out[1] = out[1] - out[1, :, 0, 0].mean(axis=0)
    return out.unsqueeze(0)


def load_model(ckpt_path, feature_sizes, device):
    from model_multihead import MultiheadModel
    import yaml
    cfg = yaml.safe_load(
        (REPO / "02_train/config/train_joint_popsign.yaml").read_text())
    model_args = dict(cfg["model_args"])
    state = torch.load(ckpt_path, map_location="cpu")
    # A pfr checkpoint has no gloss head; an isr_pfr one does. Infer rather
    # than require the caller to know which regime produced the file.
    with_gloss = any(k.startswith("fc.") for k in state)
    model = MultiheadModel(feature_sizes=feature_sizes, with_gloss=with_gloss,
                           **model_args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  WARNING {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return model.to(device).eval(), with_gloss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True,
                    help="a phase-2 checkpoint (pfr or isr_pfr)")
    ap.add_argument("--feature-vocab", type=Path, default=None,
                    help="feature vocabulary the checkpoint was trained "
                         "against (default: the 6-head "
                         "results/feature_vocab.json). Pass "
                         "results/feature_vocab16_naclass.json to score a "
                         "16-head pfr16 checkpoint: the model is built with "
                         "all 16 heads and the 6 handshape channels the LSC "
                         "bridge needs are selected by name.")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--channels", default="presence",
                    choices=["presence", "confidence"])
    ap.add_argument("--scale-mode", default="hand",
                    choices=["hand", "person", "fit", "none"],
                    help="hand: hand-bbox anchored (default; the "
                         "reference that matters for handshape). person: "
                         "shoulder-anchored (overshoots hands 4.4x). fit: "
                         "frame-relative (collapses LSC's wide shot). none: raw.")
    ap.add_argument("--hold-frames", action="store_true",
                    help="eval-time preprocessing: keep only the held-"
                         "configuration segment of each clip (longest slow, "
                         "hand-visible run), ported from the SL-GCN Sem-Lex leaf's "
                         "lsc_handshapes/select_hold_frames.py. Applied "
                         "BEFORE convert(), so hand-anchored scaling is "
                         "estimated from the hold, not the transitions.")
    ap.add_argument("--speed-pct", type=float, default=50.0,
                    help="per-clip speed percentile defining 'slow' (default 50)")
    ap.add_argument("--fps30", action="store_true",
                    help="resample LSC's 50 fps poses to PopSign's ~30 fps "
                         "before packing. The model reads a fixed 150-FRAME "
                         "window, so an unresampled LSC sign spans ~1.67x the "
                         "frames it was trained on. Ported from "
                         "the SL-GCN Sem-Lex leaf's pose_to_553.py --target-fps.")
    ap.add_argument("--reframe", action="store_true",
                    help="similarity-transform the signer to PopSign's "
                         "shoulder geometry (width AND height) in normalised "
                         "frame coords.")
    ap.add_argument("--split", default="test", choices=["test", "examples"],
                    help="test: the isolated configuration demonstrations "
                         "(default). examples: the example sign recorded "
                         "right after each configuration (coarticulated); "
                         "the v2 exclusions carry over through the 1:1 "
                         "recording-order pairing, and scoring is against "
                         "the paired configuration's label.")
    ap.add_argument("--constrain-lsc", action="store_true",
                    help="restrict each head's argmax to values occurring in "
                         "the LSC handshape inventory (still zero-shot; "
                         "appends _cons to the run name)")
    ap.add_argument("--v1-split", action="store_true",
                    help="score the ORIGINAL 39-clip split with spread scored "
                         "for every handshape. Default is the v2 split: 37 "
                         "clips, spread not scored for the joint 5B "
                         "handshapes.")
    ap.add_argument("--clip-outside", action="store_true",
                    help="with --reframe, also drop landmarks pushed outside "
                         "the frame (simulates a truncation defect, not just "
                         "framing; off by default)")
    ap.add_argument("--pose553-dir", type=Path, default=None,
                    help="read (T,553,3) crop-re-extracted npys from this "
                         "directory instead of the raw .pose files (the "
                         "video-crop ablation trees, e.g. "
                         ".../lsc_handshapes/pose553_vidcropphone/test, or "
                         "its examples/ sibling -- pass the tree matching "
                         "--split). The crop already performed the domain "
                         "adaptation in pixels, so the pose-space treatment "
                         "flags are refused in this mode.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    # A relative --feature-vocab names a file of this leaf (results/...), so
    # it resolves the same from any working directory (the wrappers run from
    # inside the SL-GCN clone).
    if (args.feature_vocab is not None and not args.feature_vocab.is_absolute()
            and not args.feature_vocab.exists()):
        args.feature_vocab = REPO / args.feature_vocab

    if args.pose553_dir:
        bad = [f for f, v in [("--reframe", args.reframe),
                              ("--fps30", args.fps30),
                              ("--clip-outside", args.clip_outside),
                              ("--hold-frames", args.hold_frames)] if v]
        if bad:
            raise SystemExit(
                f"--pose553-dir is incompatible with {', '.join(bad)}: the "
                "video crop already applied the framing, and its fps rungs "
                "come from frame-strided trees")
        # Works for both splits: every pose553_<variant> tree carries an
        # examples/ sibling (stems with the manifest's _example_sign
        # suffix). Pass the tree for the split you are scoring
        # (.../pose553_vidcrop/examples with --split examples); an absent
        # tree surfaces per clip as "no pose553 npy".
        print(f"pose553 mode: reading {args.pose553_dir}; --channels/"
              "--scale-mode are ignored (presence + x512 fit lift)")

    # Two vocabulary schemas reach here. The 6-head file
    # (results/feature_vocab.json) nests everything under `per_feature` and
    # its heads ARE the six the bridge needs, in PREDICT_ORDER. The 16-head
    # files (feature_vocab16*.json, --feature-vocab) are flat -- parallel
    # `features`/`feature_sizes` lists plus a `vocab` dict -- and carry the
    # six handshape heads at scattered indices among ten lexical ones.
    #
    # Supporting the second is what lets a 16-head checkpoint be scored on
    # LSC at all. The model is built with ALL its heads
    # so the state dict loads exactly, and the six are selected BY NAME
    # afterwards. Selecting by name matters: in the 16-head checkpoint
    # Handshape is head 15, not head 0, and load_state_dict(strict=False)
    # would have accepted a wrong-shaped build silently.
    fv = json.loads((args.feature_vocab or FEATURE_VOCAB_JSON).read_text())
    if "per_feature" in fv:
        all_names = fv["features"]
        all_sizes = [fv["per_feature"][f]["n_classes"] for f in all_names]
        classes = {f: fv["per_feature"][f]["classes"] for f in all_names}
    else:
        all_names = fv["features"]
        all_sizes = list(fv["feature_sizes"])
        # The 16-head files store each head's vocabulary as {value: index},
        # where the 6-head file stores an index-ordered list. Invert to the
        # list form the prediction loop expects. For all six bridge
        # channels the inverted order is identical to the 6-head file's, so
        # a class id means the same thing under either vocabulary.
        classes = {}
        for f in all_names:
            v = fv["vocab"][f]
            if isinstance(v, dict):
                inv = [None] * len(v)
                for value, i in v.items():
                    inv[i] = value
                classes[f] = inv
            else:
                classes[f] = list(v)

    # The bridge's six, named as this vocabulary spells them: the 6-head
    # file uses snake_case ("selected_fingers"), the 16-head ones Title Case
    # ("Selected Fingers"). Map both onto PREDICT_ORDER's spelling.
    def _canon(name):
        return name.lower().replace(" ", "_")

    by_canon = {_canon(f): f for f in all_names}
    missing_heads = [c for c in PREDICT_ORDER if c not in by_canon]
    if missing_heads:
        raise SystemExit(
            f"{args.feature_vocab or FEATURE_VOCAB_JSON} has no head(s) "
            f"{missing_heads}; the bridge needs all six of {PREDICT_ORDER}")
    # head index in the MODEL, per bridge channel
    head_index = {c: all_names.index(by_canon[c]) for c in PREDICT_ORDER}
    feature_names = PREDICT_ORDER
    feature_sizes = all_sizes
    idx_to_value = {c: classes[by_canon[c]] for c in PREDICT_ORDER}
    print("head map: " + ", ".join(f"{c}=#{head_index[c]}"
                                   f"({all_sizes[head_index[c]]})"
                                   for c in PREDICT_ORDER))

    allowed = None
    if args.constrain_lsc:
        allowed, dropped_cls = lsc_inventory_indices(idx_to_value)
        for c in allowed:
            print(f"constrain-lsc {c}: keeping {len(allowed[c])}/"
                  f"{len(idx_to_value[c])} classes, dropping {dropped_cls[c]}")

    manifest = (LSC_EXAMPLES_MANIFEST if args.split == "examples"
                else LSC_MANIFEST)
    rows = list(csv.DictReader(manifest.open()))
    if args.split == "examples":
        # V2_EXCLUDED_SPANS is keyed by CONFIGURATION spans, and the label
        # alone is ambiguous (both excluded configs shared theirs with a
        # surviving clip) -- so map the exclusions onto the examples rows
        # through the 1:1 recording-order pairing, verified here.
        cfg_rows = list(csv.DictReader(LSC_MANIFEST.open()))
        if len(cfg_rows) != len(rows):
            raise SystemExit(
                f"examples manifest has {len(rows)} rows vs "
                f"{len(cfg_rows)} configurations -- pairing broken")
        mismatched = [i for i, (c, e) in enumerate(zip(cfg_rows, rows))
                      if e["label_raw"] != c["label_raw"] + "_example_sign"]
        if mismatched:
            raise SystemExit("examples/configuration label pairing broken "
                             f"at manifest rows {mismatched[:5]}")
        excluded_idx = {i for i, c in enumerate(cfg_rows)
                        if (c["start_ms"], c["end_ms"]) in V2_EXCLUDED_SPANS}
    if not args.v1_split:
        before = len(rows)
        if args.split == "examples":
            rows = [r for i, r in enumerate(rows) if i not in excluded_idx]
        else:
            rows = [r for r in rows
                    if (r["start_ms"], r["end_ms"]) not in V2_EXCLUDED_SPANS]
        dropped = before - len(rows)
        if dropped != len(V2_EXCLUDED_SPANS):
            raise SystemExit(
                f"v2 exclusion dropped {dropped} clips, expected "
                f"{len(V2_EXCLUDED_SPANS)} — manifest spans may have changed")
        print(f"v2 labels: dropped {dropped} faulty clips")
    print(f"LSC manifest: {len(rows)} clips, "
          f"{len({r['label_raw'] for r in rows})} classes")

    device = torch.device(args.device)
    model, with_gloss = load_model(args.weights, feature_sizes, device)
    print(f"loaded {args.weights.name} (gloss head: {with_gloss})")

    results, n_missing_pose = [], 0
    for r in rows:
        stem = (f"{r['session']}_hs_{r['start_ms']}_{r['end_ms']}_"
                f"{r['label_raw']}")
        # Gold is the HANDSHAPE label: for the examples split the manifest's
        # label_raw carries the _example_sign suffix (and so do the pose
        # stems), but the truth is the paired configuration's label.
        true_label = r["label_raw"]
        if args.split == "examples":
            true_label = true_label[:-len("_example_sign")]
        hold_span = None
        if args.pose553_dir:
            npy_path = args.pose553_dir / f"{stem}.npy"
            if not npy_path.exists():
                # Same '+' -> 'p' filename sanitisation as the .pose branch.
                alt = args.pose553_dir / f"{stem.replace('+', 'p')}.npy"
                if alt.exists():
                    npy_path = alt
            if not npy_path.exists():
                n_missing_pose += 1
                print(f"  WARNING no pose553 npy for {stem}")
                continue
            wb = pose553_to_wholebody(np.load(npy_path))
        else:
            base_dir = (LSC_EXAMPLE_POSE_DIR if args.split == "examples"
                        else LSC_POSE_DIR)
            pose_dir = base_dir / f"session_{r['session']}"
            pose_path = pose_dir / f"{stem}.pose"
            if not pose_path.exists():
                # Stage-08 sanitises '+' to 'p' in filenames ('index+thumb' ->
                # 'indexpthumb'), so 4 of the 39 clips do not match the
                # manifest label verbatim. Without this the eval silently
                # drops them.
                alt = pose_dir / f"{stem.replace('+', 'p')}.pose"
                if alt.exists():
                    pose_path = alt
            if not pose_path.exists():
                n_missing_pose += 1
                print(f"  WARNING no .pose for {stem}")
                continue
            data, conf = read_lsc_pose(pose_path)
            # Order matters: resample first so the hold selector's speed
            # percentile is computed at the TARGET frame rate (speed is per
            # frame, so it changes with fps); reframe before hold selection so
            # the selector's shoulder-normalised speeds use the final
            # geometry; both before convert(), so hand-anchored scaling sees
            # the adapted pose.
            if args.fps30:
                data, conf = resample_fps(data, conf)
            if args.reframe:
                data, conf = reframe_to_popsign(
                    data, conf, clip_outside=args.clip_outside)
            if args.hold_frames:
                n_orig = len(data)
                data, conf, hold_span = select_hold(data, conf,
                                                    speed_pct=args.speed_pct)
                print(f"  {stem}: hold {hold_span[0]}..{hold_span[1]} "
                      f"({len(data)}/{n_orig} frames)")
            wb = convert(data, conf, channels=args.channels,
                         scale_mode=args.scale_mode)
        x = pack_clip(wb).to(device)

        with torch.no_grad():
            _, feat_logits = model(x, keep_prob=1.0)
        pred = {}
        for f in feature_names:
            # Index by the head's position IN THE MODEL, not by zip order:
            # a 16-head checkpoint carries the six bridge channels scattered
            # among ten lexical heads (see the head map printed at startup).
            l = feat_logits[head_index[f]]
            # `handshape` is never mapped through the bridge, so the
            # constraint does not apply to it.
            if allowed is None or f not in allowed:
                pred[f] = idx_to_value[f][int(l.argmax(dim=1))]
            else:
                keep = allowed[f].to(l.device)
                pred[f] = idx_to_value[f][int(keep[l[0][keep].argmax()])]

        results.append({
            "clip": stem, "true_label": true_label,
            "true_base": r["handshape_base"],
            "pred_handshape_head": pred["handshape"],
            **{f"pred_{k}": v for k, v in pred.items() if k != "handshape"},
            "hold_span": (f"{hold_span[0]}-{hold_span[1]}" if hold_span
                          else ""),
        })

    if n_missing_pose:
        print(f"WARNING {n_missing_pose} manifest rows have no .pose file")

    # SCORING LIVES IN common/lsc_eval, NOT HERE.
    # This script's job ends at the five raw predicted attribute streams.
    # Everything downstream -- canonicalisation, the snap onto the LSC
    # inventory, handshape identity, attribute recall -- is done by
    # common/lsc_eval/evaluate.py, identically for every model family, after
    # convert_legacy.py lifts predictions.csv into the standard TSV.
    metrics = {
        "n_clips": len(results),
        "n_missing_pose": n_missing_pose,
    }
    metrics["constrain_lsc"] = bool(args.constrain_lsc)

    run_name = args.run_name or f"lsc_handshapes_{args.weights.stem}"
    if args.constrain_lsc:
        run_name += "_cons"
    out_dir = RESULTS_DIR / "lsc_eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    write_json(out_dir / "metrics.json", metrics)
    write_run_card(out_dir, {
        "task": "zero-shot PopSign->LSC handshape transfer",
        "git_sha": git_sha(),
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "feature_vocab": str(args.feature_vocab or FEATURE_VOCAB_JSON),
        "feature_vocab_sha": fv.get("git_sha"),
        "n_model_heads": len(feature_sizes),
        "bridge_head_index": head_index,
        "split": args.split,
        "lsc_manifest": str(manifest),
        "mapping": "none in-leaf; scored by common/lsc_eval/evaluate.py",
        "channels": args.channels, "scale_mode": args.scale_mode,
        "hold_frames": bool(args.hold_frames),
        "fps30": bool(args.fps30),
        "reframe": bool(args.reframe),
        "clip_outside": bool(args.clip_outside),
        "pose553_dir": str(args.pose553_dir) if args.pose553_dir else None,
        "speed_pct": args.speed_pct if args.hold_frames else None,
        "metrics": metrics,
        "lsc_labels": "v1 (39 clips)" if args.v1_split else
                      "v2 (37 clips; spread unscored for the joint 5B "
                      "handshapes)",
        "caveat": f"{len(results)} clips / ~1 per class — a transfer probe "
                  "with very wide error bars, NOT an LSC benchmark result.",
        "finished_at": utcnow(),
    })

    print(f"\n=== {run_name} ===")
    print(f"clips {metrics['n_clips']} | predictions only — no scoring here")
    print(f"\nwrote {out_dir}")
    print("score with:\n"
          f"  cd common/lsc_eval && python3 convert_legacy.py "
          f"{out_dir}/predictions.csv -o predictions/<arm>.tsv "
          "&& python3 evaluate.py predictions/<arm>.tsv")


if __name__ == "__main__":
    main()
