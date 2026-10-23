"""LSC benchmark 586-point MediaPipe Holistic -> COCO-WholeBody 133-point.

Third member of the family (see PopSign 543 and SEM-LEX 553). LSC benchmark
ships `.pose` files in the sign-language-processing format rather than raw
npys, and its block order differs from BOTH SLGCN repos — POSE comes FIRST,
not the face:

  LSC (586)    pose 0:33 | face 33:511 (478, incl. iris) | LH 511:532 | RH 532:553 | pose_world 553:586
  SEM-LEX(553) face 0:478 (incl. iris) | pose 478:511 | LH 511:532 | RH 532:553
  PopSign(543) face 0:468 (no iris)    | pose 468:501 | LH 501:522 | RH 522:543

Verified empirically on session_06 (do NOT trust component order alone — the
equivalent assumption was wrong once on SEM-LEX and only overlay QA caught it):
nose[0] y=222 above shoulders[11,12] y≈384-390 above wrist[15] y=710; the two
hand blocks at 511/532 sit at the wrists. `POSE_WORLD_LANDMARKS` (553:586) is
a metric-space duplicate of the pose block and is IGNORED.

Two differences from the SLGCN converters that matter:

1. COORDINATES ARE ALREADY IN PIXELS (1920x1080 per the pose header), not
   normalised. The SLGCN converters multiply by 512 to lift normalised coords
   into the pixel space upstream's feeder augments in. Here we must instead
   RESCALE 1920x1080 -> the training frame, otherwise every coordinate is ~2-4x
   too large and the model sees garbage. See `--target-scale`.

2. There is a REAL confidence channel in [0,1], where MediaPipe npys have none.
   The SLGCN converters synthesise a presence indicator (1.0 detected / 0.0
   NaN). To keep the model's input distribution identical to training we
   BINARISE it by default (`--channels presence`); `--channels confidence`
   keeps the raw value for ablation.
"""
import numpy as np

# LSC .pose block offsets — verified empirically, not read from the header.
POSE_OFFSET = 0
FACE_OFFSET = 33
LEFT_HAND_OFFSET = 511
RIGHT_HAND_OFFSET = 532
POSE_WORLD_OFFSET = 553      # ignored
N_LANDMARKS = 586

# Training-time pixel space. PopSign clips are normalised coords lifted by
# SCALE=512 in map_543_to_wholebody133.py, so the model was trained on a
# 512-wide frame; LSC pixels must land in the same box.
TRAIN_SCALE = 512.0
LSC_WIDTH, LSC_HEIGHT = 1920.0, 1080.0

# (wholebody_idx, lsc586_idx); BlazePose ids + POSE_OFFSET. Identical joint
# choice to the other two converters — only the offset constants differ.
BODY_MAP = [
    (0, POSE_OFFSET + 0),    # nose
    (1, POSE_OFFSET + 2),    # left eye
    (2, POSE_OFFSET + 5),    # right eye
    (3, POSE_OFFSET + 7),    # left ear
    (4, POSE_OFFSET + 8),    # right ear
    (5, POSE_OFFSET + 11),   # left shoulder
    (6, POSE_OFFSET + 12),   # right shoulder
    (7, POSE_OFFSET + 13),   # left elbow
    (8, POSE_OFFSET + 14),   # right elbow
    (9, POSE_OFFSET + 15),   # left wrist
    (10, POSE_OFFSET + 16),  # right wrist
    (11, POSE_OFFSET + 23),  # left hip
    (12, POSE_OFFSET + 24),  # right hip
    (13, POSE_OFFSET + 25),  # left knee
    (14, POSE_OFFSET + 26),  # right knee
    (15, POSE_OFFSET + 27),  # left ankle
    (16, POSE_OFFSET + 28),  # right ankle
]
LEFT_HAND_MAP = [(91 + i, LEFT_HAND_OFFSET + i) for i in range(21)]
RIGHT_HAND_MAP = [(112 + i, RIGHT_HAND_OFFSET + i) for i in range(21)]

FULL_MAP = BODY_MAP + LEFT_HAND_MAP + RIGHT_HAND_MAP
WB_IDX = np.array([w for w, _ in FULL_MAP])
LSC_IDX = np.array([m for _, m in FULL_MAP])


def _hand_normalise(xy, present, target_span=170.0, target_center=(238.0, 388.0)):
    """Anchor on the HAND, because handshape is what the model reads.

    Measured medians (popsign val vs LSC session_06, both in the 512 box):

                  shoulder   hand span   hand/shoulder
        PopSign        284         170            0.60
        LSC (fit)       64         198            3.08

    The ratio differs 5.1x, so the two datasets are NOT related by any single
    similarity transform — no global rescale can match body AND hands at once.
    LSC hands are large relative to the body because stage-08 extracts poses
    on cropped handshape segments; PopSign is a whole-body phone shot.

    Anchoring on shoulders (the obvious choice, and what this function did
    first) scales the hands by 4.4x -> a 875 px hand span against a 170 px
    target, which is as wrong as the un-normalised version, just in the other
    direction. Since every feature head predicts a property of the HAND, the
    hand bounding box is the reference that matters; the body joints are
    along for the ride.
    """
    hand_idx = np.arange(91, 133)
    h = xy[:, hand_idx, :]
    hp = present[:, hand_idx]
    spans, centers = [], []
    for t in range(h.shape[0]):
        p = h[t][hp[t]]
        if len(p) > 3:
            spans.append(float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))))
            centers.append(p.mean(axis=0))
    if not spans:
        return xy  # no hand detected — caller reports it
    scale = target_span / float(np.median(spans))
    center = np.mean(centers, axis=0)
    out = (xy - center) * scale + np.asarray(target_center, dtype=np.float32)
    out[~present] = 0.0
    return out


def _person_normalise(xy, present, target_shoulder=279.0, target_center=(238.0, 388.0)):
    """Anchor the skeleton on the BODY instead of the video frame.

    Frame-relative scaling ('fit') fails across datasets whose framing
    differs: measured, the LSC signer spans 12.7% of frame width while a
    PopSign signer spans 54.5% of the training box — a 4.3x mismatch that
    collapses every LSC clip into a tiny cluster near the frame centre, so
    the model returns one constant prediction for all inputs.

    Here the shoulder width is rescaled to the training-set mean (279 px in
    the 512 box, measured over popsign val) and the shoulder midpoint moved
    to the training-set mean position. Scale and translation are estimated
    ONCE PER CLIP from confident frames, so within-clip motion is preserved.

    Wholebody indices: 5 = left shoulder, 6 = right shoulder.
    """
    ls, rs = xy[:, 5, :], xy[:, 6, :]
    ok = present[:, 5] & present[:, 6]
    if ok.sum() == 0:
        return xy  # no body reference — leave as-is, reported by the caller
    width = np.linalg.norm(ls[ok] - rs[ok], axis=1)
    width = width[width > 1e-3]
    if width.size == 0:
        return xy
    scale = target_shoulder / float(np.median(width))
    center = ((ls[ok] + rs[ok]) / 2.0).mean(axis=0)

    out = (xy - center) * scale + np.asarray(target_center, dtype=np.float32)
    out[~present] = 0.0
    return out


def convert(data, confidence, channels="presence", scale_mode="fit"):
    """(T, 586, 3) + (T, 586) -> (T, 133, 3) float32 in wholebody layout.

    `data` is the raw `.pose` body data for a single person, in LSC PIXELS.
    `scale_mode`:
      'fit'  — divide by the LSC frame size, multiply by TRAIN_SCALE. Keeps
               the aspect ratio distortion the training data already had
               (PopSign scaled x and y by the same 512), so an LSC 16:9 frame
               is squashed exactly the way a PopSign 4:3 frame was.
      'none' — leave pixels untouched (diagnostic only; the model was never
               trained on a 1920-wide box).
    """
    data = np.asarray(data, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    assert data.ndim == 3 and data.shape[1] == N_LANDMARKS, data.shape

    T = data.shape[0]
    src = data[:, LSC_IDX, :]           # (T, 59, 3)
    conf = confidence[:, LSC_IDX]       # (T, 59)

    out = np.zeros((T, 133, 3), dtype=np.float32)
    xy = src[:, :, :2].copy()
    # An undetected landmark is (0,0) with confidence 0 in the .pose format.
    # Zero it explicitly so it matches the SLGCN converters' NaN -> 0 rule.
    missing = conf <= 0
    xy[missing] = 0.0

    if scale_mode == "fit":
        # Frame-relative. Preserves LSC's wide-shot framing, which does NOT
        # match the training data — see _person_normalise. Kept for ablation.
        xy[:, :, 0] *= TRAIN_SCALE / LSC_WIDTH
        xy[:, :, 1] *= TRAIN_SCALE / LSC_HEIGHT
    elif scale_mode in ("hand", "person"):
        # 'hand' (default): match the HAND bounding box to the training set —
        # the reference that matters for handshape. 'person' matches shoulder
        # width instead, which overshoots the hands 4.4x (kept for ablation;
        # see _hand_normalise for the measured ratios).
        xy[:, :, 0] *= TRAIN_SCALE / LSC_WIDTH
        xy[:, :, 1] *= TRAIN_SCALE / LSC_HEIGHT
        # scatter into wholebody slots first so the 5/6 and 91..132 indices
        # used by the normalisers are valid
        tmp = np.zeros((T, 133, 2), dtype=np.float32)
        pres = np.zeros((T, 133), dtype=bool)
        tmp[:, WB_IDX] = xy
        pres[:, WB_IDX] = ~missing
        fn = _hand_normalise if scale_mode == "hand" else _person_normalise
        tmp = fn(tmp, pres)
        out[:, :, :2] = tmp
        if channels == "presence":
            out[:, WB_IDX, 2] = (~missing).astype(np.float32)
        else:
            out[:, WB_IDX, 2] = np.nan_to_num(conf, nan=0.0)
        return out
    elif scale_mode != "none":
        raise ValueError(f"unknown scale_mode: {scale_mode}")

    out[:, WB_IDX, :2] = np.nan_to_num(xy, nan=0.0)

    if channels == "presence":
        out[:, WB_IDX, 2] = (~missing).astype(np.float32)
    elif channels == "confidence":
        out[:, WB_IDX, 2] = np.nan_to_num(conf, nan=0.0)
    else:
        raise ValueError(f"unknown channels mode: {channels}")
    return out


def read_lsc_pose(path):
    """`.pose` file -> (data (T,586,3), confidence (T,586)) for person 0."""
    from pose_format import Pose
    with open(path, "rb") as f:
        pose = Pose.read(f.read())
    data = np.asarray(pose.body.data[:, 0, :, :], dtype=np.float32)
    conf = np.asarray(pose.body.confidence[:, 0, :], dtype=np.float32)
    if data.shape[1] != N_LANDMARKS:
        raise ValueError(
            f"{path}: expected {N_LANDMARKS} landmarks, got {data.shape[1]} — "
            "the LSC pose layout changed; re-verify the block offsets before "
            "trusting this mapping.")
    return data, conf


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pose_file", type=Path)
    ap.add_argument("--channels", default="presence",
                    choices=["presence", "confidence"])
    args = ap.parse_args()

    data, conf = read_lsc_pose(args.pose_file)
    wb = convert(data, conf, channels=args.channels)
    print(f"{args.pose_file.name}: {data.shape} -> {wb.shape}")
    print(f"  x range {wb[:, :, 0].min():.1f}..{wb[:, :, 0].max():.1f} "
          f"(train box is 0..{TRAIN_SCALE:.0f})")
    print(f"  populated slots: {int((np.abs(wb).sum(axis=(0, 2)) > 0).sum())}/133")
    print(f"  mean presence: {wb[:, WB_IDX, 2].mean():.3f}")
