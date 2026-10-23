"""PopSign 543-point MediaPipe Holistic -> COCO-WholeBody 133-point layout.

Upstream SAM-SLR-v2 preprocessing (SL-GCN/data_gen/sign_gendata.py) expects
per-clip arrays of shape (T, 133, 3) in COCO-WholeBody ordering and selects
its 27 joints from them (`selected_joints['27']`). We scatter the PopSign
MediaPipe landmarks into the wholebody slots and let ALL upstream code run
unchanged.

PopSign pose layout (T, 543, 3) — the Kaggle ASL-Signs schema, contiguous in
this order (verified from the parquet `type`/`landmark_index` columns):
  [0:468]   face mesh (468)
  [468:501] body pose (BlazePose 33)
  [501:522] left hand (21), [522:543] right hand (21)

RELATION TO SEM-LEX: SEM-LEX poses are the SAME
MediaPipe Holistic topology run with `refine_face_landmarks=True`, giving
553 points — 10 extra IRIS landmarks at [468:478] that shift every later
block by +10. Confirmed empirically: in SEM-LEX each 5-point iris cluster
sits between its eye corners (L centroid x=0.505 between mesh corners 0.486
and 0.527), spread ~0.02. This file is `map_553_to_wholebody133.py` with the
three offsets decremented by 10; the body/hand maps are otherwise identical
because the iris points sit past the mesh and before the pose block. Nothing
downstream reads face slots anyway — `selected_joints['27']` covers body
0-16 and hands 91-132 only.

HAND SIDE: MediaPipe's `left_hand`/`right_hand` are ANATOMICAL, not screen
side. Measured over 30 random clips, `left_hand` mean x sits +0.112 to the
screen-RIGHT of the body centroid and `right_hand` -0.199 to the screen-left
— i.e. the person's left hand appears screen-right, the mirror convention
SEM-LEX's [511:532] block also follows. So the SEM-LEX hand mapping carries
over unchanged. Verify with verify_skeleton_overlay.py before training; the
equivalent assumption on SEM-LEX was wrong once and only the overlay caught it.

COCO-WholeBody slots we populate:
  body 0-16  <- MediaPipe pose (nose, eyes, ears, shoulders, elbows, wrists,
               hips, knees, ankles)
  91-111     <- left hand, 112-132 <- right hand (both conventions use the
               standard 21-point MANO-style ordering)
Slots 17-22 (feet) and 23-90 (face) stay zero — never indexed by '27'.

Coordinate contract (data decisions, documented in docs/KEYPOINT_MAPPING.md):
  - x, y are scaled by 512: the unpatched upstream feeder augments in the
    512x512 pixel space of its original crops (mirror = `512 - x`, shift =
    +-10 px), so normalised coords MUST be lifted to that frame for the
    augmentation semantics to hold. NOTE PopSign coords routinely fall
    OUTSIDE [0,1] (MediaPipe extrapolates out-of-frame joints; sampled range
    -0.57..1.56), so scaled values land outside the nominal 512 box. The
    mirror is affine and survives this; we keep upstream behaviour and
    record the out-of-range fraction in results/frame_stats.json rather than
    clipping, which would move real joints.
  - channel 3 upstream is a detector confidence in [0, 1]. MediaPipe exports
    none, so the default is a presence indicator (1.0 detected / 0.0 NaN),
    the closest conservative analogue. `--channels xyz` keeps MediaPipe z.
  - NaN -> 0 after scaling (undetected joints at origin, upstream-zero style).
"""
import numpy as np

SCALE = 512.0

N_LANDMARKS = 543
POSE_OFFSET = 468        # SEM-LEX: 478 (+10 iris)
LEFT_HAND_OFFSET = 501   # SEM-LEX: 511
RIGHT_HAND_OFFSET = 522  # SEM-LEX: 532

# Contiguous block order in the Kaggle long format, used to pivot parquet rows.
BLOCK_ORDER = [("face", 468), ("pose", 33), ("left_hand", 21),
               ("right_hand", 21)]
BLOCK_OFFSET = {}
_off = 0
for _name, _n in BLOCK_ORDER:
    BLOCK_OFFSET[_name] = _off
    _off += _n
assert _off == N_LANDMARKS

# (wholebody_idx, mediapipe543_idx); BlazePose landmark ids + POSE_OFFSET
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
MP_IDX = np.array([m for _, m in FULL_MAP])


def convert(pose_543: np.ndarray, channels: str = "presence") -> np.ndarray:
    """(T, 543, 3) -> (T, 133, 3) float32 in wholebody layout."""
    assert pose_543.ndim == 3 and pose_543.shape[1:] == (N_LANDMARKS, 3), \
        pose_543.shape
    T = pose_543.shape[0]
    src = pose_543[:, MP_IDX, :].astype(np.float32)
    present = ~np.isnan(src).any(axis=2)  # (T, 59)

    out = np.zeros((T, 133, 3), dtype=np.float32)
    xy = np.nan_to_num(src[:, :, :2], nan=0.0) * SCALE
    out[:, WB_IDX, :2] = xy
    if channels == "presence":
        out[:, WB_IDX, 2] = present.astype(np.float32)
    elif channels == "xyz":
        out[:, WB_IDX, 2] = np.nan_to_num(src[:, :, 2], nan=0.0)
    else:
        raise ValueError(f"unknown channels mode: {channels}")
    return out


def dump_table() -> str:
    names = {0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear",
             4: "right_ear", 5: "left_shoulder", 6: "right_shoulder",
             7: "left_elbow", 8: "right_elbow", 9: "left_wrist",
             10: "right_wrist", 11: "left_hip", 12: "right_hip",
             13: "left_knee", 14: "right_knee", 15: "left_ankle",
             16: "right_ankle"}
    hand = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
    lines = ["| wholebody_idx | name | mediapipe543_idx | (sem-lex 553 idx) |",
             "|---|---|---|---|"]
    for w, m in BODY_MAP:
        lines.append(f"| {w} | {names[w]} | {m} | {m + 10} |")
    for i, (w, m) in enumerate(LEFT_HAND_MAP):
        lines.append(f"| {w} | left_{hand[i]} | {m} | {m + 10} |")
    for i, (w, m) in enumerate(RIGHT_HAND_MAP):
        lines.append(f"| {w} | right_{hand[i]} | {m} | {m + 10} |")
    lines.append("")
    lines.append(f"Populated slots: {len(FULL_MAP)} (17 body + 21 + 21 hands); "
                 "slots 17-90 (feet+face) stay zero and are never selected by "
                 "the '27' joint config. The last column is the corresponding "
                 "SEM-LEX 553-point index (= +10, the refined-face iris block).")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_config import DOCS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-table", action="store_true")
    args = parser.parse_args()
    table = dump_table()
    print(table)
    if args.dump_table:
        out = DOCS_DIR / "KEYPOINT_MAPPING.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# PopSign 543 -> COCO-WholeBody 133 mapping\n\n"
            "Generated by `01_data_prep/map_543_to_wholebody133.py "
            "--dump-table`. See module docstring for the coordinate contract "
            "(x,y scaled by 512; channel 3 = presence indicator by default) "
            "and for the relation to the SEM-LEX 553-point layout.\n\n"
            + table + "\n")
        print(f"\nwritten to {out}")
