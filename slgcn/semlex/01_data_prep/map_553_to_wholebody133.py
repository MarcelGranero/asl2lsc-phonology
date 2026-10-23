"""SEM-LEX 553-point MediaPipe Holistic -> COCO-WholeBody 133-point layout.

Upstream SAM-SLR-v2 preprocessing (SL-GCN/data_gen/sign_gendata.py) expects
per-video arrays of shape (T, 133, 3) in COCO-WholeBody ordering and selects
its 27 joints from them (`selected_joints['27']`). Rather than reimplementing
the selection/graph/mirror logic, we scatter the SEM-LEX MediaPipe landmarks
into the wholebody slots and let ALL upstream code run unchanged.

SEM-LEX pose layout (T, 553, 3), float16, (x, y, z) normalised, NaN=undetected
— verified empirically (anatomy of offset-478 block: shoulders at
y~0.57 with left appearing screen-right, elbows y~0.95, hips y~1.28 below
frame; hand-block screen sides over 30 files: [532:553] screen-left 97% =
person's right hand):
  [0:478]   face mesh incl. iris (478 points)
  [478:511] body pose (BlazePose 33; always present — MediaPipe extrapolates
            out-of-frame joints, e.g. legs at y>1)
  [511:532] left hand (21), [532:553] right hand (21)
An earlier assumption ([0:33] pose first, hands at 501/522) rendered face
points as "shoulders" and a hip as "hand wrist" — caught by the overlay QA.

COCO-WholeBody slots we populate (everything `selected_joints` can reach):
  body 0-16  <- MediaPipe pose (nose, eyes, ears, shoulders, elbows, wrists,
               hips, knees, ankles)
  91-111     <- left hand (both conventions use the standard 21-point
               MANO-style hand ordering: 0=wrist, 1-4 thumb, 5-8 index,
               9-12 middle, 13-16 ring, 17-20 pinky)
  112-132    <- right hand
Slots 17-22 (feet) and 23-90 (face) stay zero: no MediaPipe equivalent needed —
`selected_joints['27']` (and '31'/'59') never index them... except '59'/'31'
touch body 0-16 only plus hands, so zeros are never read by the '27' config we
run.

Coordinate contract (data decisions, documented in docs/KEYPOINT_MAPPING.md):
  - x, y are scaled by 512: the unpatched upstream feeder augments in the
    512x512 pixel space of its original crops (mirror = `512 - x`, shift =
    +-10 px), so normalised coords MUST be lifted to that frame for the
    augmentation semantics to hold.
  - channel 3 upstream is a detector confidence in [0, 1]. MediaPipe exports
    none, so the default is a presence indicator (1.0 detected / 0.0 NaN),
    the closest conservative analogue. `--channels xyz` keeps MediaPipe z
    (unscaled, magnitude ~O(0.1-1), commensurate with upstream confidences).
  - NaN -> 0 after scaling (undetected joints at origin, upstream-zero style).
"""
import numpy as np

SCALE = 512.0

POSE_OFFSET = 478
LEFT_HAND_OFFSET = 511
RIGHT_HAND_OFFSET = 532

# (wholebody_idx, mediapipe553_idx); BlazePose landmark ids + POSE_OFFSET
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


def convert(pose_553: np.ndarray, channels: str = "presence") -> np.ndarray:
    """(T, 553, 3) float16 -> (T, 133, 3) float32 in wholebody layout."""
    assert pose_553.ndim == 3 and pose_553.shape[1:] == (553, 3), pose_553.shape
    T = pose_553.shape[0]
    src = pose_553[:, MP_IDX, :].astype(np.float32)
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
    lines = ["| wholebody_idx | name | mediapipe553_idx |", "|---|---|---|"]
    for w, m in BODY_MAP:
        lines.append(f"| {w} | {names[w]} | {m} |")
    for i, (w, m) in enumerate(LEFT_HAND_MAP):
        lines.append(f"| {w} | left_{hand[i]} | {m} |")
    for i, (w, m) in enumerate(RIGHT_HAND_MAP):
        lines.append(f"| {w} | right_{hand[i]} | {m} |")
    lines.append("")
    lines.append(f"Populated slots: {len(FULL_MAP)} (17 body + 21 + 21 hands); "
                 "slots 17-90 (feet+face) stay zero and are never selected by "
                 "the '27' joint config.")
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
            "# SEM-LEX 553 -> COCO-WholeBody 133 mapping\n\n"
            "Generated by `01_data_prep/map_553_to_wholebody133.py "
            "--dump-table`. See module docstring for the coordinate contract "
            "(x,y scaled by 512; channel 3 = presence indicator by default).\n\n"
            + table + "\n")
        print(f"\nwritten to {out}")
