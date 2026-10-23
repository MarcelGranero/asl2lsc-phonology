"""Read a PopSign Kaggle-style landmark parquet into a dense (T, 543, 3) array.

This is the one component with no SEM-LEX counterpart: SEM-LEX ships dense
npys, PopSign ships the Kaggle ASL-Signs LONG format, one row per
(frame, landmark):

    frame int16 | row_id str | type str | landmark_index int16 | x,y,z double

`type` is one of face/pose/left_hand/right_hand and `landmark_index` restarts
at 0 within each type, so the dense index is
`BLOCK_OFFSET[type] + landmark_index` (block order face, pose, left, right —
the same order row_id enumerates).

Rows are complete and regular in every file inspected (543 rows per frame,
frames 0..T-1 contiguous), but we do NOT rely on that: the pivot scatters by
explicit index into a NaN-filled array, so missing rows stay NaN — which is
already the dataset's "undetected" encoding and what the presence channel
consumes. Absent frames are reported by `pose_stats`, not silently dropped.
"""
import numpy as np
import pyarrow.parquet as pq

from map_543_to_wholebody133 import BLOCK_OFFSET, N_LANDMARKS

COLUMNS = ["frame", "type", "landmark_index", "x", "y", "z"]


def read_pose(path) -> np.ndarray:
    """Parquet path -> (T, 543, 3) float32, NaN where undetected/absent."""
    table = pq.read_table(path, columns=COLUMNS)
    frame = table.column("frame").to_numpy(zero_copy_only=False).astype(np.int64)
    ltype = table.column("type").to_numpy(zero_copy_only=False)
    lidx = table.column("landmark_index").to_numpy(
        zero_copy_only=False).astype(np.int64)

    if frame.size == 0:
        raise ValueError(f"{path}: empty parquet")

    # Frames are 0-based and contiguous in this release; normalise defensively
    # so a non-zero first frame can never produce a leading block of NaNs.
    uniq = np.unique(frame)
    if uniq[0] != 0 or uniq[-1] != uniq.size - 1:
        remap = np.full(int(uniq[-1]) + 1, -1, dtype=np.int64)
        remap[uniq] = np.arange(uniq.size)
        frame = remap[frame]
    T = int(uniq.size)

    # type -> dense block offset, vectorised over the 4 known values
    offset = np.full(ltype.shape, -1, dtype=np.int64)
    for name, off in BLOCK_OFFSET.items():
        offset[ltype == name] = off
    if (offset < 0).any():
        bad = sorted(set(ltype[offset < 0].tolist()))
        raise ValueError(f"{path}: unknown landmark type(s) {bad}")

    dense = offset + lidx
    if dense.max() >= N_LANDMARKS or dense.min() < 0:
        raise ValueError(
            f"{path}: landmark index out of range "
            f"[{dense.min()}, {dense.max()}] vs {N_LANDMARKS}")

    out = np.full((T, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for c, name in enumerate(("x", "y", "z")):
        out[frame, dense, c] = table.column(name).to_numpy(
            zero_copy_only=False).astype(np.float32)
    return out


def pose_stats(pose: np.ndarray) -> dict:
    """Per-clip facts worth aggregating before training (see frame_stats.json)."""
    from map_543_to_wholebody133 import (LEFT_HAND_OFFSET, POSE_OFFSET,
                                         RIGHT_HAND_OFFSET)
    lh = pose[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 21, 0]
    rh = pose[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 21, 0]
    body = pose[:, POSE_OFFSET:POSE_OFFSET + 33, :2]
    xy = pose[:, :, :2]
    finite = np.isfinite(xy)
    return {
        "n_frames": int(pose.shape[0]),
        # upstream sign_gendata crops to 150 frames / cyclically pads below it
        "over_150": bool(pose.shape[0] > 150),
        "frac_nan_left_hand": float(np.isnan(lh).mean()),
        "frac_nan_right_hand": float(np.isnan(rh).mean()),
        "no_hand_frames": float(
            (np.isnan(lh).all(axis=1) & np.isnan(rh).all(axis=1)).mean()),
        "frac_body_nan": float(np.isnan(body).mean()),
        # coords outside [0,1]: MediaPipe extrapolation, kept not clipped
        "frac_xy_out_of_unit": float(
            ((xy < 0) | (xy > 1))[finite].mean()) if finite.any() else 0.0,
    }
