"""Keep the held-configuration segment of an LSC clip (eval-time only).

Port of the SL-GCN Sem-Lex leaf's hold selection (same method, same
defaults) adapted to the LSC .pose layout this eval reads: raw (T,586,3) pixel coordinates + a (T,586) confidence channel instead
of SEM-LEX's (T,553,3) NaN-coded npys.

Why: LSC handshape clips are citation-form demos — the signer raises the hand,
HOLDS the configuration, and lowers it. The transitions in/out are fast and
carry no handshape information, but the 150-frame crop/cyclic-pad feeds them
to the model anyway. This applies the selection before this repo's
converter, so the hand-anchored scaling in
map_lsc_pose_to_wholebody133.convert() is then estimated from the held
segment rather than from the transitions.

Method (unchanged from the SemLex port): per-frame hand-centroid speed,
normalised by shoulder width, smoothed; the longest run of hand-visible frames
below a per-clip speed percentile is the hold. Frames outside it are dropped
(the clip gets shorter; upstream's cyclic pad handles the rest).
"""
import numpy as np

from map_lsc_pose_to_wholebody133 import (LEFT_HAND_OFFSET, POSE_OFFSET,
                                          RIGHT_HAND_OFFSET)

L_SHOULDER, R_SHOULDER = POSE_OFFSET + 11, POSE_OFFSET + 12
HAND_SLICE = slice(LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET + 21)


def _hand_speed(data: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Per-frame hand-centroid speed, normalised by shoulder width."""
    ls, rs = data[:, L_SHOULDER, 0], data[:, R_SHOULDER, 0]
    sh_ok = present[:, L_SHOULDER] & present[:, R_SHOULDER]
    width = (float(np.median(np.abs(ls[sh_ok] - rs[sh_ok])))
             if sh_ok.any() else 1.0)
    if not np.isfinite(width) or width <= 1e-9:
        width = 1.0
    xy = data[:, HAND_SLICE, :2].astype(np.float64).copy()
    xy[~present[:, HAND_SLICE]] = np.nan
    with np.errstate(invalid="ignore"):
        centroid = np.nanmean(xy, axis=1)                     # (T, 2)
    speed = np.linalg.norm(np.diff(centroid, axis=0), axis=-1) / width
    return np.concatenate([speed, speed[-1:]]) if len(speed) else np.zeros(1)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    finite = x[np.isfinite(x)]
    # NaN frames (hands undetected) must not read as "slow" — fill them with
    # the clip's max speed so they are never selected as part of the hold.
    fill = float(finite.max()) if finite.size else 0.0
    kernel = np.ones(k) / k
    return np.convolve(np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill),
                       kernel, mode="same")


def _longest_run(mask: np.ndarray):
    """(start, end) of the longest True run; (0, len) if none."""
    best = cur_start = None
    best_len = 0
    for i, v in enumerate(mask):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best_len:
                best_len, best = i - cur_start, (cur_start, i)
            cur_start = None
    if cur_start is not None and len(mask) - cur_start > best_len:
        best_len, best = len(mask) - cur_start, (cur_start, len(mask))
    return best if best else (0, len(mask))


def select_hold(data: np.ndarray, conf: np.ndarray, speed_pct: float = 50.0,
                min_frames: int = 8, smooth_k: int = 5):
    """(T,586,3) data + (T,586) conf -> (hold data, hold conf, (start, end))."""
    present = conf > 0
    visible = present[:, HAND_SLICE].any(axis=1)
    if visible.sum() < min_frames:
        return data, conf, (0, len(data))  # too little to work with

    speed = _smooth(_hand_speed(data, present), smooth_k)
    thr = np.nanpercentile(speed[visible], speed_pct)
    # candidate hold frames: hands visible AND moving slowly
    mask = visible & (speed <= thr)
    start, end = _longest_run(mask)
    if end - start < min_frames:
        # fall back to a min_frames window around the slowest visible frame,
        # so a clip is never reduced to nothing
        idx = np.where(visible)[0]
        centre = idx[np.argmin(speed[idx])]
        half = min_frames // 2
        start, end = max(0, centre - half), min(len(data), centre + half + 1)
    return data[start:end], conf[start:end], (start, end)
