"""Frame-rate and framing adaptation for the LSC eval.

Ports two preprocessing steps of the SL-GCN Sem-Lex leaf's LSC eval
(frame-rate resampling, as in `pose_to_553.py --target-fps`, and reframing),
retargeted from SEM-LEX's geometry to PopSign's, so a PopSign-vs-SEM-LEX
comparison can apply the same preprocessing to both corpora. Both are
optional eval-time treatments; the released LSC numbers use neither.

Both operate on the raw LSC `.pose` arrays, BEFORE
map_lsc_pose_to_wholebody133.convert().

--------------------------------------------------------------------------
1. FRAME RATE (resample_fps)

The model consumes a fixed 150-FRAME window, so frame rate is not a free
parameter. Measured:

    PopSign training video   ~30 fps  (29.92-30.00 over sampled clips)
    LSC .pose                 50 fps

So an LSC sign occupies ~1.67x as many frames as the same-duration PopSign
sign, and the temporal-attention layers see motion at ~0.6x the speed they
were trained on. Identical mismatch to SEM-LEX's (also 30 fps), so the same
50 -> 30 resample applies.

Nearest-neighbour index selection, not interpolation — matching the sibling
port. Dropping frames IS what downsampling a pose track physically means; it
cannot invent landmark positions between observations, and it leaves
undetected markers intact rather than smearing them into neighbours.

--------------------------------------------------------------------------
2. FRAMING (reframe_to_popsign)

A similarity transform (uniform scale + translation, NO aspect distortion)
placing the LSC signer at PopSign's shoulder geometry. Differs from the
`person`/`hand` scale modes in map_lsc_pose_to_wholebody133 in that it also
sets the VERTICAL position, and works in normalised frame coordinates before
the 512 lift — the same formulation as the sibling's reframe.

Scale and offset are computed ONCE per clip from median shoulder geometry,
not per frame: a per-frame transform would cancel the signer's own motion,
which is signal.

`clip_outside` additionally drops landmarks that fall outside the frame,
simulating a truncation defect rather than the framing. Off by default —
discarding the well-tracked LSC hands would remove the very thing being
evaluated.
"""
import numpy as np

from map_lsc_pose_to_wholebody133 import (LSC_HEIGHT, LSC_WIDTH, POSE_OFFSET)

# BlazePose landmark ids within the body block
L_SHOULDER, R_SHOULDER = 11, 12

# Measured directly on the popsign val packed tensors (median over 75 clips
# of the per-clip median), converted to normalised frame units — packed x/y
# are the 512-lifted normalised coords, so normalised = packed / 512:
#   shoulder |dx| 275.0 px -> 0.537
#   shoulder y    351.2 px -> 0.686
# Note |dx| (horizontal separation), matching the sibling's reframe, not the
# euclidean shoulder distance used by map_lsc_pose_to_wholebody133's
# `person` mode — the two differ slightly when the signer is rotated.
POPSIGN_SHOULDER_W = 0.537
POPSIGN_SHOULDER_Y = 0.686

POPSIGN_FPS = 30.0
LSC_FPS = 50.0


def resample_fps(data, conf, src_fps=LSC_FPS, dst_fps=POPSIGN_FPS):
    """(T,P,3)+(T,P) -> resampled to dst_fps by nearest-neighbour indexing."""
    if src_fps == dst_fps or data.shape[0] == 0:
        return data, conf
    n_src = data.shape[0]
    n_dst = max(1, int(round(n_src * dst_fps / src_fps)))
    idx = np.minimum((np.arange(n_dst) * src_fps / dst_fps).astype(int),
                     n_src - 1)
    return data[idx], conf[idx]


def reframe_to_popsign(data, conf, target_w=POPSIGN_SHOULDER_W,
                       target_y=POPSIGN_SHOULDER_Y, clip_outside=False):
    """(T,586,3) LSC PIXELS -> same, at PopSign signer geometry.

    Input is in LSC pixel coordinates (1920x1080); we normalise, apply the
    similarity transform, and return pixels again so the rest of the pipeline
    is unchanged.
    """
    out = data.copy()
    conf_out = conf.copy()

    # -> normalised frame coords
    out[:, :, 0] /= LSC_WIDTH
    out[:, :, 1] /= LSC_HEIGHT

    ls = out[:, POSE_OFFSET + L_SHOULDER, :2]
    rs = out[:, POSE_OFFSET + R_SHOULDER, :2]
    vis = (conf[:, POSE_OFFSET + L_SHOULDER] > 0) & \
          (conf[:, POSE_OFFSET + R_SHOULDER] > 0)
    if not vis.any():
        out[:, :, 0] *= LSC_WIDTH
        out[:, :, 1] *= LSC_HEIGHT
        return out, conf_out          # degenerate; leave untouched

    width = float(np.median(np.abs(ls[vis, 0] - rs[vis, 0])))
    if not np.isfinite(width) or width <= 1e-6:
        out[:, :, 0] *= LSC_WIDTH
        out[:, :, 1] *= LSC_HEIGHT
        return out, conf_out

    scale = target_w / width
    cx = float(np.median((ls[vis, 0] + rs[vis, 0]) / 2))
    cy = float(np.median((ls[vis, 1] + rs[vis, 1]) / 2))

    # uniform scale about the shoulder midpoint; centre horizontally and put
    # the shoulders at PopSign's height
    out[:, :, 0] = (out[:, :, 0] - cx) * scale + 0.5
    out[:, :, 1] = (out[:, :, 1] - cy) * scale + target_y

    if clip_outside:
        off = ((out[:, :, 0] < 0) | (out[:, :, 0] > 1) |
               (out[:, :, 1] < 0) | (out[:, :, 1] > 1))
        out[:, :, :2][off] = 0.0
        conf_out[off] = 0.0

    # back to pixels for the downstream converter
    out[:, :, 0] *= LSC_WIDTH
    out[:, :, 1] *= LSC_HEIGHT
    # undetected landmarks stay at the origin
    missing = conf_out <= 0
    out[:, :, :2][missing] = 0.0
    return out, conf_out
