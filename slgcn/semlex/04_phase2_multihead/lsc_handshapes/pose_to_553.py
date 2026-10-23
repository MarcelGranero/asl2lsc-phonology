"""LSC benchmark `.pose` files -> SEM-LEX-layout (T, 553, 3) npys.

The stage-08 handshape clips ship as pose-format `.pose` (MediaPipe Holistic
via `video_to_pose`), but the SL-GCN chain in this repo starts from SEM-LEX's
`(T, 553, 3)` npy layout. The two order their components DIFFERENTLY:

    .pose component order   POSE_LANDMARKS(33), FACE_LANDMARKS(478),
                            LEFT_HAND(21), RIGHT_HAND(21),
                            POSE_WORLD_LANDMARKS(33)      = 586 points

    SEM-LEX 553 layout      [0:478]   face
                            [478:511] body (BlazePose 33)
                            [511:532] left hand
                            [532:553] right hand

So this is a reorder, not a crop: body moves from the front to offset 478 and
POSE_WORLD_LANDMARKS is dropped (SEM-LEX has no equivalent; it is a duplicate
of the body block in world coordinates).

Only 133 of the 553 points survive `map_553_to_wholebody133.convert`, and none
of them come from the face block — but the face block must still occupy the
right SLOTS or every downstream index shifts, so it is copied through rather
than zero-filled.

Coordinates: pose-format stores pixel coords with a `confidence` mask.
SEM-LEX stores normalised (x, y, z) with NaN for undetected landmarks, and
`map_553_to_wholebody133` reads channel 3 as presence. We therefore normalise
x/y by the recorded frame size and write NaN wherever confidence is 0, which
reproduces the SEM-LEX contract the converter was verified against.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

FACE_N = 478
BODY_N = 33
HAND_N = 21
TOTAL = FACE_N + BODY_N + HAND_N + HAND_N  # 553


def pose_to_553(pose) -> np.ndarray:
    """pose_format Pose -> (T, 553, 3) float32 in SEM-LEX component order."""
    names = [c.name for c in pose.header.components]
    sizes = [len(c.points) for c in pose.header.components]
    start = {}
    offset = 0
    for name, size in zip(names, sizes):
        start[name] = (offset, size)
        offset += size

    required = ("POSE_LANDMARKS", "FACE_LANDMARKS",
                "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS")
    missing = [r for r in required if r not in start]
    if missing:
        raise ValueError(f"pose file is missing components: {missing}")

    data = np.asarray(pose.body.data, dtype=np.float32)  # (T, people, P, 3)
    conf = np.asarray(pose.body.confidence, dtype=np.float32)  # (T, people, P)
    if data.shape[1] < 1:
        raise ValueError("pose file has no people")
    data = data[:, 0]  # (T, P, 3) — single signer
    conf = conf[:, 0]  # (T, P)

    T = data.shape[0]
    out = np.full((T, TOTAL, 3), np.nan, dtype=np.float32)

    def block(name):
        off, size = start[name]
        return data[:, off:off + size], conf[:, off:off + size]

    face, face_c = block("FACE_LANDMARKS")
    body, body_c = block("POSE_LANDMARKS")
    left, left_c = block("LEFT_HAND_LANDMARKS")
    right, right_c = block("RIGHT_HAND_LANDMARKS")

    # Normalise pixel coords to [0,1] using the header's frame size, matching
    # SEM-LEX's normalised storage. z is left as-is (both are relative depth).
    width = float(getattr(pose.header, "dimensions").width)
    height = float(getattr(pose.header, "dimensions").height)
    scale = np.array([width or 1.0, height or 1.0, 1.0], dtype=np.float32)

    for dst_slice, (arr, c) in (
        (slice(0, FACE_N), (face, face_c)),
        (slice(FACE_N, FACE_N + BODY_N), (body, body_c)),
        (slice(FACE_N + BODY_N, FACE_N + BODY_N + HAND_N), (left, left_c)),
        (slice(FACE_N + BODY_N + HAND_N, TOTAL), (right, right_c)),
    ):
        vals = arr / scale
        vals[c == 0] = np.nan  # undetected -> NaN, the SEM-LEX convention
        out[:, dst_slice] = vals

    return out


def resample_fps(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample a (T, P, 3) sequence from src_fps to dst_fps by index selection.

    The model consumes a fixed 150-FRAME window, so frame rate is not a free
    parameter: LSC `.pose` is 50 fps against SEM-LEX's 30, which stretches LSC
    signs ~1.7x in frames and shows the temporal-attention layers motion at
    ~0.6x the speed they were trained on.

    Nearest-neighbour index selection rather than interpolation: it is what
    downsampling a pose track to a lower rate physically means (drop frames),
    it cannot invent landmark positions between observed frames, and it leaves
    NaN (undetected) markers intact instead of smearing them into neighbours.
    """
    if src_fps == dst_fps:
        return arr
    n_src = arr.shape[0]
    n_dst = max(1, int(round(n_src * dst_fps / src_fps)))
    idx = np.minimum((np.arange(n_dst) * src_fps / dst_fps).astype(int), n_src - 1)
    return arr[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-fps", type=float, default=None,
                    help="Resample to this rate (use 30 to match SEM-LEX "
                         "training). Source rate is read from each file's "
                         "header. Default: keep the native rate.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from pose_format import Pose

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.pose_dir.rglob("*.pose"))
    if not files:
        raise SystemExit(f"no .pose files under {args.pose_dir}")

    n_ok = n_skip = 0
    lengths = []
    for path in files:
        out = args.out_dir / f"{path.stem}.npy"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        with open(path, "rb") as f:
            pose = Pose.read(f.read())
        arr = pose_to_553(pose)
        if args.target_fps:
            src = float(pose.body.fps)
            arr = resample_fps(arr, src, args.target_fps)
        lengths.append(arr.shape[0])
        np.save(out, arr)
        n_ok += 1

    note = ""
    if args.target_fps and lengths:
        note = (f" @ {args.target_fps:g}fps, "
                f"mean {np.mean(lengths):.0f} frames")
    print(f"{n_ok} converted, {n_skip} skipped -> {args.out_dir}{note}")


if __name__ == "__main__":
    main()
