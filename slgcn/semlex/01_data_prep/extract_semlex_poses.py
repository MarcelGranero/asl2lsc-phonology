"""Re-extract SEM-LEX poses from video with MediaPipe Holistic.

The released SEM-LEX poses are missing their hands: mean hand presence .248
over the training split, 73% zeros in the packed tensor (see
`04_phase2_multihead/lsc_handshapes/`). That bounds every pose-based
handshape result in this repo. The videos are all present on disk, so the
poses can be rebuilt.

Output matches the RELEASE layout exactly — `(T, 553, 3)` float32 npys,
components in SEM-LEX order (face 478, body 33, left hand 21, right hand 21),
x/y normalised to [0,1], NaN where a landmark was not detected — so every
downstream consumer (`convert_poses.py`, `map_553_to_wholebody133.py`,
`run_gendata.py`) works unchanged against a re-extracted directory.

Why not `video_to_pose` (the CLI used for the LSC benchmark clips): the
SEM-LEX webm headers declare 0 frames / 0 fps, so the pose-format reader's
exact-frame path raises "Failed to open video". OpenCV decodes them fine, so
this reads frames with cv2 and calls MediaPipe Holistic directly with the same
config the benchmark uses (`model_complexity=2`, `refine_face_landmarks`).

Resumable: existing non-empty outputs are skipped unless --force. Writes to a
temp file and renames, so an interrupted run cannot leave a truncated npy that
a later resume would skip.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

FACE_N, BODY_N, HAND_N = 478, 33, 21
TOTAL = FACE_N + BODY_N + HAND_N + HAND_N  # 553


def landmarks_to_array(result, n_face_expected=FACE_N):
    """MediaPipe Holistic result -> (553, 3), NaN where undetected."""
    out = np.full((TOTAL, 3), np.nan, dtype=np.float32)

    def fill(landmark_list, start, count):
        if landmark_list is None:
            return
        pts = landmark_list.landmark
        for i, lm in enumerate(pts[:count]):
            out[start + i] = (lm.x, lm.y, lm.z)

    fill(result.face_landmarks, 0, FACE_N)
    fill(result.pose_landmarks, FACE_N, BODY_N)
    fill(result.left_hand_landmarks, FACE_N + BODY_N, HAND_N)
    fill(result.right_hand_landmarks, FACE_N + BODY_N + HAND_N, HAND_N)
    return out


def extract_one(video_path: Path, holistic) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        frames.append(landmarks_to_array(holistic.process(rgb)))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*.webm")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N videos (probing)")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's index, for SLURM array sharding")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()

    import mediapipe as mp

    videos = sorted(args.video_dir.glob(args.glob))
    if args.limit:
        videos = videos[:args.limit]
    videos = videos[args.shard::args.num_shards]
    if not videos:
        raise SystemExit(f"no videos matching {args.glob} in {args.video_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    presences = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        refine_face_landmarks=True,
    ) as holistic:
        for i, video in enumerate(videos):
            out = args.out_dir / f"{video.stem}.npy"
            if out.exists() and out.stat().st_size > 0 and not args.force:
                n_skip += 1
                continue
            try:
                arr = extract_one(video, holistic)
            except Exception as exc:  # noqa: BLE001 — one bad file must not stop a shard
                print(f"FAIL {video.name}: {exc}", flush=True)
                n_fail += 1
                continue
            # np.save appends '.npy' unless the name already ends in it, so
            # the temp name must end in '.npy' or the rename target is wrong.
            tmp = out.with_name(f"{out.stem}.tmp{os.getpid()}.npy")
            np.save(tmp, arr)
            os.replace(tmp, out)
            hands = arr[:, FACE_N + BODY_N:, 0]
            presences.append(1 - float(np.isnan(hands).mean()))
            n_ok += 1
            if args.report_every and n_ok % args.report_every == 0:
                print(f"[{i+1}/{len(videos)}] ok={n_ok} skip={n_skip} "
                      f"fail={n_fail} mean_hand_presence="
                      f"{np.mean(presences):.3f}", flush=True)

    msg = f"done: ok={n_ok} skip={n_skip} fail={n_fail}"
    if presences:
        msg += f"  mean hand presence {np.mean(presences):.3f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
