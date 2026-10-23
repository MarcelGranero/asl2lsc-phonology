"""Re-extract SEM-LEX poses from the released videos with MediaPipe Holistic.

WHY: the released `(T, 553, 3)` npys are largely missing HANDS — mean hand
presence 0.248 over 800 train clips, 73% zeros in the packed hand joints.
Both shortfalls we measured (phase-1 gloss −10 pts, phase-2
features −6.4 pts) are consistent with that single input defect, and the
converged-encoder run ruled out encoder quality as the cause. This script
tests the hypothesis directly: extract poses ourselves and measure whether
hand presence rises.

Writes the SAME layout the release uses, so every downstream stage
(convert_poses -> run_gendata -> train) works unchanged against a different
--poses-dir:

    [0:478]   face + iris   (468 mesh + 10 iris, refine_face_landmarks=True)
    [478:511] body          (BlazePose 33)
    [511:532] left hand     (21)
    [532:553] right hand    (21)

Coordinates are normalised (x, y, z) exactly as the release stores them, and
UNDETECTED LANDMARKS ARE NaN — `map_553_to_wholebody133.convert` reads NaN as
presence 0, so writing zeros here would silently fake presence.

Settings mirror the LSC benchmark extractor (model_complexity=2,
refine_face_landmarks=True), which is the strongest Holistic configuration and
the one whose output the 553 layout was verified against.

Run in the `pose_estimation` env (mediapipe 0.10.11), NOT slgcn-env.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_config import PROCESSED_DIR, SPLITS_JSON, VIDEOS_DIR

N_FACE = 478          # 468 mesh + 10 iris (refine_face_landmarks=True)
N_POSE = 33
N_HAND = 21
N_TOTAL = 553         # 478 + 33 + 21 + 21
FACE_OFF, POSE_OFF, LH_OFF, RH_OFF = 0, 478, 511, 532

DEFAULT_OUT = PROCESSED_DIR / "poses_reextracted"


def landmarks_to_block(landmarks, n_expected):
    """MediaPipe landmark list -> (n_expected, 3) float32, NaN when absent."""
    block = np.full((n_expected, 3), np.nan, dtype=np.float32)
    if landmarks is None:
        return block
    for i, lm in enumerate(landmarks.landmark):
        if i >= n_expected:
            break
        block[i] = (lm.x, lm.y, lm.z)
    return block


def extract_one(video_path: Path, holistic) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)

        arr = np.full((N_TOTAL, 3), np.nan, dtype=np.float32)
        arr[FACE_OFF:FACE_OFF + N_FACE] = landmarks_to_block(
            res.face_landmarks, N_FACE)
        arr[POSE_OFF:POSE_OFF + N_POSE] = landmarks_to_block(
            res.pose_landmarks, N_POSE)
        arr[LH_OFF:LH_OFF + N_HAND] = landmarks_to_block(
            res.left_hand_landmarks, N_HAND)
        arr[RH_OFF:RH_OFF + N_HAND] = landmarks_to_block(
            res.right_hand_landmarks, N_HAND)
        frames.append(arr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    # float16 matches the release's dtype
    return np.stack(frames).astype(np.float16)


def hand_presence(arr: np.ndarray) -> float:
    """Fraction of (frame, hand-landmark) cells that are detected."""
    hands = arr[:, LH_OFF:RH_OFF + N_HAND, :]
    return float((~np.isnan(hands).any(axis=2)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all clips in the split; the pilot uses a few "
                         "hundred, which is enough to measure presence")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", type=Path, default=None,
                    help="write a per-clip presence CSV here")
    args = ap.parse_args()

    from mediapipe.python.solutions import holistic as mp_holistic

    ids = json.loads(SPLITS_JSON.read_text())[args.split]
    if args.limit:
        ids = ids[:args.limit]
    ids = ids[args.shard::args.num_shards]
    out_split = args.out_dir / args.split
    out_split.mkdir(parents=True, exist_ok=True)

    rows, done, skipped, failed = [], 0, 0, 0
    with mp_holistic.Holistic(static_image_mode=False,
                              model_complexity=2,
                              smooth_landmarks=True,
                              refine_face_landmarks=True) as holistic:
        for i, vid in enumerate(ids):
            out = out_split / f"{vid}.npy"
            if out.exists() and not args.force:
                skipped += 1
                continue
            video = VIDEOS_DIR[args.split] / f"{vid}.webm"
            if not video.exists():
                failed += 1
                print(f"MISSING VIDEO {vid}", flush=True)
                continue
            try:
                arr = extract_one(video, holistic)
            except Exception as e:  # noqa: BLE001 — keep going, report at end
                failed += 1
                print(f"FAIL {vid}: {e}", flush=True)
                continue
            tmp = out.with_suffix(f".tmp{os.getpid()}.npy")
            np.save(tmp, arr)
            os.replace(tmp, out)          # atomic: no half-written npy
            rows.append((vid, arr.shape[0], hand_presence(arr)))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(ids)} mean_presence="
                      f"{np.mean([r[2] for r in rows]):.3f}", flush=True)

    print(f"\n{args.split} shard {args.shard}/{args.num_shards}: "
          f"{done} extracted, {skipped} skipped, {failed} failed")
    if rows:
        pres = np.array([r[2] for r in rows])
        print(f"HAND PRESENCE (re-extracted): mean {pres.mean():.3f} "
              f"median {np.median(pres):.3f} "
              f"frac<0.10 {(pres < 0.10).mean():.3f} "
              f"frac>0.80 {(pres > 0.80).mean():.3f}")
        print("RELEASED baseline for comparison: mean 0.248, "
              "15% of clips <0.10, none >0.80")
    if args.report and rows:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            f.write("video_id,n_frames,hand_presence\n")
            for vid, n, p in rows:
                f.write(f"{vid},{n},{p:.4f}\n")
        print(f"per-clip report -> {args.report}")


if __name__ == "__main__":
    main()
