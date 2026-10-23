"""Visual QA: render the 27 selected joints over real video frames.

Loads a few clips per split, converts their 543-point landmarks through our
wholebody mapping, applies upstream's own selected_joints['27'], and draws
points + skeleton edges on sampled frames. Output PNG contact sheets under
docs/overlay_qa/ — EYEBALL THESE BEFORE TRAINING.

This is not optional ceremony: the equivalent SEM-LEX assumption about block
offsets and hand sides was WRONG and survived review until the overlay
exposed face points drawn as shoulders. The two PopSign-specific things to
check here:
  * hands land on hands (the `left_hand`/`right_hand` naming is anatomical,
    so the person's left hand appears screen-RIGHT);
  * joints outside the frame are expected — MediaPipe extrapolates, and
    PopSign is phone-shot portrait video with signers often partly out of
    frame. Points beyond the border are data, not a mapping bug.

Frame counts: landmark frames and decoded video frames can differ, so the
pose index is mapped proportionally onto the video (same as the SEM-LEX QA).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_config import (DOCS_DIR, LANDMARKS_DIR, SL_GCN_DIR, SPLIT_DIRS,
                         SPLITS_JSON)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_543_to_wholebody133 import SCALE, convert
from read_parquet_pose import read_pose

sys.path.insert(0, str(SL_GCN_DIR / "data_gen"))
from sign_gendata import selected_joints  # noqa: E402

sys.path.insert(0, str(SL_GCN_DIR))
from graph.sign_27 import inward  # noqa: E402

SEL27 = selected_joints["27"]
MAX_TILE_W = 400  # portrait 2592x1944 frames are downscaled for the sheet


def find_video(clip_id: str, split: str, label: str | None = None):
    root = SPLIT_DIRS[split]
    if label:
        p = root / label / f"{clip_id}.mp4"
        if p.exists():
            return p
    # label unknown: the class is the 2nd '-'-separated field of the clip id
    parts = clip_id.split("-")
    if len(parts) >= 2:
        p = root / parts[1] / f"{clip_id}.mp4"
        if p.exists():
            return p
    hits = list(root.glob(f"*/{clip_id}.mp4"))
    return hits[0] if hits else None


def render_clip(clip_id: str, split: str, out_dir: Path, n_frames: int = 6):
    video = find_video(clip_id, split)
    if video is None:
        print(f"  {clip_id}: no mp4 found, skipping")
        return
    pose = read_pose(LANDMARKS_DIR / f"{clip_id}.parquet")
    wb = convert(pose)                      # (T, 133, 3), xy in 512-space
    pts27 = wb[:, SEL27, :]                 # (T, 27, 3)

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"  {clip_id}: no frames decoded, skipping")
        return
    T_pose, T_vid = pts27.shape[0], len(frames)

    idxs = np.linspace(0, T_pose - 1, n_frames).astype(int)
    tiles = []
    for t in idxs:
        # proportional pose->frame alignment in case counts differ
        f = frames[min(int(round(t / max(T_pose - 1, 1) * (T_vid - 1))),
                       T_vid - 1)].copy()
        H, W = f.shape[:2]
        xy = pts27[t, :, :2] / SCALE * np.array([W, H])
        pres = pts27[t, :, 2] > 0
        for a, b in inward:
            if pres[a] and pres[b]:
                cv2.line(f, tuple(xy[a].astype(int)),
                         tuple(xy[b].astype(int)), (0, 255, 0), 3)
        for i, (x, y) in enumerate(xy):
            if pres[i]:
                color = (0, 0, 255) if i < 7 else (255, 0, 0) if i < 17 \
                    else (0, 255, 255)
                cv2.circle(f, (int(x), int(y)), 7, color, -1)
        scale = MAX_TILE_W / W
        f = cv2.resize(f, (MAX_TILE_W, int(H * scale)))
        cv2.putText(f, f"t={t}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        tiles.append(f)
    sheet = np.concatenate(tiles, axis=1)
    cv2.putText(sheet, f"{clip_id} [{split}] pose_T={T_pose} vid_T={T_vid}",
                (5, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    out = out_dir / f"{split}_{clip_id}.png"
    cv2.imwrite(str(out), sheet)
    print(f"  wrote {out} (pose {T_pose} frames, video {T_vid} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=4)
    parser.add_argument("--clip-ids", nargs="*", default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    out_dir = DOCS_DIR / "overlay_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(SPLITS_JSON.read_text())
    for split in args.splits:
        ids = args.clip_ids or splits[split][:args.n_clips]
        print(f"{split}: rendering {len(ids)} clips")
        for cid in ids:
            try:
                render_clip(cid, split, out_dir)
            except Exception as e:  # noqa: BLE001 — QA must not break the chain
                print(f"  {cid}: FAILED ({e})")


if __name__ == "__main__":
    main()
