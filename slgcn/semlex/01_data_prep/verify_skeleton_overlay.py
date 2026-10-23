"""Visual QA: render the 27 selected joints over real video frames.

Loads a few val/test clips (those splits have videos on disk), converts their
553-point poses through our wholebody mapping, applies upstream's own
selected_joints['27'], and draws points + skeleton edges on sampled frames.
Output PNG contact sheets under docs/overlay_qa/ — eyeball before training.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_config import DOCS_DIR, POSES_DIR, SL_GCN_DIR, VIDEOS_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_553_to_wholebody133 import SCALE, convert

sys.path.insert(0, str(SL_GCN_DIR / "data_gen"))
from sign_gendata import selected_joints  # noqa: E402

# sign_27.py edge list, shifted to 0..26 node ids (its nodes run 5..31)
sys.path.insert(0, str(SL_GCN_DIR))
from graph.sign_27 import inward  # noqa: E402

SEL27 = selected_joints["27"]


def render_clip(video_id: str, split: str, out_dir: Path, n_frames: int = 6):
    video = VIDEOS_DIR[split] / f"{video_id}.webm"
    pose = np.load(POSES_DIR[split] / f"{video_id}.npy")
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
        print(f"  {video_id}: no frames decoded, skipping")
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
                         tuple(xy[b].astype(int)), (0, 255, 0), 1)
        for i, (x, y) in enumerate(xy):
            if pres[i]:
                color = (0, 0, 255) if i < 7 else (255, 0, 0) if i < 17 \
                    else (0, 255, 255)
                cv2.circle(f, (int(x), int(y)), 3, color, -1)
        cv2.putText(f, f"t={t}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        tiles.append(f)
    sheet = np.concatenate(tiles, axis=1)
    cv2.putText(sheet, f"{video_id} [{split}] pose_T={T_pose} vid_T={T_vid}",
                (5, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    out = out_dir / f"{split}_{video_id}.png"
    cv2.imwrite(str(out), sheet)
    print(f"  wrote {out} (pose {T_pose} frames, video {T_vid} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=4)
    parser.add_argument("--video-ids", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = DOCS_DIR / "overlay_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["val", "test"]:
        if args.video_ids:
            ids = [v for v in args.video_ids
                   if (VIDEOS_DIR[split] / f"{v}.webm").exists()]
        else:
            vids = sorted(VIDEOS_DIR[split].glob("*.webm"))
            ids = []
            for v in vids:
                if (POSES_DIR[split] / f"{v.stem}.npy").exists():
                    ids.append(v.stem)
                if len(ids) >= args.n_clips:
                    break
        print(f"{split}: rendering {len(ids)} clips")
        for vid in ids:
            render_clip(vid, split, out_dir)


if __name__ == "__main__":
    main()
