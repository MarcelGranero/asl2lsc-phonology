import cv2
import numpy as np
import time
import json
import decord
import argparse
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dataset.utils import (  # noqa: E402
    load_file,
    is_string_in_file,
    crop_frame,
    resize_frame,
    get_bounding_box,
    adjust_bounding_box,
    dump_file_list,
)
from data_config import BBOX_CLIPS_DIR, KPE_POSES_DIR, POSE_HAND_CROPS_DIR  # noqa: E402


def select_hands(pose_landmarks, hand_landmarks, image_shape):
    """Select left/right hand landmarks by proximity to wrist points."""
    if hand_landmarks is None:
        return None, None

    left_wrist = pose_landmarks[15]
    right_wrist = pose_landmarks[16]

    wrist_from_hand = [h[0] for h in hand_landmarks]

    ih, iw, _ = image_shape

    def find_closest_wrist(wrist_point):
        if wrist_point is None:
            return None
        closest = None
        min_dist = float("inf")
        for hand_lm in hand_landmarks:
            dist = np.linalg.norm(np.array(wrist_point[:2]) - np.array(hand_lm[0][:2]))
            if dist < min_dist:
                min_dist = dist
                closest = hand_lm
        return closest if min_dist < 0.1 else None

    left_hand = find_closest_wrist(left_wrist)
    right_hand = find_closest_wrist(right_wrist)

    return left_hand, right_hand


def video_holistic(video_file, hand_dir, problem_file, pose_dir):
    """Process a video to crop and save left/right hand clips."""
    video = decord.VideoReader(video_file)
    fps = video.get_avg_fps()

    basename = Path(video_file).stem
    clip_hand1_path = Path(hand_dir) / f"{basename}_hand1.mp4"
    clip_hand2_path = Path(hand_dir) / f"{basename}_hand2.mp4"
    landmark_json_path = Path(pose_dir) / f"{basename}_pose.json"

    for p in [clip_hand1_path, clip_hand2_path]:
        if p.exists():
            p.unlink()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_hand1 = cv2.VideoWriter(str(clip_hand1_path), fourcc, fps, (224, 224))
    out_hand2 = cv2.VideoWriter(str(clip_hand2_path), fourcc, fps, (224, 224))

    if not landmark_json_path.exists():
        print(f"Landmark file {landmark_json_path} does not exist. Skipping {video_file}.")
        return
    with open(landmark_json_path, "r") as f:
        result_dict = json.load(f)

    prev_hand1_frame, prev_hand2_frame, prev_result_dict = None, None, None

    for i in range(len(video)):
        frame = video[i].asnumpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_results = result_dict.get(str(i))
        if frame_results is None:
            if prev_result_dict is None:
                continue
            frame_results = prev_result_dict
        else:
            prev_result_dict = frame_results

        pose_landmarks = frame_results.get("pose_landmarks")
        hand_landmarks = frame_results.get("hand_landmarks")

        left_hand, right_hand = select_hands(
            pose_landmarks[0] if pose_landmarks else None,
            hand_landmarks,
            frame_rgb.shape,
        )

        def write_hand_frame(hand, prev_frame, writer):
            if hand is not None:
                box = get_bounding_box(hand, frame_rgb.shape, scale_factor=1.5)
                box = adjust_bounding_box(box, frame_rgb.shape)
                cropped = crop_frame(frame_rgb, box)
                resized = resize_frame(cropped, (224, 224))
                writer.write(resized)
                return resized
            elif prev_frame is not None:
                writer.write(prev_frame)
                return prev_frame
            else:
                blank = np.zeros((224, 224, 3), dtype=np.uint8)
                writer.write(blank)
                return blank

        prev_hand1_frame = write_hand_frame(left_hand, prev_hand1_frame, out_hand1)
        prev_hand2_frame = write_hand_frame(right_hand, prev_hand2_frame, out_hand2)

    out_hand1.release()
    out_hand2.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, help="Batch index to process")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--time_limit", type=int, default=257890)
    parser.add_argument(
        "--files_list", type=Path, default=BBOX_CLIPS_DIR / "bbox_clips.list.gz"
    )
    parser.add_argument("--problem_file", type=Path, default="crop_hands_problems.log")
    parser.add_argument(
        "--pose_dir",
        type=Path,
        default=KPE_POSES_DIR,
        help="Where to find the outputs from kpe_mediapipe.py",
    )
    parser.add_argument(
        "--hand_dir",
        type=Path,
        default=POSE_HAND_CROPS_DIR,
        help="Output folder for hand crops",
    )
    parser.add_argument("--write-output-list", type=bool, default=True)
    args = parser.parse_args()

    start_time = time.time()

    files = load_file(args.files_list)
    Path(args.hand_dir).mkdir(parents=True, exist_ok=True)
    Path(args.problem_file).touch(exist_ok=True)

    batches = [
        files[i : i + args.batch_size] for i in range(0, len(files), args.batch_size)
    ]

    batch_indices = [args.index] if args.index is not None else range(len(batches))

    for batch_idx in batch_indices:
        print(f"Processing batch {batch_idx + 1}/{len(batches)}")
        batch = batches[batch_idx]
        for video_file in batch:
            if time.time() - start_time > args.time_limit:
                print("Time limit reached. Exiting.")
                return

            basename = Path(video_file).stem
            output_clip = Path(args.hand_dir) / f"{basename}_hand2.mp4"

            if output_clip.exists():
                print(f"Clip {output_clip} already exists. Skipping...")
                continue
            # if is_string_in_file(Path(args.problem_file), video_file):
            #     print(f"Video {video_file} in problem file {Path(args.problem_file)}. Skipping...")
            #     continue
            pose_file = Path(args.pose_dir) / f"{basename}_pose.json"
            if not pose_file.exists():
                print(f"Could not find {pose_file}. Skipping...")
                continue

            try:
                video_holistic(
                    video_file, args.hand_dir, args.problem_file, args.pose_dir
                )
            except Exception as e:
                print(f"Failed on {video_file}: {e}")
                with open(args.problem_file, "a") as f:
                    f.write(video_file + "\n")

    if args.write_output_list:
        dump_file_list(Path(args.hand_dir))


if __name__ == "__main__":
    main()
