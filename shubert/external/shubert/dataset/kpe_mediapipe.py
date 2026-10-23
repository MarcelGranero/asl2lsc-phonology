import argparse
import json
import os
import time
from pathlib import Path
import sys

from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import decord

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dataset.utils import load_file, is_string_in_file, dump_file_list  # noqa: E402
from data_config import BBOX_CLIPS_DIR, KPE_POSES_DIR, KPE_STATS_DIR, MODELS_DIR  # noqa: E402

# Global placeholders (set during init)
face_detector = None
hand_detector = None
mp_holistic = None


def detect_holistic(image: np.ndarray):
    global mp_holistic, face_detector, hand_detector

    results = mp_holistic.process(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    face_prediction = face_detector.detect(mp_image)
    hand_prediction = hand_detector.detect(mp_image)

    bounding_boxes = {}
    landmarks_data = {}

    # Face landmarks
    if face_prediction.face_landmarks:
        bounding_boxes["#face"] = len(face_prediction.face_landmarks)
        landmarks_data["face_landmarks"] = [
            [[l.x, l.y, l.z] for l in face] for face in face_prediction.face_landmarks
        ]
    else:
        bounding_boxes["#face"] = 0
        landmarks_data["face_landmarks"] = None

    # Hand landmarks
    if hand_prediction.hand_landmarks:
        bounding_boxes["#hands"] = len(hand_prediction.hand_landmarks)
        landmarks_data["hand_landmarks"] = [
            [[l.x, l.y, l.z] for l in hand] for hand in hand_prediction.hand_landmarks
        ]
    else:
        bounding_boxes["#hands"] = 0
        landmarks_data["hand_landmarks"] = None

    # Pose landmarks
    if results.pose_landmarks:
        bounding_boxes["#pose"] = 1
        landmarks_data["pose_landmarks"] = [
            [[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]
        ]
    else:
        bounding_boxes["#pose"] = 0
        landmarks_data["pose_landmarks"] = None

    return bounding_boxes, landmarks_data


def video_holistic(
    video_file: Path, problem_file: Path, pose_path: Path, stats_path: Path
):
    try:
        video = decord.VideoReader(str(video_file))
    except Exception as e:
        print(f"Failed to read video {video_file}: {e}", flush=True)
        with problem_file.open("a") as f:
            f.write(f"{video_file}\n")
        return

    stats = {}

    pose_file = pose_path / f"{video_file.stem}_pose.json"
    stats_file = stats_path / f"{video_file.stem}_stats.json"

    # Stream each frame's JSON straight to disk. Accumulating all landmarks
    # as Python objects needs 10-20x the JSON size in RAM (tens of GB for
    # long videos — the whole first array run died OUT_OF_MEMORY on it).
    # The concatenation below is byte-identical to json.dumps(result_dict).
    # Write to a .tmp and rename so a killed task never leaves a partial
    # pose file that would pass the exists() skip check on rerun. The tmp
    # name is PID-unique: two array tasks can race on the same video, and
    # a shared tmp name makes the loser crash on the rename.
    tmp_file = pose_file.with_name(f"{pose_file.name}.tmp{os.getpid()}")
    with tmp_file.open("w") as f:
        f.write("{")
        for i in range(len(video)):
            try:
                frame = video[i].asnumpy()
                bounding_boxes, landmarks = detect_holistic(frame)
                frame_json = json.dumps(landmarks)
                stats[i] = bounding_boxes
            except Exception as e:
                print(f"Frame {i} failed: {e}", flush=True)
                frame_json = "null"
            if i:
                f.write(", ")
            f.write(f'"{i}": {frame_json}')
        f.write("}")
    tmp_file.replace(pose_file)
    stats_file.write_text(json.dumps(stats))


def close_models():
    global face_detector, hand_detector, mp_holistic
    for det in (face_detector, hand_detector, mp_holistic):
        if det is not None:
            try:
                det.close()
            except Exception:
                pass
    face_detector = hand_detector = mp_holistic = None


def initialize_models(face_model: Path, hand_model: Path):
    global face_detector, hand_detector, mp_holistic

    face_opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=6,
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_opts)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        num_hands=6,
        min_hand_detection_confidence=0.05,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract holistic landmarks from video batches."
    )
    parser.add_argument("--index", type=int, help="Batch index.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--time-limit", type=int, default=257890, help="Time limit in seconds."
    )
    parser.add_argument(
        "--files-list",
        type=Path,
        default=BBOX_CLIPS_DIR / "bbox_clips.list.gz",
        help="Gzipped pickle list of files.",
    )
    parser.add_argument(
        "--pose-path",
        type=Path,
        default=KPE_POSES_DIR,
        help="Directory to save pose JSONs.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=KPE_STATS_DIR,
        help="Directory to save stats JSONs.",
    )
    parser.add_argument(
        "--problem-log",
        type=Path,
        default="kpe_problem.log",
        help="File to log problematic videos.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default="models/face_landmarker_v2_with_blendshapes.task",
        help="Face landmark model path.",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default="models/hand_landmarker.task",
        help="Hand landmark model path.",
    )
    parser.add_argument("--write-output-list", type=bool, default=True)
    args = parser.parse_args()

    # Prepare directories
    args.pose_path.mkdir(parents=True, exist_ok=True)
    args.stats_path.mkdir(parents=True, exist_ok=True)
    args.problem_log.touch(exist_ok=True)

    # Update list of files before processing
    source_dir = Path(args.files_list).parent
    if source_dir.exists():
        dump_file_list(source_dir)

    initialize_models(args.face_model, args.hand_model)
    video_files = load_file(args.files_list)

    video_batches = [
        video_files[i : i + args.batch_size]
        for i in range(0, len(video_files), args.batch_size)
    ]

    start_time = time.time()
    if args.index is None:
        # iterate over all batches
        batches = video_batches
    else:
        if args.index >= len(video_batches):
            print(
                f"Batch index {args.index} exceeds total batches {len(video_batches)}."
            )
            return
        batches = [video_batches[args.index]]

    skipped_videos = []
    for i, batch in enumerate(
        tqdm(batches, desc="Processing batches", disable=len(batches) <= 1)
    ):
        for video_file in tqdm(
            batch, desc=f"Processing videos in batch {i}", disable=len(batch) <= 1
        ):
            video_file = Path(video_file)
            elapsed = time.time() - start_time
            if elapsed > args.time_limit:
                print(f"Time limit reached after {elapsed:.2f}s.")
                break

            pose_file = args.pose_path / f"{video_file.stem}_pose.json"
            stats_file = args.stats_path / f"{video_file.stem}_stats.json"

            if pose_file.exists() and stats_file.exists():
                print(f"Skipping {video_file.name} (already processed).")
                skipped_videos.append(video_file)
                continue
            # if is_string_in_file(args.problem_log, str(video_file)):
            #     print(f"Skipping {video_file.name} (listed in problem log).")
            #     continue

            print(f"Processing {video_file.name}", flush=True)
            # Recreate the detectors per video: MediaPipe's working set
            # varies from ~2 to ~15 GB with video content and creeps up
            # (~1 GB per 40k-frame video); without a reset the growth
            # accumulates across the whole task and OOMs long tasks.
            close_models()
            initialize_models(args.face_model, args.hand_model)
            try:
                video_holistic(
                    video_file, args.problem_log, args.pose_path, args.stats_path
                )
            except Exception as e:
                # One broken video must not kill the rest of the batch.
                print(f"FAILED {video_file.name}: {e}", flush=True)
                with args.problem_log.open("a") as f:
                    f.write(f"{video_file}\t{e}\n")
    if args.write_output_list:
        dump_file_list(args.pose_path)

    # Print statistics
    print(f"Total time taken: {time.time() - start_time:.2f}s")
    print(f"Batch index: {args.index}")
    print(f"Total videos in batch: {len(batch)}")
    print(f"Total videos processed: {len(batch) - len(skipped_videos)}")
    print(f"Total videos skipped (already processed): {len(skipped_videos)}")
    n_processed = len(batch) - len(skipped_videos)
    if n_processed > 0:
        print(f"Time per processed video: {(time.time() - start_time) / n_processed:.2f}s")
    print(f"Total videos: {len(video_files)}")
    print(f"percentage of videos in batch: {len(batch) / len(video_files) * 100:.2f}%")
    print(f"Percentage of videos processed: {(len(batch) - len(skipped_videos)) / len(video_files) * 100:.2f}%")


if __name__ == "__main__":
    main()
