import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys


# Dynamically add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
assert (repo_root / "dataset" / "utils.py").exists(), "utils.py not found in dataset/"
sys.path.insert(0, str(repo_root))

from dataset.utils import load_file, dump_file_list  # noqa: E402
from data_config import KPE_POSES_DIR, BODY_FEATURES_DIR  # noqa: E402


def normalize_pose_keypoints(pose_landmarks: list[list[float]]) -> list[np.ndarray]:
    left_shoulder = np.array(pose_landmarks[11][:2])
    right_shoulder = np.array(pose_landmarks[12][:2])
    left_eye = np.array(pose_landmarks[2][:2])
    nose = np.array(pose_landmarks[0][:2])

    head_unit = np.linalg.norm(right_shoulder - left_shoulder) / 2
    width, height = 6 * head_unit, 7 * head_unit

    top = left_eye[1] - 0.5 * head_unit
    left = nose[0] - width / 2

    translation = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
    scale = np.array([[1 / width, 0, 0], [0, 1 / height, 0], [0, 0, 1]])
    shift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    transform = shift @ scale @ translation

    normalized = []
    for landmark in pose_landmarks:
        keypoint = np.array([landmark[0], landmark[1], 1])
        normed = transform @ keypoint
        normalized.append(normed[:2])

    return normalized


def keypoints_to_numpy(pose_file: Path, output_dir: Path) -> None:
    try:
        with pose_file.open("r") as f:
            result_dict = json.load(f)
    except Exception as e:
        print(f"Error reading {pose_file}: {e}")
        return

    prev_pose = None
    video_keypoints = []
    indices = [0, 11, 12, 13, 14, 15, 16]

    for i in range(len(result_dict)):
        frame_data = result_dict.get(str(i))

        if frame_data is None:
            frame_pose = prev_pose if prev_pose is not None else np.full(14, -9999.0)
        elif frame_data.get("pose_landmarks"):
            landmarks = frame_data["pose_landmarks"][0][:25]
            normalized = normalize_pose_keypoints(landmarks)
            selected = np.array([normalized[j] for j in indices]).flatten()
            frame_pose = selected
            prev_pose = selected
        elif prev_pose is not None:
            frame_pose = prev_pose
        else:
            frame_pose = np.full(14, -9999.0)

        video_keypoints.append(frame_pose)

    video_keypoints = np.array(video_keypoints)
    video_keypoints[:, :2] = -9999.0  # Zero out first landmark (e.g., nose)

    npy_path = output_dir / f"{pose_file.stem}.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, video_keypoints)


def main():
    parser = argparse.ArgumentParser(
        description="Extract normalized body keypoints from pose JSONs."
    )
    parser.add_argument(
        "--index", type=int, default=None, help="Batch index (optional)"
    )
    parser.add_argument(
        "--files-list",
        type=Path,
        default=KPE_POSES_DIR / "kpe_poses.list.gz",
        help="Gzipped pickle file list of pose JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BODY_FEATURES_DIR,
        help="Directory to save output .npy files",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--time-limit", type=int, default=3600, help="Time limit in seconds"
    )
    parser.add_argument(
        "--write-output-list",
        type=bool,
        default=True,
        help="Whether to write output .list.gz after processing",
    )

    args = parser.parse_args()

    # # Update list of files before processing
    # source_dir = Path(args.files_list).parent
    # if source_dir.exists():
    #     dump_file_list(source_dir)

    files = load_file(args.files_list)
    batches = [
        files[i : i + args.batch_size] for i in range(0, len(files), args.batch_size)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.index is not None:
        batch_indices = [args.index]
    else:
        batch_indices = range(len(batches))

    for batch_idx in batch_indices:
        if batch_idx >= len(batches):
            print(
                f"Batch index {batch_idx} exceeds total batches {len(batches)}. Skipping."
            )
            continue

        print(
            f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batches[batch_idx])} files."
        )

        for pose_file_str in batches[batch_idx]:
            pose_file = Path(pose_file_str)
            npy_path = args.output_dir / f"{pose_file.stem}.npy"

            if npy_path.exists():
                print(f"Skipping {pose_file.name}, output already exists.")
                continue

            if time.time() - start_time > args.time_limit:
                print(f"Time limit {args.time_limit}s reached. Stopping.")
                return

            keypoints_to_numpy(pose_file, args.output_dir)

    if args.write_output_list:
        dump_file_list(args.output_dir)


if __name__ == "__main__":
    main()
