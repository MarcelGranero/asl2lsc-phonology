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

from dataset.utils import load_file, is_string_in_file, resize_frame, dump_file_list  # noqa: E402
from data_config import BBOX_CLIPS_DIR, KPE_POSES_DIR, POSE_FACE_CROPS_DIR  # noqa: E402


def select_face(pose_landmarks, face_landmarks):
    """Select the closest face to the pose nose landmark."""
    nose_from_pose = pose_landmarks[0]
    nose_from_face = [face[0] for face in face_landmarks]

    distances = [
        np.linalg.norm(np.array(nose_from_pose) - np.array(nose))
        for nose in nose_from_face
    ]
    closest_idx = np.argmin(distances)
    return face_landmarks[closest_idx]


def calculate_bounding_box(landmarks, indices, image_shape):
    x = [landmarks[i][0] for i in indices]
    y = [landmarks[i][1] for i in indices]
    h, w = image_shape[:2]
    left, right, top, bottom = min(x), max(x), min(y), max(y)
    return int(left * w), int(top * h), int(right * w), int(bottom * h)


def cues_on_grey_background(image, facial_landmarks, keep_entire_face=False):
    h, w, _ = image.shape

    # 1) Choose which regions to crop based on keep_entire_face flag
    if not keep_entire_face:
        # original eye + mouth indices
        indices = {
            "left_eye": [69, 168, 156, 118, 54],
            "right_eye": [168, 299, 347, 336, 301],
            "mouth": [164, 212, 432, 18],
        }

        boxes = {
            key: calculate_bounding_box(facial_landmarks, idxs, image.shape)
            for key, idxs in indices.items()
        }

        min_x = min(box[0] for box in boxes.values())
        min_y = min(box[1] for box in boxes.values())
        max_x = max(box[2] for box in boxes.values())
        max_y = max(box[3] for box in boxes.values())

        pad = 10
        min_x = max(0, min_x - pad)
        min_y = max(0, min_y - pad)
        max_x = min(w, max_x + pad)
        max_y = min(h, max_y + pad)

    else:
        # keep the entire face: use *all* landmarks
        pts = np.array(facial_landmarks)  # shape (N, 2)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        # coerce into ints
        min_x, min_y, max_x, max_y = map(int, (min_x, min_y, max_x, max_y))
        # and treat this as a single “box”
        boxes = {"face": (min_x, min_y, max_x, max_y)}

    width, height = max_x - min_x, max_y - min_y
    side_length = max(width, height)

    # Square adjustment
    if width < side_length:
        diff = side_length - width
        min_x = max(0, min_x - diff // 2)
        max_x = min(w, max_x + diff // 2)
    if height < side_length:
        diff = side_length - height
        min_y = max(0, min_y - diff // 2)
        max_y = min(h, max_y + diff // 2)

    grey = np.ones((side_length, side_length, 3), dtype=np.uint8) * 128

    def crop_and_paste(src_box):
        x1, y1, x2, y2 = src_box
        cropped = image[y1:y2, x1:x2]
        dx, dy = x1 - min_x, y1 - min_y
        grey[dy : dy + cropped.shape[0], dx : dx + cropped.shape[1]] = cropped


    if not keep_entire_face:
        # paste each feature patch
        for box in boxes.values():
            crop_and_paste(box)
    else:
        # paste the single face patch
        crop_and_paste(boxes["face"])

    return grey


def video_holistic(video_file, face_dir, problem_file, pose_dir, keep_entire_face=False):
    print(f"Running holistic on {video_file}")
    video = decord.VideoReader(video_file)
    fps = video.get_avg_fps()

    basename = Path(video_file).stem
    clip_face_path = Path(face_dir) / f"{basename}_face.mp4"
    landmark_path = Path(pose_dir) / f"{basename}_pose.json"

    clip_face_path.parent.mkdir(parents=True, exist_ok=True)
    if clip_face_path.exists():
        clip_face_path.unlink()

    out = cv2.VideoWriter(
        str(clip_face_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (224, 224)
    )

    if not landmark_path.exists():
        print(f"Landmark file {landmark_path} does not exist. Skipping {video_file}.")
        return
    result_dict = json.loads(Path(landmark_path).read_text())
    prev_frame = None
    prev_pose = None

    for i in range(len(video)):
        frame = video[i].asnumpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_data = result_dict.get(str(i))

        if pose_data is None:
            if prev_pose is None:
                continue
            pose_data = prev_pose
        else:
            prev_pose = pose_data

        if pose_data.get("pose_landmarks") is None:
            if prev_frame is not None:
                out.write(prev_frame)
            else:
                out.write(np.zeros((224, 224, 3), dtype=np.uint8))
            continue

        if pose_data.get("face_landmarks") is not None:
            face = select_face(
                pose_data["pose_landmarks"][0], pose_data["face_landmarks"]
            )
            face_crop = cues_on_grey_background(
                frame_rgb, face, keep_entire_face=keep_entire_face
                )
            face_crop = resize_frame(face_crop, (224, 224))
            out.write(face_crop)
            prev_frame = face_crop
        else:
            out.write(
                prev_frame
                if prev_frame is not None
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )

    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=int, help="Batch index (if unset, process all batches)"
    )
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--time_limit", type=int, default=257890)
    parser.add_argument(
        "--files_list", type=str, default=str(BBOX_CLIPS_DIR / "bbox_clips.list.gz")
    )
    parser.add_argument(
        "--problem_file_path", type=str, default="crop_face_problems.log"
    )
    parser.add_argument(
        "--pose_dir",
        type=Path,
        default=KPE_POSES_DIR,
        help="Where to find the outputs from kpe_mediapipe.py",
    )
    parser.add_argument("--face_dir", type=Path, default=POSE_FACE_CROPS_DIR)
    parser.add_argument("--write_output_list", type=bool, default=True)
    parser.add_argument(
        "--keep_entire_face",
        action="store_true",
        help="If set, paste the whole face; otherwise only eyes+mouth."
    )

    args = parser.parse_args()
    start_time = time.time()

    files = load_file(args.files_list)
    Path(args.face_dir).mkdir(parents=True, exist_ok=True)
    Path(args.problem_file_path).touch(exist_ok=True)

    batches = [
        files[i : i + args.batch_size] for i in range(0, len(files), args.batch_size)
    ]
    batch_indices = [args.index] if args.index is not None else range(len(batches))

    for batch_idx in batch_indices:
        print(f"Processing batch {batch_idx + 1}/{len(batches)}")
        for video_file in batches[batch_idx]:
            print(f"Processing {video_file}")
            if time.time() - start_time > args.time_limit:
                print("Time limit reached, exiting.")
                return

            basename = Path(video_file).stem
            clip_path = Path(args.face_dir) / f"{basename}_face.mp4"

            if clip_path.exists():
                print(f"Clip {clip_path} already exists. Skipping...")
                continue
            # if is_string_in_file(Path(args.problem_file_path), video_file):
            #     print(f"{video_file} is a problem file, skipping")
            #     continue
            pose_file = Path(args.pose_dir) / f"{basename}_pose.json"
            if not pose_file.exists():
                print(f"Could not find {pose_file}. Skipping...")
                continue

            try:
                video_holistic(
                    video_file, args.face_dir, args.problem_file_path, args.pose_dir, keep_entire_face=args.keep_entire_face,
                )
            except Exception as e:
                print(f"Error on {video_file}: {e}")
                with open(args.problem_file_path, "a") as pf:
                    pf.write(video_file + "\n")

    if args.write_output_list:
        dump_file_list(Path(args.face_dir))


if __name__ == "__main__":
    main()
