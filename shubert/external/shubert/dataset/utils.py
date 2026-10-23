from pathlib import Path
import logging
import numpy as np
import cv2
import gzip
import pickle

logger = logging.getLogger(__name__)


def load_file(filename: Path):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def is_string_in_file(file_path: Path, target_string: str) -> bool:
    """
    Check if a given string exists in any line of the file.

    Args:
        file_path (Path): Path to the file.
        target_string (str): String to search for.

    Returns:
        bool: True if the string is found, False otherwise.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return any(target_string in line for line in f)
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
    return False


def dump_file_list(input_dir: Path, output_file: Path | None = None) -> None:
    """
    Collect absolute file paths from input_dir (excluding .list.gz files)
    and write them as a pickled gzip list to output_file.

    Args:
        input_dir (Path): Directory to scan for files.
        output_file (Path, optional): Output .list.gz file path.
            Defaults to <input_dir>/<input_dir.name>.list.gz.
    """
    input_files_list = [
        str(p.resolve())
        for p in input_dir.iterdir()
        if p.is_file() and not p.name.endswith(".list.gz")
    ]

    if output_file is None:
        output_file = input_dir / f"{input_dir.name}.list.gz"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_file, "wb") as f:
        pickle.dump(input_files_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Written {len(input_files_list)} file paths to {output_file}")


def crop_frame(image, bounding_box):
    x, y, w, h = bounding_box
    cropped_frame = image[y : y + h, x : x + w]
    return cropped_frame


def resize_frame(frame, frame_size):
    if frame is not None and frame.size > 0:
        return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    else:
        return None


def get_bounding_box(landmarks, image_shape, scale_factor=1.2):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    xb, yb, wb, hb = cv2.boundingRect(landmarks_px)
    box_size = max(wb, hb)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size

    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding

    return x, y, w, h


def adjust_bounding_box(bounding_box, image_shape):
    x, y, w, h = bounding_box
    ih, iw, _ = image_shape

    # Adjust x-coordinate if the bounding box extends beyond the image's right edge
    if x + w > iw:
        x = iw - w

    # Adjust y-coordinate if the bounding box extends beyond the image's bottom edge
    if y + h > ih:
        y = ih - h

    # Ensure bounding box's x and y coordinates are not negative
    x = max(x, 0)
    y = max(y, 0)

    return x, y, w, h


def get_centered_box(landmarks, image_shape, box_size, scale_factor=1.5):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size
    return x, y, w, h


def is_center_inside_frame(landmarks, image_shape):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    return 0 <= center_x <= iw and 0 <= center_y <= ih


def isl_wrist_below_elbow(pose_landmarks):
    left_elbow = pose_landmarks[13]
    left_wrist = pose_landmarks[15]
    return left_wrist[1] > left_elbow[1]


def isr_wrist_below_elbow(pose_landmarks):
    right_elbow = pose_landmarks[14]
    right_wrist = pose_landmarks[16]
    return right_wrist[1] > right_elbow[1]


def shift_bounding_box(bounding_box, direction, magnitude):
    x, y, w, h = bounding_box
    shift_x = int(direction[0] * magnitude)
    shift_y = int(direction[1] * magnitude)
    shifted_box = (x + shift_x, y + shift_y, w, h)
    return shifted_box


def get_hand_direction(elbow_landmark, wrist_landmark):
    direction = np.array([wrist_landmark[0], wrist_landmark[1]]) - np.array(
        [elbow_landmark[0], elbow_landmark[1]]
    )
    direction = direction / np.linalg.norm(direction)
    return direction
