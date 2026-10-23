import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from decord import VideoReader
import numpy as np
from pathlib import Path
import argparse
import time
import sys
from tqdm import tqdm

# Dynamically add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
assert (repo_root / "dataset" / "utils.py").exists(), "utils.py not found in dataset/"
sys.path.insert(0, str(repo_root))

from dataset.utils import load_file, dump_file_list  # noqa: E402
from data_config import (  # noqa: E402
    POSE_FACE_CROPS_DIR,
    POSE_HAND_CROPS_DIR,
    DINOV2_FACE_EMBEDDINGS_DIR,
    DINOV2_HAND_EMBEDDINGS_DIR,
    MODELS_DIR,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_dino_finetuned(dino_path: Path) -> torch.nn.Module:
    # Load from local hub cache to avoid network access on compute nodes.
    # Falls back to the GitHub string if the local cache is absent.
    hub_cache = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if hub_cache.exists():
        model = torch.hub.load(
            str(hub_cache), "dinov2_vits14_reg", pretrained=False, source="local"
        )
    else:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=False
        )
    pretrained = torch.load(dino_path, map_location=device)
    new_state_dict = {}
    for key, value in pretrained["teacher"].items():
        if "dino_head" not in key:
            new_state_dict[key.replace("backbone.", "")] = value
    model.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.fromarray(frame)
    return transform(image)[:3]


@torch.no_grad()
def video_to_embeddings(
    video_path: Path, output_path: Path, model: torch.nn.Module, batch_size: int = 128
):
    try:
        vr = VideoReader(str(video_path), width=224, height=224)
    except Exception as e:
        print(f"Failed to load {video_path}: {e}")
        return

    total_frames = len(vr)
    embeddings = []

    for idx in range(0, total_frames, batch_size):
        frames = vr.get_batch(range(idx, min(idx + batch_size, total_frames))).asnumpy()
        batch = torch.stack([preprocess_frame(f) for f in frames]).to(device)
        emb = model(batch).cpu().numpy()
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")


def process_crop_type(
    files_list: Path,
    output_dir: Path,
    dino_path: Path,
    index: int | None,
    batch_size: int,
    time_limit: int,
):
    files = load_file(files_list)
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]
    output_dir.mkdir(parents=True, exist_ok=True)
    model = get_dino_finetuned(dino_path)

    start_time = time.time()
    batch_indices = range(len(batches)) if index is None else [index]

    for idx in batch_indices:
        if idx >= len(batches):
            print(f"Skipping index {idx}, only {len(batches)} batches.")
            continue
        print(
            f"Processing batch {idx + 1}/{len(batches)} with {len(batches[idx])} files."
        )

        for video_file in batches[idx]:
            video_path = Path(video_file)
            npy_path = output_dir / f"{video_path.stem}.npy"
            if npy_path.exists():
                continue
            if time.time() - start_time > time_limit:
                print(f"Time limit of {time_limit}s reached.")
                return
            video_to_embeddings(video_path, npy_path, model)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings for face/hand crops."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional batch index (runs all if not set).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for file processing."
    )
    parser.add_argument(
        "--time-limit", type=int, default=360000, help="Time limit in seconds."
    )

    parser.add_argument(
        "--face-crops-file-list",
        type=Path,
        default=POSE_FACE_CROPS_DIR / "pose_face_crops.list.gz",
    )
    parser.add_argument(
        "--hand-crops-file-list",
        type=Path,
        default=POSE_HAND_CROPS_DIR / "pose_hand_crops.list.gz",
    )
    parser.add_argument(
        "--models-folder", type=Path, default=MODELS_DIR / "SHuBERT_ckpts"
    )
    parser.add_argument(
        "--output-folder-face", type=Path, default=DINOV2_FACE_EMBEDDINGS_DIR
    )
    parser.add_argument(
        "--output-folder-hand", type=Path, default=DINOV2_HAND_EMBEDDINGS_DIR
    )
    def _str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("yes", "true", "t", "1")

    parser.add_argument("--write-output-list", type=_str2bool, default=True)
    parser.add_argument("--face-hand", type=str, default="both", choices=["face", "hand", "both"])

    args = parser.parse_args()

    if not args.models_folder.exists():
        print(f"Error: Models folder {args.models_folder} does not exist.")
        return

    if args.face_hand == "face" or args.face_hand == "both":
        print("\n=== Processing Face Crops ===")
        face_dino = args.models_folder / "dinov2face.pth"
        process_crop_type(
            files_list=args.face_crops_file_list,
            output_dir=args.output_folder_face,
            dino_path=face_dino,
            index=args.index,
            batch_size=args.batch_size,
            time_limit=args.time_limit,
        )
        if args.write_output_list:
            dump_file_list(args.output_folder_face)

    if args.face_hand == "hand" or args.face_hand == "both":
        print("\n=== Processing Hand Crops ===")
        hand_dino = args.models_folder / "dinov2hand.pth"
        process_crop_type(
            files_list=args.hand_crops_file_list,
            output_dir=args.output_folder_hand,
            dino_path=hand_dino,
            index=args.index,
            batch_size=args.batch_size,
            time_limit=args.time_limit,
        )
        if args.write_output_list:
            dump_file_list(args.output_folder_hand)


if __name__ == "__main__":
    main()
