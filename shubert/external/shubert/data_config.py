"""
Central configuration for all data directories used in the SHuBERT pipeline.

This file defines the paths to all data directories. By default, it points to
the shared data storage location. You can override these paths by:

1. Setting environment variables (e.g., SHUBERT_DATA_ROOT)
2. Creating a local_data_config.py file that overrides these values
3. Passing command-line arguments to individual scripts

Directory structure:
- raw_vids: Original YouTube-ASL videos
- raw_clips: Clips extracted from raw videos
- bbox_clips: Cropped clips with signer bounding boxes
- kpe_poses: MediaPipe / DWPose pose estimation results (JSON)
- kpe_stats: Statistics from pose estimation
- pose_face_crops: Face crops based on pose landmarks (full face in v2)
- pose_hand_crops: Hand crops based on pose landmarks
- body_features: Normalized body pose features (npy)
- dinov2_face_embeddings: DINOv2 face embeddings (v1)
- dinov2_hand_embeddings: DINOv2 hand embeddings (v1)
- dinov3_face_embeddings: DINOv3 face embeddings (v2)
- dinov3_hand_embeddings: DINOv3 hand embeddings (v2)
- dinov3_pretraining_face: Sampled face frames for DINOv3 continued pre-training
- dinov3_pretraining_hands: Sampled hand frames for DINOv3 continued pre-training
- shubert_features: SHuBERT model features
"""

import os
from pathlib import Path

# Base data directory - can be overridden by environment variable
DATA_ROOT = Path(os.environ.get(
    "SHUBERT_DATA_ROOT",
    "data"
))

# YouTube-ASL data subdirectory
YOUTUBE_ASL_DATA = DATA_ROOT / "SHuBERT_YouTube_ASL"

# Define all data directories
RAW_VIDS_DIR = DATA_ROOT / "YouTube_ASL"
RAW_CLIPS_DIR = YOUTUBE_ASL_DATA / "raw_clips"
BBOX_CLIPS_DIR = YOUTUBE_ASL_DATA / "bbox_clips"
KPE_POSES_DIR = YOUTUBE_ASL_DATA / "kpe_poses"
KPE_STATS_DIR = YOUTUBE_ASL_DATA / "kpe_stats"
POSE_FACE_CROPS_DIR = YOUTUBE_ASL_DATA / "pose_face_crops"
POSE_HAND_CROPS_DIR = YOUTUBE_ASL_DATA / "pose_hand_crops"
BODY_FEATURES_DIR = YOUTUBE_ASL_DATA / "body_features"
DINOV2_FACE_EMBEDDINGS_DIR = YOUTUBE_ASL_DATA / "dinov2_face_embeddings"
DINOV2_HAND_EMBEDDINGS_DIR = YOUTUBE_ASL_DATA / "dinov2_hand_embeddings"
SHUBERT_FEATURES_DIR = YOUTUBE_ASL_DATA / "shubert_features"

# SHuBERT v2: DINOv3 embeddings and pre-training image directories
DINOV3_FACE_EMBEDDINGS_DIR = YOUTUBE_ASL_DATA / "dinov3_face_embeddings"
DINOV3_HAND_EMBEDDINGS_DIR = YOUTUBE_ASL_DATA / "dinov3_hand_embeddings"
DINOV3_PRETRAINING_FACE_DIR = YOUTUBE_ASL_DATA / "dinov3_pretraining_face"
DINOV3_PRETRAINING_HANDS_DIR = YOUTUBE_ASL_DATA / "dinov3_pretraining_hands"

# Model directory (within repo)
REPO_ROOT = Path(__file__).parent
MODELS_DIR = REPO_ROOT / "models"

# Manifest files directory (within repo)
MANIFEST_DIR = REPO_ROOT / "manifest_files"

# Allow local overrides
try:
    from local_data_config import *  # noqa: F401, F403
except ImportError:
    pass


def get_data_path(key: str) -> Path:
    """
    Get a data path by key name.

    Args:
        key: One of 'raw_vids', 'raw_clips', 'bbox_clips', 'kpe_poses',
             'kpe_stats', 'pose_face_crops', 'pose_hand_crops',
             'body_features', 'dinov2_face_embeddings',
             'dinov2_hand_embeddings', 'shubert_features', 'models', 'manifest'

    Returns:
        Path object for the requested directory
    """
    paths = {
        'raw_vids': RAW_VIDS_DIR,
        'raw_clips': RAW_CLIPS_DIR,
        'bbox_clips': BBOX_CLIPS_DIR,
        'kpe_poses': KPE_POSES_DIR,
        'kpe_stats': KPE_STATS_DIR,
        'pose_face_crops': POSE_FACE_CROPS_DIR,
        'pose_hand_crops': POSE_HAND_CROPS_DIR,
        'body_features': BODY_FEATURES_DIR,
        'dinov2_face_embeddings': DINOV2_FACE_EMBEDDINGS_DIR,
        'dinov2_hand_embeddings': DINOV2_HAND_EMBEDDINGS_DIR,
        'shubert_features': SHUBERT_FEATURES_DIR,
        'dinov3_face_embeddings': DINOV3_FACE_EMBEDDINGS_DIR,
        'dinov3_hand_embeddings': DINOV3_HAND_EMBEDDINGS_DIR,
        'dinov3_pretraining_face': DINOV3_PRETRAINING_FACE_DIR,
        'dinov3_pretraining_hands': DINOV3_PRETRAINING_HANDS_DIR,
        'models': MODELS_DIR,
        'manifest': MANIFEST_DIR,
    }

    if key not in paths:
        raise ValueError(f"Unknown data path key: {key}. Valid keys: {list(paths.keys())}")

    return paths[key]


def print_config():
    """Print the current data configuration."""
    print("SHuBERT Data Configuration")
    print("=" * 60)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"YOUTUBE_ASL_DATA: {YOUTUBE_ASL_DATA}")
    print()
    print("Data Directories:")
    print(f"  raw_vids: {RAW_VIDS_DIR}")
    print(f"  raw_clips: {RAW_CLIPS_DIR}")
    print(f"  bbox_clips: {BBOX_CLIPS_DIR}")
    print(f"  kpe_poses: {KPE_POSES_DIR}")
    print(f"  kpe_stats: {KPE_STATS_DIR}")
    print(f"  pose_face_crops: {POSE_FACE_CROPS_DIR}")
    print(f"  pose_hand_crops: {POSE_HAND_CROPS_DIR}")
    print(f"  body_features: {BODY_FEATURES_DIR}")
    print(f"  dinov2_face_embeddings: {DINOV2_FACE_EMBEDDINGS_DIR}")
    print(f"  dinov2_hand_embeddings: {DINOV2_HAND_EMBEDDINGS_DIR}")
    print(f"  shubert_features: {SHUBERT_FEATURES_DIR}")
    print()
    print("Repository Directories:")
    print(f"  models: {MODELS_DIR}")
    print(f"  manifest: {MANIFEST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
