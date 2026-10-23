"""Central path/config constants for the SL-GCN PopSign leaf.

Every script imports from here; personal overrides go in local_data_config.py
(gitignored).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Heavy-data locations resolve through ASL2LSC_* environment variables
# (registry shared with the sibling leaves). On the cluster the
# _SITE_DEFAULTS below keep everything working with no env set; the public
# release ships with the block emptied, so an unset variable fails loudly.
# --- site defaults (optional per-machine values) ---
_SITE_DEFAULTS = {}
# --- end site defaults ---


def _path_from_env(var):
    value = os.environ.get(var) or _SITE_DEFAULTS.get(var)
    if not value:
        raise RuntimeError(
            f"Environment variable {var} is not set. Point it at the "
            f"corresponding data root (see the repository README, "
            f"'Data layout')."
        )
    return Path(value)

# --- PopSign v1.0 source data (read-only) ---
# Two parallel resources with 1:1 matching basenames
# (e.g. "4a.6001-TV-2022_12_14_14_59_01.480-0"):
#   * landmarks: per-clip parquet in the Kaggle ASL-Signs long format
#     (frame, row_id, type, landmark_index, x, y, z), 543 landmarks/frame =
#     468 face + 33 pose (BlazePose) + 21 left hand + 21 right hand,
#     normalised image coords, NaN where undetected.
#   * videos: mp4 (H.264), foldered by class. NOT uniform: 1944x2592 portrait
#     (84%) or 1080x1920 (16%) -- and the split mix is extreme, val being 100%
#     1944x2592 while test is 63% 1080x1920; frame rate is ~30 fps (81%) or a
#     GENUINE ~120 fps (19%, frames and duration agree). 2 mp4s are truncated
#     (unreadable) -- distinct from the 2 corrupt parquet in
#     results/convert_failures.json.
# We read the LANDMARKS for training data and the VIDEOS only for overlay QA.
POPSIGN_ROOT = _path_from_env("ASL2LSC_POPSIGN_PREPROC")
LANDMARKS_DIR = POPSIGN_ROOT / "popsign_kaggle_style_landmarks"  # flat, 174,243
VIDEOS_ROOT = POPSIGN_ROOT / "popsign_v1_0_videos" / "game"      # <split>/<class>/
SPLIT_DIRS = {s: VIDEOS_ROOT / s for s in ("train", "val", "test")}

# Per-clip ASL-LEX phonological labels (handshape + 5 subhandshape fields),
# in merged_videos_dataset_batch*.json under <split>/. These files ALSO embed
# the landmarks, which we ignore — the parquet is the pose source.
# Coverage is ~82% of the packed clips (see results/feature_vocab.json).
POPSIGN_JSON_ROOT = _path_from_env("ASL2LSC_POPSIGN_JSON")

# Canonical 250-class index from the official Kaggle release. Used ONLY to fix
# label ordering; the class SET is derived from the split directories and the
# two are cross-checked in derive_vocab.py.
KAGGLE_SIGN_MAP_JSON = (
    _path_from_env("ASL2LSC_KAGGLE_ASL_SIGNS")
    / "sign_to_prediction_index_map.json"
)

# --- Processed outputs (heavy artifacts, never in git) ---
PROCESSED_DIR = _path_from_env("ASL2LSC_SLGCN_POPSIGN_DATA")
# Per-clip (T, 133, 3) float32 npys in COCO-WholeBody layout, named
# <clip_id>_color.mp4.npy so upstream sign_gendata.py consumes them unchanged.
WHOLEBODY_NPY_DIR = PROCESSED_DIR / "wholebody133"
# Per-split "name,label_int" CSVs (no header) — upstream sign_gendata contract.
LABELS_DIR = PROCESSED_DIR / "labels"
# Output of upstream sign_gendata.py: {split}_data_joint.npy + {split}_label.pkl
GENDATA_DIR = PROCESSED_DIR / "gendata_27"

# --- Repo-relative outputs (small artifacts, committed) ---
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"
MANIFEST_CSV = RESULTS_DIR / "manifest.csv"
MANIFEST_REPORT_JSON = RESULTS_DIR / "manifest_report.json"
VOCAB_JSON = RESULTS_DIR / "vocab.json"
SPLITS_JSON = RESULTS_DIR / "splits.json"
FRAME_STATS_JSON = RESULTS_DIR / "frame_stats.json"
# Clips whose source parquet could not be read (PopSign ships at least one
# truncated file). Written by convert_poses.py, consumed by
# build_splits.py --exclude-failures so they never reach upstream gendata.
CONVERT_FAILURES_JSON = RESULTS_DIR / "convert_failures.json"

# --- Phase 2: phonological features (04_phase2_multihead) ---
# Per-feature label space + coverage (small, committed).
FEATURE_VOCAB_JSON = RESULTS_DIR / "feature_vocab.json"
# {clip_id: [int per feature]} sidecar — deliberately SEPARATE from upstream's
# {split}_label.pkl so gendata output stays byte-identical to a gloss-only run.
FEATURE_LABELS_PKL = PROCESSED_DIR / "feature_labels.pkl"

# --- External reference code (the SL-GCN implementation) ---
# PRIVATE clone (gitignored): same pinned commit ef0ab363 and the same 9-line
# compat patch as the SL-GCN Sem-Lex leaf, but a SEPARATE checkout on purpose. Upstream
# main.py hardcodes ./save_models and ./work_dir relative to CWD, so each
# project symlinks those into its own processed dir — sharing one clone would
# make two concurrent trainings fight over the symlinks and write checkpoints
# into each other's directories. Recreate with:
#   mkdir -p external && cp -r ../semlex/external/SAM-SLR-v2 external/ \
#     && rm -f external/SAM-SLR-v2/SL-GCN/{save_models,work_dir}
SAM_SLR_V2_DIR = REPO_ROOT / "external" / "SAM-SLR-v2"
SL_GCN_DIR = SAM_SLR_V2_DIR / "SL-GCN"

# --- Environment ---
CONDA_INIT = (os.environ.get("ASL2LSC_CONDA_INIT")
              or _SITE_DEFAULTS.get("ASL2LSC_CONDA_INIT")
              or "source ~/miniconda3/etc/profile.d/conda.sh")
ENV_NAME = "popsign-slgcn-env"  # environment.yml: slgcn-env + pyarrow + pose_format

# --- Dataset reference facts (measured) ---
DATASET = {
    "name": "PopSign ASL v1.0 (game subset)",
    "n_classes": 250,
    "n_clips_video": 174876,     # mp4 under VIDEOS_ROOT
    "n_clips_landmarks": 174243,  # parquet under LANDMARKS_DIR
    "landmarks_per_frame": 543,
    "video_resolution": "2592x1944",
    "video_fps": 30,
    "signers": {"train": 31, "val": 8, "test": 8},
    "signer_independent": True,  # asserted in build_splits.py
}

try:
    from local_data_config import *  # noqa: F401,F403
except ImportError:
    pass
