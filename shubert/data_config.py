"""Single source of truth for every absolute path and reference number
used by the SHuBERT leaf. Stdlib only (must import under shubert_train_env,
which has no pandas).

Heavy artifacts live under DATA_ROOT; the repo holds only code + small
committed results.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
QA_DIR = REPO_ROOT / "docs" / "qa"

# Every heavy-data location resolves through an ASL2LSC_* environment
# variable (shared registry across the leaves). On the cluster the
# _SITE_DEFAULTS below make everything work with no env set; the public
# release ships with the block emptied, so an unset variable fails loudly
# with the variable name instead of a dead absolute path.
# --- site defaults (optional per-machine values) ---
_SITE_DEFAULTS = {}
# --- end site defaults ---


def _path_from_env(var: str) -> Path:
    value = os.environ.get(var) or _SITE_DEFAULTS.get(var)
    if not value:
        raise RuntimeError(
            f"Environment variable {var} is not set. Point it at the "
            f"corresponding data root (see the repository README, "
            f"'Data layout')."
        )
    return Path(value)


# ---------------------------------------------------------------- source data
SEMLEX_ROOT = _path_from_env("ASL2LSC_SEMLEX_ROOT")
SEMLEX_METADATA = SEMLEX_ROOT / "semlex_metadata.csv"
# Reused verbatim from the SL-GCN reproduction (row dedupe prefers the
# asllex row; 33 cross-split byte-identical clips dropped). Monorepo sibling;
# the file is gitignored there (regenerable), so after a fresh clone rerun
# slgcn/semlex/01_data_prep or copy it in — the sha assert below catches a
# wrong copy either way.
METADATA_DEDUPED = (
    Path(__file__).resolve().parent.parent
    / "slgcn" / "semlex" / "results" / "metadata_deduped.csv"
)
METADATA_DEDUPED_SHA256 = (
    "5d471306abe24adf696a8c71c3331853231dbbe88ac6f9b5a964dfc582858985"
)

# Released .webm videos (train/ appears once train.tar.gz is extracted).
SEMLEX_VIDEO_DIRS = {
    "train": SEMLEX_ROOT / "extracted" / "train" / "train",
    "val": SEMLEX_ROOT / "extracted" / "val" / "val",
    "test": SEMLEX_ROOT / "extracted" / "test" / "test",
}

SPLITS = ("train", "val", "test")

# ------------------------------------------------------------- heavy outputs
DATA_ROOT = _path_from_env("ASL2LSC_SHUBERT_DATA")
VIDEOS_MP4_DIR = DATA_ROOT / "videos_mp4"          # {split}/<video_id>.mp4
WORK_LISTS_DIR = DATA_ROOT / "work_lists"          # *.list.gz (dedicated dir:
#   kpe_mediapipe.py rewrites <parent>/<parent-name>.list.gz at startup, so
#   work lists must not share a directory with any stage outputs)
STAGE_DIRS = (
    "kpe_poses",
    "kpe_stats",
    "pose_face_crops",
    "pose_hand_crops",
    "dinov2_face",
    "dinov2_hands",
    "body_features",
)


def stage_dir(split: str, stage: str) -> Path:
    assert split in SPLITS and stage in STAGE_DIRS
    return DATA_ROOT / split / stage


MANIFESTS_DIR = DATA_ROOT / "manifests"
PACKED_DIR = DATA_ROOT / "packed"
RUNS_DIR = DATA_ROOT / "runs"
PIPELINE_LOGS_DIR = DATA_ROOT / "pipeline_logs"    # per-stage problem logs

# ------------------------------------------------------ upstream model assets
SHUBERT_CKPTS_DIR = _path_from_env("ASL2LSC_SHUBERT_CKPTS")
SHUBERT_CHECKPOINT = SHUBERT_CKPTS_DIR / "checkpoint_836_400000.pt"
SHUBERT_CHECKPOINT_SHA256 = (
    "9087f47602401c07c44a9c70164ca1387373938ac4642149126a43e61c20cd7d"
)
# dinov2face.pth / dinov2hand.pth are looked up inside SHUBERT_CKPTS_DIR by
# the vendored dinov2_features.py (--models-folder).

MEDIAPIPE_FACE_MODEL = (
    REPO_ROOT / "external" / "shubert" / "models"
    / "face_landmarker_v2_with_blendshapes.task"
)
MEDIAPIPE_HAND_MODEL = (
    REPO_ROOT / "external" / "shubert" / "models" / "hand_landmarker.task"
)

# --------------------------------------------------------------- label schema
# The 16 phonological features of SHuBERT Table 9 (= ASL-LEX 2.0), in the
# paper's row order. Metadata column name -> expected number of classes
# (paper Table 10; counts INCLUDE the NA class where the column has blanks).
# `Flexion Change` exists in the metadata but is not part of Table 9.
FEATURES_TABLE10 = [
    ("Major Location", 5),
    ("Minor Location", 37),
    ("Second Minor Location", 37),
    ("Contact", 2),
    ("Thumb Contact", 3),
    ("Sign Type", 6),
    ("Repeated Movement", 2),
    ("Path Movement", 8),
    ("Wrist Twist", 2),
    ("Selected Fingers", 12),
    ("Thumb Position", 2),
    ("Flexion", 8),
    ("Spread", 3),
    ("Spread Change", 3),
    ("Nondominant Handshape", 56),
    ("Handshape", 58),
]
NA_CLASS = "__NA__"

# Paper Table 9, Sem-Lex column (Recall@1) — the reproduction target.
PAPER_TABLE9_SEMLEX = {
    "Major Location": 0.8477,
    "Minor Location": 0.7130,
    "Second Minor Location": 0.7328,
    "Contact": 0.8684,
    "Thumb Contact": 0.8474,
    "Sign Type": 0.8464,
    "Repeated Movement": 0.8265,
    "Path Movement": 0.7275,
    "Wrist Twist": 0.9058,
    "Selected Fingers": 0.7953,
    "Thumb Position": 0.8604,
    "Flexion": 0.7264,
    "Spread": 0.7942,
    "Spread Change": 0.8160,
    "Nondominant Handshape": 0.7632,
    "Handshape": 0.6293,
}
PAPER_TABLE9_SEMLEX_AVG = 0.7938
