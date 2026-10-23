"""Central path/config constants for the SL-GCN Sem-Lex leaf.

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

# --- SEM-LEX source data (the authors' Drive release) ---
# Poses are MediaPipe Holistic, float16, shape (T, 553, 3) = (x, y, z) normalised
# image coords, NaN where a landmark was not detected. Layout (empirically
# verified — see 01_data_prep/map_553_to_wholebody133.py):
#   [0:478] face mesh incl. iris, [478:511] body pose (BlazePose 33),
#   [511:532] left hand, [532:553] right hand.
# Videos exist for ALL splits (train 51,443 / val 18,495 / test 15,654);
# training runs on poses only, so the videos matter for QA only.
# They are NOT uniform: ~76% webm/VP9, ~13% mp4/H.264 (every file is named
# .webm regardless — probe the container, never trust the extension), ~11%
# webm/VP8; 94% 640x480 and 6% 1280x720; effective frame rate is ~30 fps for
# 80% of clips with a real low-rate tail (~11% at ~15 fps, ~9% at 20-24 fps)
# concentrated in four signers who appear in train/val but NOT in the
# signer-independent test split.
SEMLEX_ROOT = _path_from_env("ASL2LSC_SEMLEX_ROOT")
SEMLEX_METADATA_CSV = SEMLEX_ROOT / "semlex_metadata.csv"
POSES_DIR = {
    "train": SEMLEX_ROOT / "extracted/sem-lex-train-poses/train",
    "val": SEMLEX_ROOT / "extracted/sem-lex-val-poses/val",
    "test": SEMLEX_ROOT / "extracted/sem-lex-test-poses/test",
}
VIDEOS_DIR = {
    "train": SEMLEX_ROOT / "extracted/train/train",  # pending download
    "val": SEMLEX_ROOT / "extracted/val/val",
    "test": SEMLEX_ROOT / "extracted/test/test",
}

# --- Processed outputs (heavy artifacts, never in git) ---
PROCESSED_DIR = _path_from_env("ASL2LSC_SLGCN_SEMLEX_DATA")
# Per-video (T, 133, 3) float32 npys in COCO-WholeBody layout, named
# <video_id>_color.mp4.npy so upstream sign_gendata.py consumes them unchanged.
WHOLEBODY_NPY_DIR = PROCESSED_DIR / "wholebody133"
# Per-split "name,label_int" CSVs (no header) — upstream sign_gendata contract.
LABELS_DIR = PROCESSED_DIR / "labels"
# Output of upstream sign_gendata.py: {split}_data_joint.npy + {split}_label.pkl
GENDATA_DIR = PROCESSED_DIR / "gendata_27"

# --- Repo-relative outputs (small artifacts, committed) ---
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"
METADATA_DEDUPED_CSV = RESULTS_DIR / "metadata_deduped.csv"
DEDUPE_REPORT_JSON = RESULTS_DIR / "dedupe_report.json"
VOCAB_JSON = RESULTS_DIR / "vocab.json"
SPLITS_JSON = RESULTS_DIR / "splits.json"

# --- Phase 2: phonological features (04_phase2_multihead) ---
# Per-feature label space + deltas vs the paper's Table 1 (small, committed).
FEATURE_VOCAB_JSON = RESULTS_DIR / "feature_vocab.json"
# {video_id: [int per feature]} sidecar — deliberately SEPARATE from upstream's
# {split}_label.pkl so gendata output stays byte-identical to a gloss-only run.
FEATURE_LABELS_PKL = PROCESSED_DIR / "feature_labels.pkl"

# --- External reference code ---
SAM_SLR_V2_DIR = REPO_ROOT / "external" / "SAM-SLR-v2"
SL_GCN_DIR = SAM_SLR_V2_DIR / "SL-GCN"

# --- Environment ---
CONDA_INIT = (os.environ.get("ASL2LSC_CONDA_INIT")
              or _SITE_DEFAULTS.get("ASL2LSC_CONDA_INIT")
              or "source ~/miniconda3/etc/profile.d/conda.sh")
ENV_NAME = "slgcn-env"

# --- Paper reference numbers (Kezar et al., ASSETS'23, arXiv 2310.00196) ---
# NOTE: the paper is internally inconsistent. §5.1 prose says ISR top-1
# "67.7%" and §5.3 says "71.3%" for ISR+PFR, but Table 4 and Table 6 both
# report 66.6% (ISR) / 68.6% (ISR+PFR) on test, and Table 6 gives the val
# figures. We compare against the TABLE values (used twice, mutually
# consistent) and record the prose values for completeness.
PAPER = {
    "n_lemmatized_signs": 3149,       # incl. 945 signs with <5 instances...
    "n_signs_with_split": 2204,       # ...which are "not given a split"
    "n_lemmatized_videos": 65935,     # asllex 60,203 + signbank 5,732
    "n_asllex_videos": 60203,         # the paper's experimental set
    "n_val": 11954,                   # seen signers
    "n_test": 11127,                  # 10 held-out, more diverse signers
    "gloss_only_top1": 0.666,         # Table 4 + Table 6 (test)
    "gloss_only_top3": 0.815,         # Table 4
    "gloss_only_top1_val": 0.682,     # Table 6 (val, seen signers)
    "gloss_phoneme_top1": 0.686,      # Table 4 + Table 6 (test)
    "prose_top1": 0.677,              # §5.1 — inconsistent with Table 4
    "prose_phoneme_top1": 0.713,      # §5.3 — inconsistent with Table 4
}

try:
    from local_data_config import *  # noqa: F401,F403
except ImportError:
    pass
