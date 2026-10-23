"""Single source of truth for every path and reference number used by
the baseline MLP. Stdlib only, so it imports under any of the envs we touch.

The MLP baseline is a static single-hand model: one 63-float landmark frame
in, one prediction per phonological feature out (architecture after Carbo &
Nalisnick, arXiv:2509.18309; see 20_train/model.py). This file supplies the
Sem-Lex / PopSign inputs and labels around it.
"""

import json as _json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

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

# ------------------------------------------------------------- source labels
# Reused verbatim from the SHuBERT leaf: same 16 ASL-LEX features, same class
# maps, already gated against the paper's Table 10 counts. Re-deriving them
# here would risk a silently different label space and make the SHuBERT /
# SL-GCN / MLP comparison meaningless.
SHUBERT_REPO = REPO_ROOT.parent / "shubert"
LABEL_MAPS_JSON = SHUBERT_REPO / "results" / "label_maps.json"
SHUBERT_LABELS_CSV = {
    split: SHUBERT_REPO / "results" / f"{split}_labels.csv"
    for split in ("train", "val", "test")
}

# ------------------------------------------------------------- source inputs
# MediaPipe keypoints already extracted by the SHuBERT-leaf pipeline
# (kpe_mediapipe.py). Per-clip JSON: {frame_idx: {hand_landmarks, ...}}.
SEMLEX_SHUBERT_DATA = _path_from_env("ASL2LSC_SHUBERT_DATA")


def kpe_dir(split: str) -> Path:
    assert split in SPLITS
    return SEMLEX_SHUBERT_DATA / split / "kpe_poses"


SPLITS = ("train", "val", "test")

# LSC benchmark clips as (T,553,3) npys, converted once by the SL-GCN Sem-Lex leaf
# (pose_format is absent from every env here, so the .pose files are unusable
# directly). Lives under the SL-GCN Sem-Lex data root.
POSE553_DIR = (
    _path_from_env("ASL2LSC_SLGCN_SEMLEX_DATA") / "lsc_handshapes" / "pose553"
)

# ------------------------------------------------------------ heavy outputs
DATA_ROOT = _path_from_env("ASL2LSC_MLP_DATA")
FRAMES_DIR = DATA_ROOT / "frames"        # {split}.npz: X (N,63), ids

# --- PopSign arm ----------------------------------------------------------
# Same architecture and the same 63-float input format, but a different
# corpus: frames come from 10_data_prep/extract_frames_popsign.py (hand
# landmarks lifted out of the wholebody-133 poses), and the labels are
# PopSign's OWN encoding. The two label spaces are NOT interchangeable --
# the feature ORDER matches, but 11 of the 16 heads have a different number
# of classes because PopSign's vocabulary is derived from its 250-gloss
# subset (e.g. Minor Location 25 vs Sem-Lex's 37). Anything reading these
# must use POPSIGN_VOCAB16, never LABEL_MAPS_JSON.
POPSIGN_ROOT = _path_from_env("ASL2LSC_SLGCN_POPSIGN_DATA")
POPSIGN_FRAMES_DIR = DATA_ROOT / "popsign_frames"
# Legacy NA-masked build (NA as -1), kept so older checkpoints decode;
# new runs use the NA-as-class build below.
POPSIGN_LABELS16 = POPSIGN_ROOT / "feature_labels16.pkl"
POPSIGN_VOCAB16 = (Path(__file__).resolve().parent.parent / "slgcn" /
                   "popsign" / "results" / "feature_vocab16.json")
# NA-as-class build (build_feature_labels16.py --null-policy class): NA is a
# real class like in Sem-Lex; -1 remains only for genuinely unknown values
# (unmatched gloss). The default for every new training.
POPSIGN_LABELS16_NACLASS = POPSIGN_ROOT / "feature_labels16_naclass.pkl"
POPSIGN_VOCAB16_NACLASS = (Path(__file__).resolve().parent.parent / "slgcn" /
                           "popsign" / "results" /
                           "feature_vocab16_naclass.json")
RUNS_DIR = DATA_ROOT / "runs"            # checkpoints per run

# --------------------------------------------------------- the two head sets
# "all 16 heads, like SL-GCN and SHuBERT" and the 6-head handshape subset that
# the LSC Benchmark actually scores. Trained as two parallel runs.
#
# HEADS6 order matches the `hierarchical` contract in 20_train/model.py:
#   [handshape, selected_fingers, flexion, spread, thumb_position, thumb_contact]
# (num_classes[0] sizes the handshape head; the rest are subheads feeding the
# concat classifier).
FEATURE_ORDER_16 = [
    "Major Location", "Minor Location", "Second Minor Location", "Contact",
    "Thumb Contact", "Sign Type", "Repeated Movement", "Path Movement",
    "Wrist Twist", "Selected Fingers", "Thumb Position", "Flexion", "Spread",
    "Spread Change", "Nondominant Handshape", "Handshape",
]

HEADS6 = [
    "Handshape", "Selected Fingers", "Flexion", "Spread",
    "Thumb Position", "Thumb Contact",
]

# The 5 attributes that are directly comparable with the LSC benchmark labels
# (LSC_Benchmark_handshapes/labels/README.txt). Handshape is NOT: LSC uses a
# 37-class <base>-<variant> vocabulary with zero id overlap.
LSC_COMPARABLE = [
    "Selected Fingers", "Flexion", "Spread", "Thumb Position", "Thumb Contact",
]

# Heads the MLP cannot observe: it sees ONE still frame of ONE hand, with no
# body, no face and no temporal axis. Recorded so the 16-head report can flag
# them rather than pass a learned class prior off as signal.
UNOBSERVABLE_HEADS = [
    "Major Location", "Minor Location", "Second Minor Location", "Contact",
    "Sign Type", "Repeated Movement", "Path Movement", "Wrist Twist",
    "Spread Change", "Nondominant Handshape",
]

# ----------------------------------------------------------- LSC benchmark
# A separate release: ASL2LSC_LSC_BENCH is its root (labels/ holds
# lsc_handshape_distance.py); ASL2LSC_LSC_LABELS holds the 37-clip gold labels
# and the per-split manifests.
LSC_BENCH = _path_from_env("ASL2LSC_LSC_BENCH")
LSC_LABELS_DIR = LSC_BENCH / "labels"
LSC_V2_DIR = Path(os.environ.get("ASL2LSC_LSC_LABELS")
                  or REPO_ROOT.parent / "shubert" / "50_eval" / "lsc_labels_v2")

# ------------------------------------------------------------- environments
CONDA_INIT = (os.environ.get("ASL2LSC_CONDA_INIT")
              or _SITE_DEFAULTS.get("ASL2LSC_CONDA_INIT")
              or "source ~/miniconda3/etc/profile.d/conda.sh")
# Training env: needs torch + torch_geometric (upstream DataLoader path).
TRAIN_ENV = (os.environ.get("ASL2LSC_TRAIN_ENV")
             or _SITE_DEFAULTS.get("ASL2LSC_TRAIN_ENV")
             or "shubert_train_env")

try:
    from local_data_config import *  # noqa: F401,F403
except ImportError:
    pass
