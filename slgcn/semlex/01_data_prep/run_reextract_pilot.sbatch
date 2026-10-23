#!/bin/bash
#SBATCH --job-name=semlex_repose
#SBATCH --output=logs/repose_%A_%a.out
#SBATCH --error=logs/repose_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=medium
#SBATCH --array=0-3

# PILOT: re-extract poses for a sample of SEM-LEX test clips and measure hand
# presence against the released 0.248 baseline. CPU-only (MediaPipe Holistic
# runs on CPU here), 4-way array over disjoint shards.
#
#   sbatch 01_data_prep/run_reextract_pilot.sbatch              # 400 test clips
#   LIMIT=0 SPLIT=train sbatch 01_data_prep/run_reextract_pilot.sbatch  # full
#
# Decision rule (write the answer down before looking):
#   released mean presence = 0.248.
#   If re-extracted mean is >= ~0.60, the release is the problem and a full
#   re-extraction + retrain is justified.
#   If it lands near 0.248, the videos themselves do not support hand
#   detection (resolution/motion blur) and the hands hypothesis is DEAD —
#   the shortfall must be explained some other way. Either outcome is a
#   result; do not re-run until it says what you want.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-400}"
OUT_DIR="${OUT_DIR:-${ASL2LSC_SLGCN_SEMLEX_DATA}/poses_reextracted}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pose_estimation      # mediapipe 0.10.11; slgcn-env has none
set -euo pipefail

python "$REPO/01_data_prep/reextract_poses.py" \
    --split "$SPLIT" --limit "$LIMIT" --out-dir "$OUT_DIR" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --report "$REPO/results/repose_pilot/${SPLIT}_shard${SLURM_ARRAY_TASK_ID:-0}.csv"
echo "REEXTRACT SHARD DONE"
