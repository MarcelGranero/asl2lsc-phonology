#!/bin/bash
#SBATCH --job-name=semlex_repose
#SBATCH --output=logs/repose_%A_%a.out
#SBATCH --error=logs/repose_%A_%a.err
#SBATCH --time=07:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --partition=medium
#SBATCH --array=0-31

# Re-extract SEM-LEX poses from video with MediaPipe Holistic, recovering the
# hands the release is missing (mean hand presence .248 -> ~.42 measured on a
# 12-clip probe, improved in 12/12).
#
# CPU-only: MediaPipe Holistic runs on CPU here and the job is decode-bound.
# ~8.5 s/clip/process => 85,592 videos ~ 203 CPU-hours. At 32 shards the
# largest split (train, 51,443) is ~3.8 h/shard, comfortably inside the
# `medium` partition's 8 h cap — note that cap is why walltime is 7:45 and
# not longer; asking for 12 h leaves the array PENDING on PartitionTimeLimit
# forever rather than failing loudly.
#
#   sbatch --export=ALL,SPLIT=train 01_data_prep/run_extract_semlex_poses.sbatch
#
# Resumable: re-running skips existing non-empty npys, so a timed-out shard
# just needs resubmitting. Output goes to a NEW directory tree — the released
# poses are left untouched so results can be compared against them.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SEMLEX=${ASL2LSC_SEMLEX_ROOT}
SPLIT="${SPLIT:?set SPLIT=train|val|test}"
OUT="${OUT:-${ASL2LSC_SLGCN_SEMLEX_DATA}/reposed/$SPLIT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsc-benchmark-env
set -euo pipefail

python "$REPO/01_data_prep/extract_semlex_poses.py" \
    --video-dir "$SEMLEX/extracted/$SPLIT/$SPLIT" \
    --out-dir "$OUT" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT"

echo "SHARD $SLURM_ARRAY_TASK_ID DONE"
