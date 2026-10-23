#!/bin/bash
#SBATCH --job-name=slsh_kpe
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out

# env: SPLIT, SMOKE; array index = batch of 100 videos from the work list
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_feature_extraction

REPO="${SLURM_SUBMIT_DIR}"   # the shubert/ leaf (submit_pipeline.sh submits from it)
DATA=${ASL2LSC_SHUBERT_DATA}
PREFIX="${SPLIT}"
if [ "${SMOKE:-0}" = "1" ]; then PREFIX="${SPLIT}_smoke"; fi

mkdir -p "${DATA}/${SPLIT}/kpe_poses" "${DATA}/${SPLIT}/kpe_stats"

# --time-limit: walltime minus 15 min so the task exits cleanly; the
# remainder is picked up on resubmit (exists-skip).
python "${REPO}/external/shubert/dataset/kpe_mediapipe.py" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --batch-size 100 \
    --files-list "${DATA}/work_lists/${PREFIX}_videos.list.gz" \
    --pose-path "${DATA}/${SPLIT}/kpe_poses" \
    --stats-path "${DATA}/${SPLIT}/kpe_stats" \
    --face-model "${REPO}/external/shubert/models/face_landmarker_v2_with_blendshapes.task" \
    --hand-model "${REPO}/external/shubert/models/hand_landmarker.task" \
    --problem-log "${DATA}/pipeline_logs/${SPLIT}_kpe_problems.log" \
    --time-limit 13500
