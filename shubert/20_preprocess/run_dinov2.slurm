#!/bin/bash
#SBATCH --job-name=slsh_dinov2
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out

# env: SPLIT, SMOKE, CROP_TYPE (face|hands); array index = batch of 2000 crops
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_feature_extraction

REPO="${SLURM_SUBMIT_DIR}"   # the shubert/ leaf (submit_pipeline.sh submits from it)
DATA=${ASL2LSC_SHUBERT_DATA}
CKPTS=${ASL2LSC_SHUBERT_CKPTS}
PREFIX="${SPLIT}"
if [ "${SMOKE:-0}" = "1" ]; then PREFIX="${SPLIT}_smoke"; fi

mkdir -p "${DATA}/${SPLIT}/dinov2_face" "${DATA}/${SPLIT}/dinov2_hands"

if [ "${CROP_TYPE}" = "face" ]; then
    FH=face
    LIST="${DATA}/work_lists/${PREFIX}_face_crops.list.gz"
else
    FH=hand
    LIST="${DATA}/work_lists/${PREFIX}_hand_crops.list.gz"
fi

python "${REPO}/external/shubert/features/dinov2_features.py" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --batch-size 2000 \
    --face-hand "${FH}" \
    --face-crops-file-list "${DATA}/work_lists/${PREFIX}_face_crops.list.gz" \
    --hand-crops-file-list "${DATA}/work_lists/${PREFIX}_hand_crops.list.gz" \
    --models-folder "${CKPTS}" \
    --output-folder-face "${DATA}/${SPLIT}/dinov2_face" \
    --output-folder-hand "${DATA}/${SPLIT}/dinov2_hands" \
    --time-limit 27000
