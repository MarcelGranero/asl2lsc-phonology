#!/bin/bash
#SBATCH --job-name=slsh_body
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A.out

# env: SPLIT, SMOKE; single task, iterates the whole pose list (fast: JSON->npy)
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_feature_extraction

REPO="${SLURM_SUBMIT_DIR}"   # the shubert/ leaf (submit_pipeline.sh submits from it)
DATA=${ASL2LSC_SHUBERT_DATA}
PREFIX="${SPLIT}"
if [ "${SMOKE:-0}" = "1" ]; then PREFIX="${SPLIT}_smoke"; fi

mkdir -p "${DATA}/${SPLIT}/body_features"

python "${REPO}/external/shubert/features/body_features.py" \
    --files-list "${DATA}/work_lists/${PREFIX}_kpe_poses.list.gz" \
    --output-dir "${DATA}/${SPLIT}/body_features" \
    --batch-size 500 \
    --time-limit 6600
