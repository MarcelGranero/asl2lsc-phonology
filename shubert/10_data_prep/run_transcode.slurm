#!/bin/bash
#SBATCH --job-name=slsh_transcode
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%A_%a.out

# env: SPLIT (train|val|test), SMOKE (0|1); array index selects the file batch
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_feature_extraction

REPO="${SLURM_SUBMIT_DIR}"   # the shubert/ leaf (submit_pipeline.sh submits from it)
LIMIT_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then LIMIT_ARGS=(--limit 200); fi

python "${REPO}/10_data_prep/transcode_videos.py" \
    --split "${SPLIT}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --batch-size 300 \
    "${LIMIT_ARGS[@]}"
