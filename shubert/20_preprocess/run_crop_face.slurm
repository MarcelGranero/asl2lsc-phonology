#!/bin/bash
#SBATCH --job-name=slsh_crop_face
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out

# env: SPLIT, SMOKE; array index = batch of 200 videos.
# NO --keep_entire_face: the default masked eyes+mouth-on-grey crop is the
# paper's privacy-masked face stream — required for reproduction.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_feature_extraction

REPO="${SLURM_SUBMIT_DIR}"   # the shubert/ leaf (submit_pipeline.sh submits from it)
DATA=${ASL2LSC_SHUBERT_DATA}
PREFIX="${SPLIT}"
if [ "${SMOKE:-0}" = "1" ]; then PREFIX="${SPLIT}_smoke"; fi

mkdir -p "${DATA}/${SPLIT}/pose_face_crops"

python "${REPO}/external/shubert/dataset/crop_face.py" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --batch_size 200 \
    --files_list "${DATA}/work_lists/${PREFIX}_videos.list.gz" \
    --pose_dir "${DATA}/${SPLIT}/kpe_poses" \
    --face_dir "${DATA}/${SPLIT}/pose_face_crops" \
    --problem_file_path "${DATA}/pipeline_logs/${SPLIT}_crop_face_problems.log" \
    --time_limit 13500
