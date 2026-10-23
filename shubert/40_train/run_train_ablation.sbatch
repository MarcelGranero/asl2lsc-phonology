#!/bin/bash
#SBATCH --job-name=slsh_abl
#SBATCH --partition=high
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Input-modality ablation: hands only (face + body slots filled with the
# pretrained mask_emb). Identical to run_train.sbatch in every other respect
# -- same config, same data, same seed -- so it differs from the `main`
# baseline in exactly one variable. Auto-resumes from
# runs/hands_only/checkpoints/last.pt; resubmit as-is if walltime is hit.
#
# Usage: sbatch run_train_ablation.sbatch [KEEP] [RUN_NAME]
#   KEEP defaults to "hands" (see 40_train/ablation.py PRESETS).
#
# NOTE: `set -u` must come AFTER conda activate (shared env's MKL hook).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

KEEP="${1:-hands}"
RUN_NAME="${2:-${KEEP}_only}"

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
python "${REPO}/40_train/train.py" \
    --run-name "${RUN_NAME}" \
    --keep-channels "${KEEP}"
