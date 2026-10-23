#!/bin/bash
#SBATCH --job-name=slsh_train
#SBATCH --partition=high
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Full fine-tune on train, early-stopped on val. Auto-resumes from
# runs/main/checkpoints/last.pt — resubmit as-is if walltime is hit.
# NOTE: `set -u` must come AFTER conda activate — the shared env's MKL
# activation hook reads $MKL_INTERFACE_LAYER unset and would abort the job.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
python "${REPO}/40_train/train.py" --run-name main
