#!/bin/bash
#SBATCH --job-name=slsh_seed
#SBATCH --partition=high
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Seed-variance arm of the `main` run: identical recipe, different seed.
# Usage: sbatch --job-name=slsh_<run> run_train_seed.sbatch <run-name> <seed>
# Auto-resumes from runs/<run-name>/checkpoints/last.pt — submit as an
# afterany chain of 24h continuations, same as main.
# NOTE: `set -u` must come AFTER conda activate — the shared env's MKL
# activation hook reads $MKL_INTERFACE_LAYER unset and would abort the job.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
RUN_NAME="${1:?run name, e.g. main_s43}"
SEED="${2:?seed}"
python "${REPO}/40_train/train.py" --run-name "$RUN_NAME" --seed "$SEED"
