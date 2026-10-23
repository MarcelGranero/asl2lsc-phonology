#!/bin/bash
#SBATCH --job-name=slsh_heads
#SBATCH --partition=high
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Supervision ablation: train ONLY the 6 handshape-family heads (Handshape,
# Selected Fingers, Flexion, Spread, Thumb Position, Thumb Contact) — exactly
# the supervision the PopSign models get, so SemLex-vs-PopSign LSC transfer
# comparisons stop being confounded by head set. Identical to run_train.sbatch
# in every other respect (same config, data, seed): one variable vs `main`.
# The other 10 heads stay at random init; early stop is on the 6-head val
# macro. Auto-resumes from runs/handshape_heads/checkpoints/last.pt;
# resubmit as-is if walltime is hit.
#
# Usage: sbatch run_train_headablation.sbatch [HEADS] [RUN_NAME]
#   HEADS defaults to the "handshape" preset (see 40_train/train.py).
#
# NOTE: `set -u` must come AFTER conda activate (shared env's MKL hook).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

HEADS="${1:-handshape}"
RUN_NAME="${2:-handshape_heads}"

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
python "${REPO}/40_train/train.py" \
    --run-name "${RUN_NAME}" \
    --keep-heads "${HEADS}"
