#!/bin/bash
#SBATCH --job-name=mlp_train
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# Usage: sbatch --job-name=<run> run_train.sbatch <head-set> <run-name> [arch] [seed]
#   sbatch --job-name=all16  20_train/run_train.sbatch all16  all16
#   sbatch --job-name=heads6 20_train/run_train.sbatch heads6 heads6
#   sbatch --job-name=heads6_s43 20_train/run_train.sbatch heads6 heads6_s43 multi 43

# No `set -u` around conda activate: shubert_train_env's
# activate.d/libblas_mkl_activate.sh reads an unbound MKL_INTERFACE_LAYER.
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

cd "${SLURM_SUBMIT_DIR}"   # submit from baseline_mlp/; mkdir -p logs first

HEAD_SET="${1:?head set: all16|heads6}"
RUN_NAME="${2:?run name}"
ARCH="${3:-multi}"
SEED="${4:-42}"
CORPUS="${5:-semlex}"   # semlex|popsign; default keeps existing call sites

echo "host $(hostname)  head-set $HEAD_SET  run $RUN_NAME  arch $ARCH" \
     "seed $SEED  corpus $CORPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python 20_train/train.py \
    --head-set "$HEAD_SET" \
    --arch "$ARCH" \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --corpus "$CORPUS"
