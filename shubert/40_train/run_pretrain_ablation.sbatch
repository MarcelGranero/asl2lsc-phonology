#!/bin/bash
#SBATCH --job-name=slsh_pta
#SBATCH --partition=high
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# A full 125-epoch baseline peaks at ~3.2 GB RSS, so 16G is ~5x headroom
# and keeps four arms inside the per-user QOS cap instead of serialising
# them behind it.
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Pretraining ablation, one cell per job.
#   $1 run-name      e.g. pta_semlex16_scratch
#   $2 init          pretrained | scratch
#   $3 heads         all | handshape
#   $4 (optional)    extra args, e.g. --train-split popsign16_train ...
#
# Every arm is seed-matched to its twin (config.yaml seed is fixed), so the
# ONLY difference between a scratch run and its pretrained twin is whether the
# encoder weights were loaded. --random-init also raises the backbone LR to the
# head LR: the 1/10 discriminative rate exists to protect PRETRAINED features,
# and applying it to a random encoder would handicap that arm and bias the
# ablation toward "pretraining helps".

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eu

RUN=$1; INIT=$2; HEADS=$3; shift 3
REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir

ARGS=(--run-name "$RUN" --fresh)
[ "$INIT" = scratch ] && ARGS+=(--random-init)
[ "$HEADS" = handshape ] && ARGS+=(--keep-heads handshape)

echo "=== $RUN | init=$INIT heads=$HEADS | $(date) ==="
python "$REPO/40_train/train.py" "${ARGS[@]}" "$@"
echo "=== $RUN done | $(date) ==="
