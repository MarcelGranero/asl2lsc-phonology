#!/bin/bash
#SBATCH --job-name=popsign_mh_eval
#SBATCH --output=logs/mh_eval_%j.out
#SBATCH --error=logs/mh_eval_%j.err
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# Phase-2 test-split eval. --mem=48G: the feeder loads the full packed test
# array (1.6 GB) plus the train array for the majority-class baseline; 8G OOMs.
REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate popsign-slgcn-env
set -euo pipefail
cd "$SLGCN"
export PYTHONPATH="$REPO/04_phase2_multihead:$REPO:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # $PWD = the clone (upstream model/, feeders/, graph/)
# EXTRA_ARGS mirrors the SemLex twin: the 16-head NA-class arms need
# --feature-vocab/--feature-labels (the defaults build the 6-head model
# and the checkpoint load fails on head sizes).
python "$REPO/04_phase2_multihead/eval_multihead.py" \
    --checkpoint "$CKPT" --regime "$REGIME" --run-name "$RUN_NAME" \
    --workers "$SLURM_CPUS_PER_TASK" ${EXTRA_ARGS:-}
echo "MH EVAL DONE"
