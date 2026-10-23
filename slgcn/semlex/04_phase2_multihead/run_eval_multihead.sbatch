#!/bin/bash
#SBATCH --job-name=semlex_mh_eval
#SBATCH --output=logs/mh_eval_%j.out
#SBATCH --error=logs/mh_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# Score a phase-2 checkpoint on Sem-Lex_test, Table 5 layout + majority
# baselines.
#
#   sbatch --export=ALL,REGIME=pfr,CKPT=save_models/semlex_pfr-best.pt,\
#RUN_NAME=semlex_pfr 04_phase2_multihead/run_eval_multihead.sbatch
#
# A checkpoint trained on the RE-POSED tensors (e.g. the released
# slgcn-semlex-pfr-repose.pt) must be scored on them too: add
#   EXTRA_ARGS="--gendata-dir $ASL2LSC_SLGCN_SEMLEX_DATA/gendata_27_reposed"
# Without it the released-pose gendata_27 is scored and the numbers are
# silently far lower.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
PROCESSED=${ASL2LSC_SLGCN_SEMLEX_DATA}

REGIME="${REGIME:?set REGIME=pfr or REGIME=isr_pfr}"
CKPT="${CKPT:?set CKPT=<checkpoint path>}"
RUN_NAME="${RUN_NAME:?set RUN_NAME=<results dir name>}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slgcn-env
set -euo pipefail
nvidia-smi -L || true

mkdir -p "$PROCESSED/save_models" "$PROCESSED/work_dir"
ln -sfn "$PROCESSED/save_models" "$SLGCN/save_models"
ln -sfn "$PROCESSED/work_dir" "$SLGCN/work_dir"

cd "$SLGCN"
export PYTHONPATH="$REPO/04_phase2_multihead:$REPO:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # $PWD = the clone (upstream model/, feeders/, graph/)

python "$REPO/04_phase2_multihead/eval_multihead.py" \
    --regime "$REGIME" --checkpoint "$CKPT" --run-name "$RUN_NAME" \
    --workers "$SLURM_CPUS_PER_TASK" ${EXTRA_ARGS:-}
echo "EVAL OK"
