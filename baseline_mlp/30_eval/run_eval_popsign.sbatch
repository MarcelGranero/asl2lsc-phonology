#!/bin/bash
#SBATCH --job-name=mlp_eval_popsign
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# CPU-only, same reasoning as run_eval.sbatch. In-corpus test scoring for a
# PopSign-trained baseline MLP. The LSC transfer arm is corpus-agnostic:
# 30_eval/eval_lsc_handshapes.py selects the label maps from the checkpoint's
# `corpus` field (and hard-fails on a class-count mismatch), so run it with
# the PopSign run name, or through ../common/lsc_eval/run_inference_all.sh.
#
# Usage: sbatch --job-name=<run> 30_eval/run_eval_popsign.sbatch <run-name>
# Submit from baseline_mlp/ (mkdir -p logs first).

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u          # AFTER activate: the MKL hook reads an unset var

cd "${SLURM_SUBMIT_DIR}"
RUN_NAME="${1:?run name}"

echo "########## PopSign test (in-corpus) ##########"
# evaluate_semlex.py follows the checkpoint's `corpus` field, so this scores
# PopSign frames against PopSign's label maps despite the script's name.
python 30_eval/evaluate_semlex.py --run-name "$RUN_NAME" --split test

echo "POPSIGN EVAL DONE"
