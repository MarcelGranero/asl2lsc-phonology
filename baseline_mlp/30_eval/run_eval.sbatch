#!/bin/bash
#SBATCH --job-name=mlp_eval
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# CPU-only on purpose: this is a 2-layer MLP over ~15k 63-d vectors plus 74
# benchmark clips -- seconds of compute.
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err

# Usage: sbatch --job-name=<run> run_eval.sbatch <run-name>
# Scores the Sem-Lex test split, then the LSC benchmark on both splits
# (plain + inventory-constrained, + an xy-only depth diagnostic).

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

cd "${SLURM_SUBMIT_DIR}"   # submit from baseline_mlp/; mkdir -p logs first
RUN_NAME="${1:?run name}"

echo "########## Sem-Lex test ##########"
python 30_eval/evaluate_semlex.py --run-name "$RUN_NAME" --split test

echo "########## LSC benchmark ##########"
for split in test examples; do
    python 30_eval/eval_lsc_handshapes.py --run-name "$RUN_NAME" --split "$split"
    python 30_eval/eval_lsc_handshapes.py --run-name "$RUN_NAME" --split "$split" --constrain-lsc
    python 30_eval/eval_lsc_handshapes.py --run-name "$RUN_NAME" --split "$split" --xy-only
done
