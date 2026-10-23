#!/bin/bash
#SBATCH --job-name=slsh_seed_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Evaluate ONE seed-variance run: Sem-Lex test + zero-shot LSC at NATIVE fps
# (the comparison baseline per baseline_mlp/data_config.py — never 25fps),
# plain and inventory-constrained, both splits. LSC outputs go to the repo's
# results/lsc_seed_evals/ (tracked), not the shared benchmark labels dir.
# Usage: sbatch --job-name=slsh_e_<run> run_eval_seed.sbatch <run-name>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eu

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
RUN_NAME="${1:?run name, e.g. main_s43}"
CKPT="${ASL2LSC_SHUBERT_DATA}/runs/${RUN_NAME}/checkpoints/best.pt"
OUT="${REPO}/results/lsc_seed_evals"
mkdir -p "$OUT"

python "${REPO}/50_eval/evaluate.py" --split test --run-label "$RUN_NAME" --ckpt "$CKPT"

for split in test examples; do
  python "${REPO}/50_eval/lsc_handshapes/eval_shubert_lsc.py" \
    --ckpt "$CKPT" --split "$split" \
    --run-label "$RUN_NAME" --out-dir "$OUT"
  python "${REPO}/50_eval/lsc_handshapes/eval_shubert_lsc.py" \
    --ckpt "$CKPT" --split "$split" \
    --run-label "${RUN_NAME}_cons" --out-dir "$OUT" --constrain-lsc
done
echo "SEED EVAL DONE: ${RUN_NAME}"
