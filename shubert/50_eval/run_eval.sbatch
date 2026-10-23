#!/bin/bash
#SBATCH --job-name=slsh_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# NOTE: `set -eu` must come AFTER conda activate — the shared env's MKL
# activation hook reads $MKL_INTERFACE_LAYER unset and would abort the job.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eu

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
CKPT_ROOT=${ASL2LSC_SHUBERT_DATA}/runs

# Baseline: the reported Table 9 comparison. test first (the comparable
# split — val shares 33 signers with train and is NOT comparable).
python "${REPO}/50_eval/evaluate.py" --split test --run-label main
python "${REPO}/50_eval/evaluate.py" --split val  --run-label main

# Every ablation run that has produced a checkpoint. Discovered, not
# hardcoded: ablations get added over time (hands_only, body_only, ...) and a
# hardcoded list silently skips the new one. keep_channels is re-applied from
# the ckpt config by evaluate.py — do NOT pass --keep-channels here.
for D in "${CKPT_ROOT}"/*/; do
  RUN=$(basename "$D")
  case "$RUN" in main|smoke|*_smoke) continue ;; esac
  ABL="${D}checkpoints/best.pt"
  [ -f "$ABL" ] || { echo "skip ${RUN}: no best.pt yet"; continue; }
  echo "evaluating ablation run: ${RUN}"
  python "${REPO}/50_eval/evaluate.py" --split test --run-label "$RUN" --ckpt "$ABL"
  python "${REPO}/50_eval/evaluate.py" --split val  --run-label "$RUN" --ckpt "$ABL"
done

python "${REPO}/50_eval/make_comparison.py"
