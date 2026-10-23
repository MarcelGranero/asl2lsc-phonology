#!/bin/bash
#SBATCH --job-name=slsh_pta_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Held-out TEST evaluation for the pretraining ablation.
#
# Runs are DISCOVERED from runs/pta_* rather than hardcoded, so a cell that
# finishes later is picked up by re-running this script. Each run's test split
# is derived from its own name: Sem-Lex cells score on `test`, PopSign cells on
# `popsign16nac_test` (or the older `popsign{6,16}_test` label builds) --
# scoring a PopSign model against Sem-Lex labels would "work" and produce
# nonsense, so the mapping is explicit.
#
# PopSign features are not packed, hence --no-pack there.
#
# evaluate.py re-applies keep_heads from each checkpoint's own config, so the
# 6-head cells report macro over their TRAINED heads only; the 16-head macro
# would be diluted by 10 never-trained heads.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
RUNS=${ASL2LSC_SHUBERT_DATA}/runs

for D in "${RUNS}"/pta_*/; do
  RUN=$(basename "$D")
  CKPT="${D}checkpoints/best.pt"
  [ -f "$CKPT" ] || { echo "skip ${RUN}: no best.pt"; continue; }

  case "$RUN" in
    *popsign16nac*) SPLIT=popsign16nac_test; EXTRA=(--no-pack) ;;
    *popsign16*) SPLIT=popsign16_test; EXTRA=(--no-pack) ;;
    *popsign6*)  SPLIT=popsign6_test;  EXTRA=(--no-pack) ;;
    *)           SPLIT=test;           EXTRA=() ;;
  esac

  echo "=== ${RUN} -> split ${SPLIT} ==="
  python "${REPO}/50_eval/evaluate.py" --split "$SPLIT" --run-label "$RUN" \
      --ckpt "$CKPT" "${EXTRA[@]}" || echo "FAILED: ${RUN}"
done
