#!/bin/bash
#SBATCH --job-name=slsh_popsign_indomain_eval
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# In-corpus test scoring for a PopSign-trained SHuBERT checkpoint,
# parameterised by RUN (default the NA-as-class main run). The LSC transfer
# arm is NOT here -- it goes through common/lsc_eval/run_inference_all.sh
# so every model scores through the one shared evaluator.
#
#   sbatch --export=ALL,RUN=popsign16nac_main 50_eval/run_eval_popsign_indomain.sbatch
#
# SPLIT defaults to "<RUN minus _main>_test"; override it when the run name
# is not "<split prefix>_main" (e.g. RUN=popsign16nac_heads6,SPLIT=popsign16nac_test).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
RUNS=${ASL2LSC_SHUBERT_DATA}/runs
RUN="${RUN:-popsign16nac_main}"
SPLIT="${SPLIT:-${RUN%_main}_test}"

nvidia-smi -L || true
python "${REPO}/50_eval/evaluate.py" \
    --ckpt "$RUNS/$RUN/checkpoints/best.pt" \
    --split "$SPLIT" --run-label "$RUN"
echo "POPSIGN IN-DOMAIN EVAL DONE"
