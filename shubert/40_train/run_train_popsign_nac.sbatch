#!/bin/bash
#SBATCH --job-name=slsh_popsign_nac_train
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%A.out

# SHuBERT paper protocol on PopSign: batch size 128 (the Sem-Lex protocol
# value) and the NA-as-class label build (popsign16nac splits).
#
# GPU: the worst 128-clip padded batch needs ~27 GB (docs/POPSIGN_NOTES.md),
# so the job asks for a 48 GB card (adapt the --gres type to your site); the
# head's BatchNorm makes gradient accumulation a non-equivalent substitute.
# 8 h legs with self-chaining (the trainer resumes from last.pt).
#
# Self-chaining: a follow-up is submitted BEFORE training so a walltime kill
# resumes; a completed run (train.py prints its DONE line and exits 0)
# submits the in-domain eval instead and ends the chain.
#
#   sbatch [--dependency=afterok:<prep job>] 40_train/run_train_popsign_nac.sbatch

L="${SLURM_SUBMIT_DIR}/logs"   # where the #SBATCH --output above writes
RUN_NAME=popsign16nac_main
CHAIN="${CHAIN:-0}"
MAX_CHAIN=20

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
nvidia-smi -L || true

# Chain already finished? (a previous leg printed the completion marker)
if grep -hs "DONE: best val macro" "$L/${SLURM_JOB_NAME}"_*.out | grep -q .; then
    echo "training already complete (DONE marker found) -- ending chain"
    exit 0
fi

if (( CHAIN < MAX_CHAIN )); then
    FOLLOWUP=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --job-name="$SLURM_JOB_NAME" \
        --export=ALL,CHAIN=$((CHAIN + 1)) \
        "$REPO/40_train/run_train_popsign_nac.sbatch")
    echo "follow-up job $FOLLOWUP submitted (chain $((CHAIN + 1)))"
else
    echo "chain cap $MAX_CHAIN reached -- no follow-up submitted"
fi

# --min-epochs 49: sample-matched to Sem-Lex rather than epoch-matched
# (4.49M Sem-Lex sample-presentations / 91,296 PopSign train clips = 49.2
# epochs).
python "${REPO}/40_train/train.py" \
    --run-name "$RUN_NAME" \
    --train-split popsign16nac_train --val-split popsign16nac_val \
    --batch-size 128 \
    --min-epochs 49

# Reached only on a clean exit (walltime kills the python process instead):
# training genuinely finished, so submit the eval and let the follow-up
# leg find the DONE marker and end the chain. LSC transfer: add the run's
# best.pt to common/lsc_eval/inference/checkpoints.json and run
# run_inference_all.sh with that key.
echo "training complete -- submitting the in-domain eval"
E1=$(sbatch --parsable --export=ALL,RUN="$RUN_NAME" \
     "$REPO/50_eval/run_eval_popsign_indomain.sbatch")
echo "in-domain eval job $E1 submitted"
