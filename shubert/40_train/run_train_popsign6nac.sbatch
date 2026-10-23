#!/bin/bash
#SBATCH --job-name=slsh_popsign6nac_train
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%A.out

# PopSign heads6 ablation -- the SHuBERT sibling of the Sem-Lex
# `handshape_heads` run. Same data and config as run_train_popsign_nac.sbatch
# (popsign16nac splits = NA-as-class, batch 128, --min-epochs 49, a 48 GB
# GPU) with --keep-heads handshape as the single variable: only the 6
# handshape-family heads receive loss, the other 10 stay at random init,
# early stop on the 6-head val macro. Using the 16nac label space keeps
# labels and head sizes identical to the main arm, as the Sem-Lex ablation
# did.
#
# Self-chaining is identical to run_train_popsign_nac.sbatch: a follow-up is
# submitted BEFORE training so a walltime kill resumes; the completing leg
# submits the in-domain eval.
#
#   sbatch 40_train/run_train_popsign6nac.sbatch

L="${SLURM_SUBMIT_DIR}/logs"   # where the #SBATCH --output above writes
RUN_NAME=popsign16nac_heads6
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
        "$REPO/40_train/run_train_popsign6nac.sbatch")
    echo "follow-up job $FOLLOWUP submitted (chain $((CHAIN + 1)))"
else
    echo "chain cap $MAX_CHAIN reached -- no follow-up submitted"
fi

# --min-epochs 49: sample-matched to Sem-Lex (see run_train_popsign_nac.sbatch).
python "${REPO}/40_train/train.py" \
    --run-name "$RUN_NAME" \
    --train-split popsign16nac_train --val-split popsign16nac_val \
    --batch-size 128 \
    --min-epochs 49 \
    --keep-heads handshape

# Reached only on a clean exit (walltime kills the python process instead).
echo "training complete -- submitting evals"
E1=$(sbatch --parsable --export=ALL,RUN="$RUN_NAME",SPLIT=popsign16nac_test \
     "$REPO/50_eval/run_eval_popsign_indomain.sbatch")
echo "in-domain eval job $E1 submitted"
