#!/bin/bash
#SBATCH --job-name=popsign_mh
#SBATCH --output=logs/mh_train_%j.out
#SBATCH --error=logs/mh_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=high

# Phase-2 multi-head training (04_phase2_multihead/train_multihead.py).
#
#   REGIME=pfr      §5.2/Table 5 — requires INIT_CKPT (gloss-pretrained)
#   REGIME=isr_pfr  §5.3/Table 4 — joint from scratch
#
#   sbatch --export=ALL,REGIME=pfr,INIT_CKPT=<best gloss checkpoint, e.g. save_models/popsign_joint-<E>.pt> \
#          --job-name=popsign_pfr \
#          04_phase2_multihead/run_multihead.sbatch
#
# Walltime chaining mirrors 02_train/run_train.sbatch: a follow-up job is
# submitted with --dependency=afterany BEFORE training starts, so a
# walltime-kill resumes from the last epoch checkpoint; a completed run exits
# before submitting, ending the chain. CHAIN caps runaway crash-loops.
#
# NOTE the resume semantics differ from phase 1 in one way that matters for
# --regime pfr: on resume we continue from OUR OWN latest checkpoint via
# --init-from, which already contains the gloss-pretrained encoder plus the
# partially-trained feature heads. INIT_CKPT is therefore only used for the
# very first job in a chain.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
PROCESSED=${ASL2LSC_SLGCN_POPSIGN_DATA}

REGIME="${REGIME:?set REGIME=pfr or REGIME=isr_pfr}"
EXP_NAME="${EXP_NAME:-popsign_${REGIME}}"
CONFIG="${CONFIG:-$REPO/02_train/config/train_joint_popsign.yaml}"
CHAIN="${CHAIN:-0}"
MAX_CHAIN=15

source ~/miniconda3/etc/profile.d/conda.sh
conda activate popsign-slgcn-env
set -euo pipefail
nvidia-smi -L || true

NUM_EPOCH="${NUM_EPOCH:-$(grep -Po '^num_epoch:\s*\K\d+' "$CONFIG")}"

mkdir -p "$PROCESSED/save_models" "$PROCESSED/work_dir"
ln -sfn "$PROCESSED/save_models" "$SLGCN/save_models"
ln -sfn "$PROCESSED/work_dir" "$SLGCN/work_dir"

# FROM_SCRATCH=1 opts out of the §5.2 requirement: feature heads on a randomly
# initialised encoder, no gloss objective. A deliberate deviation (recorded in
# the run card as deviates_from_paper_pfr) — it measures how much handshape
# phonology is learnable from pose alone, and is NOT comparable to Table 5.
if [[ "$REGIME" == "pfr" && -z "${INIT_CKPT:-}" && "${FROM_SCRATCH:-0}" != "1" ]]; then
    echo "REGIME=pfr reproduces the paper's §5.2 setup (feature heads on a" >&2
    echo "gloss-pretrained encoder) and needs INIT_CKPT=<gloss checkpoint>." >&2
    echo "Set FROM_SCRATCH=1 to train from a random encoder instead." >&2
    exit 1
fi
if [[ "${FROM_SCRATCH:-0}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS:-} --from-scratch"
fi

# Resume from our own newest per-epoch checkpoint, if any. Files are named
# <exp>-<epoch>.pt (0-based) alongside a rolling <exp>-best.pt, which is
# excluded from the scan by the numeric sed match.
RESUME_ARGS=()
LATEST=$(ls -1 "$SLGCN/save_models/$EXP_NAME"-*.pt 2>/dev/null \
         | sed -nE 's/.*-([0-9]+)\.pt/\1 &/p' | sort -n | tail -1 || true)
if [[ -n "${LATEST:-}" ]]; then
    EPOCH=$(echo "$LATEST" | cut -d' ' -f1)
    WEIGHTS=$(echo "$LATEST" | cut -d' ' -f2-)
    DONE=$((EPOCH + 1))
    if (( DONE >= NUM_EPOCH )); then
        echo "training already complete ($WEIGHTS) — ending chain"; exit 0
    fi
    echo "resuming from $WEIGHTS ($DONE epochs done of $NUM_EPOCH)"
    # --start-epoch is what keeps the LR schedule, checkpoint names and logs
    # on ABSOLUTE epochs across a resume. Without it the counter restarts at
    # 0, so warmup re-runs and step:[150,200] is measured from the resume.
    RESUME_ARGS=(--init-from "$WEIGHTS" --num-epoch $((NUM_EPOCH - DONE))
                 --start-epoch "$DONE")
    # A resume ALWAYS loads our own checkpoint, so --from-scratch (mutually
    # exclusive with --init-from) must be dropped from the chained job. The
    # encoder was randomly initialised at epoch 0; that fact lives in the
    # run card, not in the resume arguments.
    # The `:-` guard is required under `set -u`: EXTRA_ARGS is unset on
    # every run that does not pass it.
    EXTRA_ARGS="${EXTRA_ARGS:-}"
    EXTRA_ARGS="${EXTRA_ARGS//--from-scratch/}"
else
    # Fresh start. --num-epoch is passed even here: train_multihead.py
    # otherwise falls back to the config's num_epoch and ignores NUM_EPOCH.
    RESUME_ARGS=(--num-epoch "$NUM_EPOCH")
    if [[ -n "${INIT_CKPT:-}" ]]; then
        RESUME_ARGS+=(--init-from "$INIT_CKPT")
    fi
fi

if (( CHAIN < MAX_CHAIN )); then
    FOLLOWUP=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --job-name="$SLURM_JOB_NAME" \
        --export=ALL,REGIME="$REGIME",EXP_NAME="$EXP_NAME",CONFIG="$CONFIG",\
INIT_CKPT="${INIT_CKPT:-}",NUM_EPOCH="$NUM_EPOCH",FROM_SCRATCH="${FROM_SCRATCH:-0}",\
CHAIN=$((CHAIN + 1)) \
        "$REPO/04_phase2_multihead/run_multihead.sbatch")
    echo "follow-up job $FOLLOWUP submitted (chain $((CHAIN + 1)))"
else
    echo "chain cap $MAX_CHAIN reached — no follow-up submitted"
fi

# Run from upstream's dir so feeders./model./graph. resolve to the clone;
# our modules arrive via PYTHONPATH. Nothing in the clone is modified.
# The cd MUST stay below the follow-up submission: the chained job inherits
# this job's CWD as its submit dir, and both its relative logs/ output path
# and REPO="${SLURM_SUBMIT_DIR}" only resolve at the leaf root.
cd "$SLGCN"
export PYTHONPATH="$REPO/04_phase2_multihead:$REPO:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # $PWD = the clone (upstream model/, feeders/, graph/)

python "$REPO/04_phase2_multihead/train_multihead.py" \
    --regime "$REGIME" --exp-name "$EXP_NAME" --config "$CONFIG" \
    --workers "$SLURM_CPUS_PER_TASK" "${RESUME_ARGS[@]}" ${EXTRA_ARGS:-}
echo "train_multihead.py exited cleanly"
