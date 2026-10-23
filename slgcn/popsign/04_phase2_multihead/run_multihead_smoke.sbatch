#!/bin/bash
#SBATCH --job-name=popsign_mh_smoke
#SBATCH --output=logs/mh_smoke_%j.out
#SBATCH --error=logs/mh_smoke_%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# 2-epoch smoke of BOTH phase-2 regimes on the first 100 samples.
# Verifies: multi-head forward/backward, the summed CE loss, the sidecar join,
# per-feature eval, and that a phase-1 gloss checkpoint loads into the
# multi-head model. Results land under results/_smoke/.
#
# Pass a phase-1 checkpoint to exercise the pfr regime too:
#   sbatch --export=ALL,INIT_CKPT=<gloss checkpoint> \
#          04_phase2_multihead/run_multihead_smoke.sbatch

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
PROCESSED=${ASL2LSC_SLGCN_POPSIGN_DATA}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate popsign-slgcn-env
set -euo pipefail
nvidia-smi -L || true

mkdir -p "$PROCESSED/save_models" "$PROCESSED/work_dir"
ln -sfn "$PROCESSED/save_models" "$SLGCN/save_models"
ln -sfn "$PROCESSED/work_dir" "$SLGCN/work_dir"

# Run from upstream's dir so `feeders.`/`model.`/`graph.` resolve to the clone;
# our own modules come in via PYTHONPATH. Nothing in the clone is modified.
cd "$SLGCN"
export PYTHONPATH="$REPO/04_phase2_multihead:$REPO:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # $PWD = the clone (upstream model/, feeders/, graph/)

echo "=== regime isr_pfr (joint, from scratch) ==="
time python "$REPO/04_phase2_multihead/train_multihead.py" \
    --regime isr_pfr --smoke --workers 4

if [[ -n "${INIT_CKPT:-}" ]]; then
    echo "=== regime pfr (feature heads on a gloss-pretrained encoder) ==="
    time python "$REPO/04_phase2_multihead/train_multihead.py" \
        --regime pfr --smoke --workers 4 --init-from "$INIT_CKPT"
else
    echo "=== regime pfr SKIPPED (no INIT_CKPT given) ==="
fi

echo "MULTIHEAD SMOKE OK"
