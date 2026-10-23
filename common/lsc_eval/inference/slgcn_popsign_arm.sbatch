#!/bin/bash
#SBATCH --job-name=lsc_infer_slgcn_pop
#SBATCH --output=logs/infer_slgcn_pop_%j.out
#SBATCH --error=logs/infer_slgcn_pop_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# One SL-GCN/PopSign checkpoint + treatment -> LSC-benchmark predictions
# (test split; the leaf script packs the raw .pose files on the fly). A GPU
# is REQUIRED: upstream's decouple_gcn_attn allocates parameters directly
# on CUDA. The leaf script reads the benchmark manifest via
# ASL2LSC_LSC_BENCH. predictions.csv lands in
# slgcn/popsign/results/lsc_eval/<run_name>/.
#
# Submitted by ../run_inference_all.sh from common/lsc_eval/ (SLURM_SUBMIT_DIR).

ARM="${1:?usage: sbatch slgcn_popsign_arm.sbatch <arm-key>}"
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO="$(cd "$HERE/../.." && pwd)"
MANIFEST="$HERE/inference/manifest.py"

WEIGHTS="$(python3 "$MANIFEST" "$ARM" checkpoint)"
RUN_NAME="$(python3 "$MANIFEST" "$ARM" run_name)"
FLAGS="$(python3 "$MANIFEST" "$ARM" flags)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate popsign-slgcn-env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

nvidia-smi -L || true

LEAF="$REPO/slgcn/popsign"
SLGCN="$LEAF/external/SAM-SLR-v2/SL-GCN"
[ -f "$WEIGHTS" ] || { echo "MISSING weights: $WEIGHTS" >&2; exit 1; }
[ -d "$SLGCN" ]   || { echo "MISSING SL-GCN clone: $SLGCN" >&2; exit 1; }

cd "$SLGCN"
export PYTHONPATH="$LEAF/04_phase2_multihead:$LEAF:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # $PWD = the clone (upstream model/, feeders/, graph/)

# Both decoding variants: the paper decodes constrained throughout, so an
# arm that only ever ran unconstrained cannot appear in the main LSC
# table. --constrain-lsc is a decoding choice over the SAME inputs; the
# eval script appends _cons to the run name, so the two passes never
# collide. The split comes from this arm's manifest FLAGS.
for cons in "" "--constrain-lsc"; do
  echo "=== $ARM: ${RUN_NAME} / ${cons:-plain} ==="
  python "$LEAF/05_lsc_eval/eval_lsc_handshapes.py" \
    --weights "$WEIGHTS" --run-name "$RUN_NAME" ${FLAGS} ${cons}
done
echo "$ARM DONE"
