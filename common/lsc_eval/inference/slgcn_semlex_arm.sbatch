#!/bin/bash
#SBATCH --job-name=lsc_infer_slgcn_sem
#SBATCH --output=logs/infer_slgcn_sem_%j.out
#SBATCH --error=logs/infer_slgcn_sem_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# One SL-GCN/Sem-Lex checkpoint -> LSC-benchmark predictions across the
# reformatting-variant ladder. Runs the leaf's own eval script from inside
# the external SAM-SLR-v2 clone (upstream modules resolve relative to CWD);
# needs the pre-packed gendata tensors for each variant to exist already
# (slgcn/semlex/04_phase2_multihead/lsc_handshapes/prepare_lsc_handshapes.py).
# Per-clip CSVs land in slgcn/semlex/results/<run_name>/.
#
# Submitted by ../run_inference_all.sh from common/lsc_eval/ (SLURM_SUBMIT_DIR).

ARM="${1:?usage: sbatch slgcn_semlex_arm.sbatch <arm-key>}"
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO="$(cd "$HERE/../.." && pwd)"
MANIFEST="$HERE/inference/manifest.py"

CKPT="$(python3 "$MANIFEST" "$ARM" checkpoint)"
REGIME="$(python3 "$MANIFEST" "$ARM" regime pfr)"
RUN_NAME="$(python3 "$MANIFEST" "$ARM" run_name)"
VARIANTS="$(python3 "$MANIFEST" "$ARM" variants none)"
# optional: 6-head (heads6) checkpoints need their own vocab to rebuild the model
FVOCAB="$(python3 "$MANIFEST" "$ARM" feature_vocab "")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slgcn-env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

nvidia-smi -L || true

LEAF="$REPO/slgcn/semlex"
SLGCN="$LEAF/external/SAM-SLR-v2/SL-GCN"
[ -f "$CKPT" ]  || { echo "MISSING checkpoint: $CKPT" >&2; exit 1; }
[ -d "$SLGCN" ] || { echo "MISSING SL-GCN clone: $SLGCN" >&2; exit 1; }

cd "$SLGCN"
export PYTHONPATH="$LEAF/04_phase2_multihead:$LEAF:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # $PWD = the clone (upstream model/, feeders/, graph/)

# Both decoding variants: the paper decodes constrained throughout, so an
# arm that only ever ran unconstrained cannot appear in the main LSC
# table. --constrain-lsc is a decoding choice over the SAME inputs; the
# eval script appends its own _cons output tag, so the two passes never
# collide.
for split in test examples; do
  for variant in $VARIANTS; do
    VFLAG=(); [ "$variant" != "none" ] && VFLAG=(--variant "$variant")
    for cons in "" "--constrain-lsc"; do
      echo "=== $ARM: ${RUN_NAME} / ${split} / ${variant} / ${cons:-plain} ==="
      python "$LEAF/04_phase2_multihead/lsc_handshapes/eval_lsc_handshapes.py" \
        --checkpoint "$CKPT" \
        --regime "$REGIME" --split "$split" \
        --run-name "$RUN_NAME" "${VFLAG[@]+"${VFLAG[@]}"}" \
        ${cons} \
        ${FVOCAB:+--feature-vocab "$FVOCAB"}
    done
  done
done
echo "$ARM DONE"
