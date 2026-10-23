#!/bin/bash
#SBATCH --job-name=lsc_infer_mlp
#SBATCH --output=logs/infer_mlp_%j.out
#SBATCH --error=logs/infer_mlp_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short

# One Baseline-MLP run -> LSC-benchmark predictions (CPU on purpose: a
# 2-layer MLP over 37 clips is seconds of compute). The eval script picks the label map from the
# checkpoint's own `corpus` field, so Sem-Lex- and PopSign-trained runs both
# work; per-clip CSVs land in baseline_mlp/results/lsc_v2/.
#
# Submitted by ../run_inference_all.sh from common/lsc_eval/ (SLURM_SUBMIT_DIR).

ARM="${1:?usage: sbatch mlp_arm.sbatch <arm-key>}"
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO="$(cd "$HERE/../.." && pwd)"
MANIFEST="$HERE/inference/manifest.py"

RUN_NAME="$(python3 "$MANIFEST" "$ARM" run_name)"
# optional: an explicit checkpoint path (released-checkpoint arms); without
# it the eval resolves runs/<run_name>/best.pt from the leaf's RUNS_DIR
CKPT="$(python3 "$MANIFEST" "$ARM" checkpoint "")"
# optional alternative pose trees for the same clips (pose553_<variant>/)
VARIANTS="$(python3 "$MANIFEST" "$ARM" variants "")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

cd "$REPO/baseline_mlp"
# Gold labels + per-split feature manifests ship with the LSC benchmark.
V2="${ASL2LSC_LSC_LABELS:-$REPO/shubert/50_eval/lsc_labels_v2}"
[ -f "$V2/test_labels_audit_v2.csv" ] || {
  echo "no LSC gold labels under $V2 -- set ASL2LSC_LSC_LABELS" >&2; exit 1; }
OUT="results/lsc_v2"
mkdir -p "$OUT"

POSE553=${ASL2LSC_SLGCN_SEMLEX_DATA}/lsc_handshapes

for split in test examples; do
  MAN="$V2/shubert_inference_handshape_${split}_v2.csv"
  for variant in "" $VARIANTS; do
    VFLAG=()
    if [ -n "$variant" ]; then
      # the crop trees are built by the SL-GCN leaf; skip rather than fail
      # so an MLP arm still runs its native cells before that lands
      [ -d "$POSE553/pose553_${variant}/${split}" ] || {
        echo "SKIP missing pose tree pose553_${variant}/${split}" >&2
        continue
      }
      VFLAG=(--variant "$variant")
    fi
    for extra in "" "--constrain-lsc"; do
      echo "=== $ARM: ${RUN_NAME} / ${split} / ${variant:-native} ${extra:-plain} ==="
      python 30_eval/eval_lsc_handshapes.py \
        --run-name "$RUN_NAME" --split "$split" \
        --manifest "$MAN" --audit "$V2/test_labels_audit_v2.csv" \
        --out-dir "$OUT" ${extra} "${VFLAG[@]+"${VFLAG[@]}"}" \
        ${CKPT:+--checkpoint "$CKPT"}
    done
  done
done
echo "$ARM DONE"
