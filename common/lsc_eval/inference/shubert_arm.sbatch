#!/bin/bash
#SBATCH --job-name=lsc_infer_shubert
#SBATCH --output=logs/infer_shubert_%j.out
#SBATCH --error=logs/infer_shubert_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# One SHuBERT checkpoint -> LSC-benchmark predictions, parameterised by an
# arm key in inference/checkpoints.json. Runs the leaf's own eval script
# (shubert/50_eval/lsc_handshapes/eval_shubert_lsc.py) over the full grid
# splits x rates x {plain, constrained}; per-clip CSVs land in
# shubert/results/lsc_eval_v2/ where ../run_all.sh picks them up.
#
# Submitted by ../run_inference_all.sh from common/lsc_eval/ (SLURM_SUBMIT_DIR).

ARM="${1:?usage: sbatch shubert_arm.sbatch <arm-key>}"
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO="$(cd "$HERE/../.." && pwd)"
MANIFEST="$HERE/inference/manifest.py"

CKPT="$(python3 "$MANIFEST" "$ARM" checkpoint)"
RUN_LABEL="$(python3 "$MANIFEST" "$ARM" run_label)"
RATES="$(python3 "$MANIFEST" "$ARM" rates "50")"
# optional alternative inputs: manifest-filename tokens,
# e.g. "vidcrop vidcropphone"; "" (always run) is the uncropped ladder
VARIANTS="$(python3 "$MANIFEST" "$ARM" variants "")"
LABEL_MAPS="$(python3 "$MANIFEST" "$ARM" label_maps)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

nvidia-smi -L || true

# Gold labels + per-split feature manifests ship with the LSC benchmark.
V2="${ASL2LSC_LSC_LABELS:-$REPO/shubert/50_eval/lsc_labels_v2}"
[ -f "$V2/test_labels_audit_v2.csv" ] || {
  echo "no LSC gold labels under $V2 -- set ASL2LSC_LSC_LABELS" >&2; exit 1; }
OUT="$REPO/shubert/results/lsc_eval_v2"
EVAL="$REPO/shubert/50_eval/lsc_handshapes/eval_shubert_lsc.py"

[ -f "$CKPT" ] || { echo "MISSING checkpoint: $CKPT" >&2; exit 1; }
mkdir -p "$OUT"

EXTRA=()
[ -n "$LABEL_MAPS" ] && EXTRA+=(--label-maps "$LABEL_MAPS")

for split in test examples; do
  for variant in "" $VARIANTS; do
    vsuf=""; [ -n "$variant" ] && vsuf="_${variant}"
    for rate in $RATES; do
      suffix=""
      [ "$rate" = "25" ] && suffix="_25fps"
      [ "$rate" = "17" ] && suffix="_17fps"
      MAN="$V2/shubert_inference_handshape_${split}${vsuf}${suffix}_v2.csv"
      [ -f "$MAN" ] || { echo "SKIP missing manifest $MAN" >&2; continue; }
      for cons in "" "--constrain-lsc"; do
        tag="$RUN_LABEL"
        [ -n "$cons" ] && tag="${tag}_cons"
        tag="${tag}${vsuf}${suffix}"
        echo "=== $ARM: ${tag} / ${split} ==="
        python "$EVAL" \
          --ckpt "$CKPT" \
          --split "$split" \
          --manifest "$MAN" \
          --audit "$V2/test_labels_audit_v2.csv" \
          --run-label "$tag" \
          --out-dir "$OUT" ${cons} "${EXTRA[@]+"${EXTRA[@]}"}"
      done
    done
  done
done
echo "$ARM DONE"
