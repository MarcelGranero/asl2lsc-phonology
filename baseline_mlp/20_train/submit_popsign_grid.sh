#!/bin/bash
# Submit the PopSign baseline-MLP grid: both head sets x three seeds, on
# the NA-as-class labels (the trainer default). Run names carry the _nac
# token.
#
#   all16   the SL-GCN / SHuBERT Table-9 head list (16 heads)
#   heads6  Handshape + the 5 attributes the LSC benchmark scores
#
# Seeds 42/43/44 match the Sem-Lex seed-variance runs, so the spread is
# measurable on both corpora and the two are directly comparable in METHOD --
# though NOT head-for-head in numbers: PopSign's label space differs from
# Sem-Lex's in 11 of the 16 heads (its vocabulary comes from a 250-gloss
# subset), so a PopSign head is a different classification problem.
#
# Each run is seeded end-to-end (numpy + torch + CUDA) and records the seed
# in best.pt's config, so a rerun is reproducible in intent; GPU kernels make
# it not bit-exact.
#
# Usage (from baseline_mlp/):
#   mkdir -p logs && bash 20_train/submit_popsign_grid.sh
#   bash 20_train/submit_popsign_grid.sh --dry-run
#   DEPENDENCY=afterok:<jobid> bash 20_train/submit_popsign_grid.sh
#     (chain the grid behind e.g. the naclass label rebuild)

set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1
DEP_ARGS=()
[[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS=(--dependency="$DEPENDENCY")

FRAMES=${ASL2LSC_MLP_DATA}/popsign_frames
for split in train val test; do
  if [[ ! -f "$FRAMES/$split.npz" ]]; then
    echo "MISSING $FRAMES/$split.npz — run 10_data_prep/run_extract_popsign.sbatch first" >&2
    exit 1
  fi
done

mkdir -p logs
for HEAD_SET in all16 heads6; do
  for SEED in 42 43 44; do
    RUN="popsign_${HEAD_SET}_nac_s${SEED}"
    if [[ -n "$DRY" ]]; then
      echo "would submit: $RUN (head-set $HEAD_SET, seed $SEED, corpus popsign, NA-as-class)"
      continue
    fi
    sbatch --job-name="$RUN" "${DEP_ARGS[@]+"${DEP_ARGS[@]}"}" \
      20_train/run_train.sbatch \
      "$HEAD_SET" "$RUN" multi "$SEED" popsign
  done
done
