#!/bin/bash
# Regenerate the predictions TSVs and metrics under common/lsc_eval/ from
# the per-leaf per-clip outputs of the released checkpoints (produced by
# run_inference_all.sh). Idempotent; run from anywhere; plain python3
# (stdlib only), no conda env needed.
#
# Arm name scheme: <family>_<corpus>[_h6]_<split>_<decoding>; one released
# checkpoint per <family>_<corpus> (inference/checkpoints.json).
set -eo pipefail
cd "$(dirname "$0")"
REPO=../..

# arm name                               per-clip CSV
ARMS='
shubert_semlex_test_unconstrained        shubert/results/lsc_eval_v2/lsc_eval_semlex_main_test_per_clip.csv
shubert_semlex_test_constrained          shubert/results/lsc_eval_v2/lsc_eval_semlex_main_cons_test_per_clip.csv
shubert_semlex_examples_unconstrained    shubert/results/lsc_eval_v2/lsc_eval_semlex_main_examples_per_clip.csv
shubert_semlex_examples_constrained      shubert/results/lsc_eval_v2/lsc_eval_semlex_main_cons_examples_per_clip.csv
shubert_popsign_test_unconstrained       shubert/results/lsc_eval_v2/lsc_eval_popsign_main_test_per_clip.csv
shubert_popsign_test_constrained         shubert/results/lsc_eval_v2/lsc_eval_popsign_main_cons_test_per_clip.csv
shubert_popsign_examples_unconstrained   shubert/results/lsc_eval_v2/lsc_eval_popsign_main_examples_per_clip.csv
shubert_popsign_examples_constrained     shubert/results/lsc_eval_v2/lsc_eval_popsign_main_cons_examples_per_clip.csv
slgcn_semlex_test_unconstrained          slgcn/semlex/results/semlex_pfr_repose/lsc_handshapes_test_per_clip.csv
slgcn_semlex_test_constrained            slgcn/semlex/results/semlex_pfr_repose/lsc_handshapes_test_cons_per_clip.csv
slgcn_semlex_examples_unconstrained      slgcn/semlex/results/semlex_pfr_repose/lsc_handshapes_examples_per_clip.csv
slgcn_semlex_examples_constrained        slgcn/semlex/results/semlex_pfr_repose/lsc_handshapes_examples_cons_per_clip.csv
slgcn_popsign_test_unconstrained         slgcn/popsign/results/lsc_eval/popsign_pfr16_test/predictions.csv
slgcn_popsign_test_constrained           slgcn/popsign/results/lsc_eval/popsign_pfr16_test_cons/predictions.csv
slgcn_popsign_examples_unconstrained     slgcn/popsign/results/lsc_eval/popsign_pfr16_examples/predictions.csv
slgcn_popsign_examples_constrained       slgcn/popsign/results/lsc_eval/popsign_pfr16_examples_cons/predictions.csv
slgcn_popsign_h6_test_unconstrained      slgcn/popsign/results/lsc_eval/popsign_pfr_h6_test/predictions.csv
slgcn_popsign_h6_test_constrained        slgcn/popsign/results/lsc_eval/popsign_pfr_h6_test_cons/predictions.csv
slgcn_popsign_h6_examples_unconstrained  slgcn/popsign/results/lsc_eval/popsign_pfr_h6_examples/predictions.csv
slgcn_popsign_h6_examples_constrained    slgcn/popsign/results/lsc_eval/popsign_pfr_h6_examples_cons/predictions.csv
mlp_semlex_test_unconstrained            baseline_mlp/results/lsc_v2/lsc_test_all16_per_clip.csv
mlp_semlex_test_constrained              baseline_mlp/results/lsc_v2/lsc_test_all16_cons_per_clip.csv
mlp_semlex_examples_unconstrained        baseline_mlp/results/lsc_v2/lsc_examples_all16_per_clip.csv
mlp_semlex_examples_constrained          baseline_mlp/results/lsc_v2/lsc_examples_all16_cons_per_clip.csv
mlp_popsign_test_unconstrained           baseline_mlp/results/lsc_v2/lsc_test_popsign_all16_nac_s42_per_clip.csv
mlp_popsign_test_constrained             baseline_mlp/results/lsc_v2/lsc_test_popsign_all16_nac_s42_cons_per_clip.csv
mlp_popsign_examples_unconstrained       baseline_mlp/results/lsc_v2/lsc_examples_popsign_all16_nac_s42_per_clip.csv
mlp_popsign_examples_constrained         baseline_mlp/results/lsc_v2/lsc_examples_popsign_all16_nac_s42_cons_per_clip.csv
'

mkdir -p predictions results
# predictions/ and results/ are generated from ARMS; clear both so a
# renamed arm never leaves its old files behind.
rm -f predictions/*.tsv results/*.metrics.json results/*.per_clip.csv

# An arm whose per-clip CSV is not on disk yet is SKIPPED with a warning:
# run the inference first (run_inference_all.sh), or score just the arms
# you have.
echo "$ARMS" | while read -r name src; do
    [ -z "$name" ] && continue
    if [ ! -f "$REPO/$src" ]; then
        echo "SKIP $name: no per-clip CSV yet ($src)" >&2
        continue
    fi
    python3 convert_legacy.py "$REPO/$src" -o "predictions/$name.tsv"
done

python3 evaluate.py predictions/*.tsv --out-dir results >/dev/null
echo "metrics written to $(pwd)/results/"
