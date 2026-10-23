#!/bin/bash
# Zero-shot cross-lingual LSC evaluation, end to end: submit one SLURM
# inference job per arm in inference/checkpoints.json (all of them by
# default, or just the arm keys given as arguments), then one gate job that
# re-scores every tracked arm through the shared evaluator (run_all.sh).
#
#   bash run_inference_all.sh                        # every manifest arm
#   bash run_inference_all.sh mlp_semlex             # one checkpoint
#   bash run_inference_all.sh --score-only           # skip inference, just
#                                                    # submit the gate
#
# Each family runs in its own job with its own conda env and resources
# (slgcn needs a GPU unconditionally, the MLP is CPU-only on purpose), so a
# slow or failed family never blocks the others; the gate uses afterany and
# scores whatever is on disk.
#
# Plugging in a NEW checkpoint:
#   1. add one entry to inference/checkpoints.json (family + checkpoint +
#      the leaf flags -- see the _readme key there);
#   2. if its outputs should become a tracked arm, add one ARMS line in
#      run_all.sh mapping the arm name to the leaf's per-clip CSV;
#   3. bash run_inference_all.sh <new-key>
#
# Entries whose checkpoint is not on disk yet (e.g. a training chain still
# running) are skipped with a warning.
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs

if [ "${1:-}" = "--score-only" ]; then
  gate=$(sbatch --parsable inference/score_gate.sbatch)
  echo "gate (re-score) -> job $gate"
  exit 0
fi

KEYS="${*:-$(python3 inference/manifest.py --list)}"
ids=()
for key in $KEYS; do
  fam=$(python3 inference/manifest.py "$key" family)
  ckpt=$(python3 inference/manifest.py "$key" checkpoint)
  skip=$(python3 inference/manifest.py "$key" skip)
  if [ -n "$skip" ]; then
    echo "SKIP $key: $skip"
    continue
  fi
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "SKIP $key: checkpoint not on disk yet ($ckpt)"
    continue
  fi
  wrapper="inference/${fam}_arm.sbatch"
  [ -f "$wrapper" ] || { echo "unknown family '$fam' for $key" >&2; exit 1; }
  id=$(sbatch --parsable "$wrapper" "$key")
  echo "submitted $key -> job $id ($fam)"
  ids+=("$id")
done

[ "${#ids[@]}" -gt 0 ] || { echo "nothing submitted" >&2; exit 1; }
dep=$(IFS=,; echo "${ids[*]}")
# INFER_JOBS travels through the environment: inside --export=... the
# commas of the job list would split it into separate variables.
gate=$(INFER_JOBS="${dep}" sbatch --parsable --dependency="afterany:${dep}" \
       --export=ALL inference/score_gate.sbatch)
echo "gate (re-score) -> job ${gate} (afterany:${dep})"
echo "watch:  squeue -u \$USER"
