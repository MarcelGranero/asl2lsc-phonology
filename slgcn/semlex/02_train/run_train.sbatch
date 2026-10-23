#!/bin/bash
#SBATCH --job-name=semlex_slgcn
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=high

# Runs UPSTREAM SAM-SLR-v2 main.py unmodified (compat patch aside).
#
# Resume: picks the newest save_models/<EXP>-<epoch>.pt and continues via
# upstream's own --start-epoch/--weights (upstream keeps no optimizer state,
# so momentum restarts at each resume — documented deviation-free behaviour).
#
# Walltime chaining: BEFORE training starts, a follow-up job is submitted with
# --dependency=afterany on this one; when a job is walltime-killed the
# follow-up resumes from the last epoch checkpoint. When training is complete
# a job exits before submitting anything, ending the chain. CHAIN caps runaway
# crash-loops.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
PROCESSED=${ASL2LSC_SLGCN_SEMLEX_DATA}
CONFIG="${CONFIG:-$REPO/02_train/config/train_joint_semlex.yaml}"
CHAIN="${CHAIN:-0}"
MAX_CHAIN=15
EXP=$(grep -Po '^Experiment_name:\s*\K\S+' "$CONFIG")
NUM_EPOCH=$(grep -Po '^num_epoch:\s*\K\d+' "$CONFIG")

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slgcn-env
set -euo pipefail

# The configs name their data through ${ASL2LSC_*} roots and upstream main.py
# reads YAML verbatim, so it gets an expanded per-job copy (CONFIG itself
# stays the source file, which is what a chained follow-up leg needs).
RENDERED="${TMPDIR:-/tmp}/slgcn_config_${SLURM_JOB_ID:-$$}.yaml"
python - "$CONFIG" "$RENDERED" <<'EOF'
import os, sys
text = os.path.expandvars(open(sys.argv[1]).read())
if "${ASL2LSC_" in text:
    sys.exit(f"{sys.argv[1]}: unset ASL2LSC_* variable in the config")
open(sys.argv[2], "w").write(text)
EOF
nvidia-smi -L || true

# heavy run artifacts live on projects, reached via symlinks from the
# clone-relative paths main.py hardcodes (./save_models, ./work_dir)
mkdir -p "$PROCESSED/save_models" "$PROCESSED/work_dir"
ln -sfn "$PROCESSED/save_models" "$SLGCN/save_models"
ln -sfn "$PROCESSED/work_dir" "$SLGCN/work_dir"

# config num_class must match the vocab this run was built from.
# VOCAB_JSON pins a run to its own frozen vocab, so a resume of an older run
# is not validated against (or silently retrained on) a newer rebuilt vocab.
VOCAB_JSON="${VOCAB_JSON:-$REPO/results/vocab.json}"
python - "$CONFIG" "$VOCAB_JSON" <<'EOF'
import sys, yaml, json, pathlib
cfg = yaml.safe_load(open(sys.argv[1]))
vocab = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert cfg["model_args"]["num_class"] == vocab["n_classes"], (
    f"config num_class {cfg['model_args']['num_class']} != "
    f"vocab n_classes {vocab['n_classes']} ({sys.argv[2]})")
EOF

cd "$SLGCN"

RESUME_ARGS=()
LATEST=$(ls -1 "save_models/$EXP"-*.pt 2>/dev/null \
         | sed -E 's/.*-([0-9]+)\.pt/\1 &/' | sort -n | tail -1 || true)
if [[ -n "${LATEST:-}" ]]; then
    EPOCH=$(echo "$LATEST" | cut -d' ' -f1)
    WEIGHTS=$(echo "$LATEST" | cut -d' ' -f2-)
    START=$((EPOCH + 1))
    if (( START >= NUM_EPOCH )); then
        echo "training already complete ($WEIGHTS) — ending chain"; exit 0
    fi
    echo "resuming from $WEIGHTS (start epoch $START)"
    RESUME_ARGS=(--start-epoch "$START" --weights "$WEIGHTS")
fi

if (( CHAIN < MAX_CHAIN )); then
    FOLLOWUP=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,CONFIG="$CONFIG",CHAIN=$((CHAIN + 1)) \
        "$REPO/02_train/run_train.sbatch")
    echo "follow-up job $FOLLOWUP submitted (chain $((CHAIN + 1)))"
else
    echo "chain cap $MAX_CHAIN reached — no follow-up submitted"
fi

python main.py --config "$RENDERED" "${RESUME_ARGS[@]}"
echo "main.py exited cleanly"
