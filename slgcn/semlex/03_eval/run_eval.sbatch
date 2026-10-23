#!/bin/bash
#SBATCH --job-name=semlex_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# 1. pick the best-val-epoch checkpoint from the training log
# 2. run UPSTREAM main.py --phase test on the test split (saves score pickle)
# 3. compute top-1/3/5, per-signer metrics, run card, paper comparison
#
# POSE SET MUST MATCH THE CHECKPOINT. The default CONFIG scores the
# RELEASED-pose tensors (gendata_27). A checkpoint trained on reposed data
# must be scored with CONFIG=03_eval/config/test_joint_semlex_reposed.yaml
# (gendata_27_reposed) -- nothing here can detect the mismatch, and it does
# not crash: it silently reports a train/test-distribution-mismatch number
# (the mismatch looks like a WORSE model, not a broken eval). Same trap as
# eval_multihead.py's --gendata-dir.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
PROCESSED=${ASL2LSC_SLGCN_SEMLEX_DATA}
CONFIG="${CONFIG:-$REPO/03_eval/config/test_joint_semlex.yaml}"
WEIGHTS="${WEIGHTS:-}"

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

mkdir -p "$PROCESSED/save_models" "$PROCESSED/work_dir"
ln -sfn "$PROCESSED/save_models" "$SLGCN/save_models"
ln -sfn "$PROCESSED/work_dir" "$SLGCN/work_dir"

if [[ -z "$WEIGHTS" ]]; then
    WEIGHTS=$(python "$REPO/03_eval/score_semlex.py" find-best | tail -1)
fi
# RUN_NAME keeps mid-training probes out of the final eval's results dir
RUN_NAME="${RUN_NAME:-slgcn_joint}"
TEST_EXP=$(grep -Po '^Experiment_name:\s*\K\S+' "$CONFIG")
echo "evaluating $WEIGHTS -> results/$RUN_NAME"

cd "$SLGCN"
# Keep only this run's scores. upstream main.py opens
# work_dir/<exp>/eval_results/*.pkl for writing but only ever mkdirs it in
# save_arg() during a TRAIN run, so the dir must exist before a test phase —
# clearing it without recreating crashes a test run after its full,
# successful inference pass.
rm -rf "work_dir/$TEST_EXP/eval_results"
mkdir -p "work_dir/$TEST_EXP/eval_results"
python main.py --config "$RENDERED" --weights "$WEIGHTS"

python "$REPO/03_eval/score_semlex.py" score --weights "$WEIGHTS" \
    --run-name "$RUN_NAME" --test-exp "$TEST_EXP"
echo "EVAL DONE"
