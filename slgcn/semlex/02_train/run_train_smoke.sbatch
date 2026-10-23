#!/bin/bash
#SBATCH --job-name=semlex_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# 2-epoch smoke run of upstream main.py on the first 100 samples
# (upstream's own feeder debug flag). Flushes out torch-compat issues.

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
SLGCN="$REPO/external/SAM-SLR-v2/SL-GCN"
PROCESSED=${ASL2LSC_SLGCN_SEMLEX_DATA}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slgcn-env
set -euo pipefail
CONFIG="$REPO/02_train/config/train_joint_smoke.yaml"

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

mkdir -p "$PROCESSED/save_models" "$PROCESSED/work_dir"
ln -sfn "$PROCESSED/save_models" "$SLGCN/save_models"
ln -sfn "$PROCESSED/work_dir" "$SLGCN/work_dir"

cd "$SLGCN"
time python main.py --config "$RENDERED"
echo "SMOKE OK"
