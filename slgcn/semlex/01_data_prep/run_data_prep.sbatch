#!/bin/bash
#SBATCH --job-name=semlex_dataprep
#SBATCH --output=logs/dataprep_%j.out
#SBATCH --error=logs/dataprep_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slgcn-env
set -euo pipefail
cd "$REPO/01_data_prep"

echo "=== 1/6 dedupe_metadata ==="
python dedupe_metadata.py
echo "=== 2/6 derive_vocab ==="
python derive_vocab.py
echo "=== 3/6 build_splits ==="
python build_splits.py
echo "=== 4/6 convert_poses ==="
python convert_poses.py --workers "$SLURM_CPUS_PER_TASK" ${CONVERT_ARGS:-}
echo "=== 5/6 run_gendata (upstream sign_gendata.py) ==="
python run_gendata.py > /dev/null   # upstream prints one line per clip
echo "=== 6/6 overlay QA + mapping table ==="
python map_553_to_wholebody133.py --dump-table > /dev/null
python verify_skeleton_overlay.py --n-clips 4
echo "DONE"
