#!/bin/bash
#SBATCH --job-name=popsign_dataprep
#SBATCH --output=logs/dataprep_%j.out
#SBATCH --error=logs/dataprep_%j.err
# 8h is the `medium` cap (a 12h request sits in PartitionTimeLimit forever).
# Ample: the smoke run converted 120 clips in 11 s on 8 workers, so the full
# 174k on 16 workers is well under an hour; the headroom covers gendata's
# single-threaded packing of the 110k-clip train split.
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=medium

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate popsign-slgcn-env
set -euo pipefail
cd "$REPO/01_data_prep"

echo "=== 1/7 build_manifest ==="
python build_manifest.py
echo "=== 2/7 derive_vocab ==="
python derive_vocab.py
echo "=== 3/7 build_splits ==="
python build_splits.py
echo "=== 4/7 convert_poses (parquet -> wholebody133 npy) ==="
python convert_poses.py --workers "$SLURM_CPUS_PER_TASK" ${CONVERT_ARGS:-}
# PopSign ships at least one truncated parquet. convert_poses tolerates a few
# (--max-failures) and lists them; this pass drops those clips from the label
# CSVs so upstream gendata never tries to load an npy that was never written.
# No-op when nothing failed.
echo "=== 5/7 build_splits --exclude-failures ==="
python build_splits.py --exclude-failures
echo "=== 6/7 run_gendata (upstream sign_gendata.py) ==="
python run_gendata.py > /dev/null   # upstream prints one line per clip
echo "=== 7/7 overlay QA + mapping table ==="
python map_543_to_wholebody133.py --dump-table > /dev/null
python verify_skeleton_overlay.py --n-clips 4
echo "DONE — inspect docs/overlay_qa/*.png before training"
