#!/bin/bash
#SBATCH --job-name=semlex_prep_repose
#SBATCH --output=logs/prep_repose_%j.out
#SBATCH --error=logs/prep_repose_%j.err
#SBATCH --time=07:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium

# Data prep for the RE-POSED run: reposed 553-npys -> wholebody133 -> packed
# (N,3,150,27,1) tensors, all in their own trees so the released-pose run's
# artifacts are never touched.
#
# Vocabulary and splits are NOT re-derived — the run reuses the frozen 1,914
# classes and the same val/test clips, so the only difference from phase 1 is
# the pose source (and 2,401 train clips with no source video; see
# results/reposed_labels_report.json).

REPO="${SLURM_SUBMIT_DIR}"   # submit from this leaf's root dir
PROCESSED=${ASL2LSC_SLGCN_SEMLEX_DATA}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slgcn-env
set -euo pipefail
cd "$REPO/01_data_prep"

echo "=== 1/3 label CSVs (available clips only) ==="
python build_reposed_labels.py

echo "=== 2/3 convert reposed poses -> wholebody133 ==="
python convert_poses.py --workers "$SLURM_CPUS_PER_TASK" --skip-absent \
    --poses-dir "$PROCESSED/reposed" \
    --out-dir "$PROCESSED/wholebody133_reposed"

echo "=== 3/3 upstream gendata pack ==="
python run_gendata.py \
    --wholebody-dir "$PROCESSED/wholebody133_reposed" \
    --labels-dir "$PROCESSED/labels_reposed" \
    --out-dir "$PROCESSED/gendata_27_reposed" > /dev/null

echo "=== hand presence in the PACKED tensors (the point of the experiment) ==="
python - <<'EOF'
import numpy as np
from pathlib import Path
# nodes 7..26 of the 27-joint layout are the two hands; channel 3 is presence
for split in ["train", "val", "test"]:
    for tag, d in (("released", "gendata_27"), ("reposed", "gendata_27_reposed")):
        p = Path(f"${ASL2LSC_SLGCN_SEMLEX_DATA}/{d}/{split}_data_joint.npy")
        if not p.exists():
            continue
        a = np.load(p, mmap_mode="r")[:2000, 2, :, 7:27, 0]   # (N,T,20) presence
        print(f"{split:5s} {tag:8s} hand-joint presence {float(np.asarray(a).mean()):.3f}")
EOF
echo "REPOSED DATA PREP DONE"
