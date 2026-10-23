#!/bin/bash
#SBATCH --job-name=labels16_naclass
#SBATCH --output=logs/labels16_naclass_%j.out
#SBATCH --error=logs/labels16_naclass_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=short

# Rebuild the 16-feature PopSign labels with NA as a real class (the
# default policy). Writes NEW files so the legacy NA-masked build stays on
# disk for decoding pre-existing checkpoints:
#   feature_labels16_naclass.pkl   (PROCESSED_DIR)
#   feature_vocab16_naclass.json   (results/)
# Downstream consumers (baseline_mlp popsign runs, shubert popsign16
# manifests, slgcn popsign 16-head pfr) all read these new files.
#
# Submit from slgcn/popsign/:  mkdir -p logs && \
#   sbatch 01_data_prep/run_labels16_naclass.sbatch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate popsign-slgcn-env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROCESSED=${ASL2LSC_SLGCN_POPSIGN_DATA}

cd "$REPO"
python 01_data_prep/build_feature_labels16.py \
  --null-policy class \
  --out-pkl "$PROCESSED/feature_labels16_naclass.pkl" \
  --out-vocab results/feature_vocab16_naclass.json

echo "LABELS16 NACLASS DONE"
