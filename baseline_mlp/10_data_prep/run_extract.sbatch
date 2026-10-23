#!/bin/bash
#SBATCH --job-name=mlp_extract
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# NOTE: no `set -u` around conda activate -- shubert_train_env's
# activate.d/libblas_mkl_activate.sh reads $MKL_INTERFACE_LAYER unbound and
# aborts the job under `set -u`. Strictness is re-enabled afterwards.
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u
cd "${SLURM_SUBMIT_DIR}"   # submit from baseline_mlp/; mkdir -p logs first

python 10_data_prep/extract_frames.py --splits train val test
