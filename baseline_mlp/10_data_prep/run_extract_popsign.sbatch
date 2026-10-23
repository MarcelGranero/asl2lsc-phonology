#!/bin/bash
#SBATCH --job-name=mlp_extract_popsign
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/extract_popsign_%j.out
#SBATCH --error=logs/extract_popsign_%j.err

# One static hand frame per PopSign clip, in the 63-float format the baseline
# MLP already consumes for Sem-Lex. CPU-only: this is npy reads plus a
# per-frame argmin, no model involved.
#
# Reads 27 GB of wholebody-133 npys over ~174k files. One file per clip, so
# it parallelises directly: the script pools over SLURM_CPUS_PER_TASK and
# emits rows in clip-id order, so the output does NOT depend on the worker
# count (asserted: 1 vs 8 workers give byte-identical npz). Measured on 3000
# clips: 13.1 s serial, 3.7 s at 4 workers, 2.2 s at 16. The first
# (single-threaded) run took ~50 min for the full corpus; 16 workers brings
# that under 10, which fits the 2 h short partition comfortably.
#
# Submit from baseline_mlp/:  mkdir -p logs && sbatch 10_data_prep/run_extract_popsign.sbatch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

cd "${SLURM_SUBMIT_DIR}"

python 10_data_prep/extract_frames_popsign.py --all

echo "POPSIGN FRAME EXTRACTION DONE"
