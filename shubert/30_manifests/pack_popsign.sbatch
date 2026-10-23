#!/bin/bash
#SBATCH --job-name=slsh_pack_popsign
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A.out

# Pack the popsign16 splits into memmap blobs (same reason as Sem-Lex: 125
# epochs x 143k clips x 4 files would be ~70M small-file GPFS opens).
# Sources are the precomputed PopSign SHuBERT feature streams under
# $ASL2LSC_POPSIGN_PREPROC/popsign (read-only), addressed via the manifests
# written by 10_data_prep/make_popsign_manifests.py.
# NOTE: `set -u` must come AFTER conda activate — the shared env's MKL
# activation hook reads $MKL_INTERFACE_LAYER unset and would abort the job.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eu

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
python "${REPO}/30_manifests/pack_features.py" \
    --split popsign16_train --split popsign16_val --split popsign16_test
echo "PACK POPSIGN DONE"
