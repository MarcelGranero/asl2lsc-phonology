#!/bin/bash
#SBATCH --job-name=slsh_verify
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A.out

# Verify gate between pipeline stages. env: SPLIT, SMOKE, STAGE
# (transcode|kpe|crops|features). Submitted afterany:<stage array>; the next
# stage depends afterok on THIS job, so a failed check here halts the chain.
# After a passing check it (re)builds the input lists the next stage needs.
set -eu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_feature_extraction

REPO="${SLURM_SUBMIT_DIR}"   # the shubert/ leaf (submit_pipeline.sh submits from it)
DATA=${ASL2LSC_SHUBERT_DATA}
PREFIX="${SPLIT}"
LIMIT=0
if [ "${SMOKE:-0}" = "1" ]; then PREFIX="${SPLIT}_smoke"; LIMIT=200; fi
VLIST="${DATA}/work_lists/${PREFIX}_videos.list.gz"

case "${STAGE}" in
  transcode)
    python "${REPO}/20_preprocess/verify_stage.py" \
        --stage transcode --split "${SPLIT}" --limit "${LIMIT}"
    python "${REPO}/10_data_prep/make_work_lists.py" \
        --split "${SPLIT}" --limit "${LIMIT}"
    ;;
  kpe|crops)
    python "${REPO}/20_preprocess/verify_stage.py" \
        --stage "${STAGE}" --split "${SPLIT}" --videos-list "${VLIST}"
    python "${REPO}/20_preprocess/make_stage_lists.py" \
        --split "${SPLIT}" --videos-list "${VLIST}"
    ;;
  features)
    python "${REPO}/20_preprocess/verify_stage.py" \
        --stage features --split "${SPLIT}" --videos-list "${VLIST}"
    echo "PIPELINE COMPLETE for ${PREFIX} — ready for 30_manifests"
    ;;
  *)
    echo "unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac
