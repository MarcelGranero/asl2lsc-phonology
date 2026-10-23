#!/bin/bash
# Submit the full preprocessing chain for one split:
#   transcode -> [gate] -> kpe -> [gate] -> crop_face + crop_hands + body
#             -> [gate] -> dinov2 face + hands -> [final gate]
#
# Usage: bash 20_preprocess/submit_pipeline.sh --split {train|val|test} [--smoke]
#
# Chaining rule (cluster gotcha): NEVER `afterok` on a job array — one failed
# task stalls the dependency forever. Arrays chain via afterany into a verify
# gate job; the next stage is afterok on the GATE, which exits non-zero when
# coverage/integrity checks fail.
#
# --smoke: first 200 clips only, distinct work-list prefix ({split}_smoke),
# same output dirs (idempotent overlap with the full run is by design).
set -euo pipefail

SPLIT="" ; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[ -n "${SPLIT}" ] || { echo "usage: $0 --split {train|val|test} [--smoke]" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA=${ASL2LSC_SHUBERT_DATA}
WEBM_DIR="$(python3 -c "
import sys; sys.path.insert(0, '${REPO}')
from data_config import SEMLEX_VIDEO_DIRS
print(SEMLEX_VIDEO_DIRS['${SPLIT}'])")"
[ -d "${WEBM_DIR}" ] || { echo "no video dir ${WEBM_DIR} (train not downloaded yet?)" >&2; exit 1; }

N=$(find "${WEBM_DIR}" -maxdepth 1 -name '*.webm' | wc -l)
[ "${SMOKE}" = "1" ] && [ "${N}" -gt 200 ] && N=200
[ "${N}" -gt 0 ] || { echo "no webm files in ${WEBM_DIR}" >&2; exit 1; }

ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }
A_T=$(( $(ceil_div "${N}" 300) - 1 ))    # transcode, 300 clips/task
A_K=$(( $(ceil_div "${N}" 100) - 1 ))    # kpe, 100 clips/task
A_C=$(( $(ceil_div "${N}" 200) - 1 ))    # crops, 200 clips/task
A_DF=$(( $(ceil_div "${N}" 2000) - 1 ))  # dinov2 face, 2000 crops/task
A_DH=$(( $(ceil_div $((2 * N)) 2000) - 1 ))  # dinov2 hands (2 crops/video)

mkdir -p "${REPO}/logs" "${DATA}/pipeline_logs" "${DATA}/work_lists" \
         "${DATA}/${SPLIT}"/{kpe_poses,kpe_stats,pose_face_crops,pose_hand_crops,dinov2_face,dinov2_hands,body_features}

EXP="ALL,SPLIT=${SPLIT},SMOKE=${SMOKE}"
cd "${REPO}"

t=$(sbatch --parsable --array=0-${A_T} --export="${EXP}" 10_data_prep/run_transcode.slurm)
g1=$(sbatch --parsable --dependency=afterany:${t} --export="${EXP},STAGE=transcode" 20_preprocess/run_verify.slurm)
k=$(sbatch --parsable --dependency=afterok:${g1} --array=0-${A_K} --export="${EXP}" 20_preprocess/run_kpe.slurm)
g2=$(sbatch --parsable --dependency=afterany:${k} --export="${EXP},STAGE=kpe" 20_preprocess/run_verify.slurm)
cf=$(sbatch --parsable --dependency=afterok:${g2} --array=0-${A_C} --export="${EXP}" 20_preprocess/run_crop_face.slurm)
ch=$(sbatch --parsable --dependency=afterok:${g2} --array=0-${A_C} --export="${EXP}" 20_preprocess/run_crop_hands.slurm)
b=$(sbatch --parsable --dependency=afterok:${g2} --export="${EXP}" 20_preprocess/run_body.slurm)
g3=$(sbatch --parsable --dependency=afterany:${cf}:${ch} --export="${EXP},STAGE=crops" 20_preprocess/run_verify.slurm)
df=$(sbatch --parsable --dependency=afterok:${g3} --array=0-${A_DF} --export="${EXP},CROP_TYPE=face" 20_preprocess/run_dinov2.slurm)
dh=$(sbatch --parsable --dependency=afterok:${g3} --array=0-${A_DH} --export="${EXP},CROP_TYPE=hands" 20_preprocess/run_dinov2.slurm)
g4=$(sbatch --parsable --dependency=afterany:${df}:${dh}:${b} --export="${EXP},STAGE=features" 20_preprocess/run_verify.slurm)

cat <<EOF
Submitted chain for split=${SPLIT} smoke=${SMOKE} (${N} videos):
  transcode      ${t}  (array 0-${A_T})
  gate:transcode ${g1}
  kpe            ${k}  (array 0-${A_K})
  gate:kpe       ${g2}
  crop_face      ${cf} (array 0-${A_C})
  crop_hands     ${ch} (array 0-${A_C})
  body           ${b}
  gate:crops     ${g3}
  dinov2 face    ${df} (array 0-${A_DF})
  dinov2 hands   ${dh} (array 0-${A_DH})
  gate:features  ${g4}
Logs: ${REPO}/logs/ (job output), ${DATA}/pipeline_logs/ (per-stage problem logs)   Re-run: resubmit this script (stages skip existing outputs).
EOF
