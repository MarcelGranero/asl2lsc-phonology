#!/bin/bash
#SBATCH --job-name=lsc_score_gate
#SBATCH --output=logs/score_gate_%j.out
#SBATCH --error=logs/score_gate_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=short

# Gate job: after all inference jobs finish (afterany -- runs even if one
# failed, so a partial refresh still gets scored), re-score every tracked
# arm from the per-leaf per-clip outputs. Plain python3 (stdlib); no conda
# env.
#
# INFER_JOBS (comma-separated job ids) is exported by run_inference_all.sh
# so the log states which inference jobs fed this scoring pass.

set -eo pipefail
set -u
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ -n "${INFER_JOBS:-}" ]; then
  echo "=== inference job states ==="
  sacct -j "$INFER_JOBS" --format=JobID,JobName%25,State,Elapsed -X || true
fi

echo "=== re-scoring every tracked arm (run_all.sh) ==="
bash "$HERE/run_all.sh"

echo "GATE DONE -- metrics in $HERE/results/"
