#!/bin/bash
#SBATCH --job-name=slsh_train_smoke
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A.out

# Trainer smoke with ZERO train-split dependency: train on val, eval on test,
# 3 epochs. Pass criteria (vs results/labels_report.json majority floors):
# loss strictly decreasing; every feature >= its floor; macro >= floor+5pts;
# then a --resume roundtrip must reproduce epoch-3 val macro.
# NOTE: `set -u` must come AFTER conda activate — the shared env's MKL
# activation hook reads $MKL_INTERFACE_LAYER unset and would abort the job.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -u

set -e   # any stage failing must fail the job

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
python "${REPO}/40_train/model_loading.py"   # self-test: sha256 + 226/226 + padding

# 3 epochs from scratch...
python "${REPO}/40_train/train.py" \
    --run-name smoke \
    --train-split val --val-split test \
    --epochs 3 --min-epochs 3 \
    --fresh

# ...then assert it actually learned (vs majority-class floors).
python "${REPO}/40_train/check_smoke.py" --run-name smoke --val-split test

# Resume roundtrip: relaunching with --epochs 4 must load last.pt, run ONLY
# epoch 3, and leave the epoch-0..2 history intact.
python "${REPO}/40_train/train.py" \
    --run-name smoke \
    --train-split val --val-split test \
    --epochs 4 --min-epochs 3
python - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])
from data_config import RUNS_DIR
rows = [json.loads(l) for l in
        (RUNS_DIR / "smoke" / "log.jsonl").read_text().splitlines() if l.strip()]
epochs = [r["epoch"] for r in rows]
assert epochs == [0, 1, 2, 3], f"resume did not continue cleanly: {epochs}"
print(f"RESUME ROUNDTRIP OK: epochs {epochs}, "
      f"best {max(r['val_macro_recall1'] for r in rows):.4f}")
PY
echo "TRAINER SMOKE PASSED"
