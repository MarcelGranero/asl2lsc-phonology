#!/bin/bash
#SBATCH --job-name=popsign_nac_prep
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/popsign_nac_prep_%j.out
#SBATCH --error=logs/popsign_nac_prep_%j.err

# Stage the NA-as-class PopSign splits (prefix popsign16nac): manifests +
# label CSVs + label maps from the naclass pickle/vocab, packed-blob
# symlinks (the features are label-independent -- the packed dataset joins
# labels per clip id, so the popsign16 blobs serve both builds), and the
# majority floors. NA-as-class is the default for new experiments.
#
# Chain: sbatch --dependency=afterok:<labels16_naclass job> from shubert/.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shubert_train_env
set -eo pipefail
set -u          # AFTER activate: the MKL hook reads an unset var

REPO="${SLURM_SUBMIT_DIR}"   # submit from the shubert/ leaf dir
PROCESSED=${ASL2LSC_SLGCN_POPSIGN_DATA}
PACKED=${ASL2LSC_SHUBERT_DATA}/packed
NACLASS_PKL="$PROCESSED/feature_labels16_naclass.pkl"
NACLASS_VOCAB="$REPO/../slgcn/popsign/results/feature_vocab16_naclass.json"

[ -f "$NACLASS_PKL" ]   || { echo "MISSING $NACLASS_PKL" >&2; exit 1; }
[ -f "$NACLASS_VOCAB" ] || { echo "MISSING $NACLASS_VOCAB" >&2; exit 1; }

echo "=== manifests + labels + label maps (prefix popsign16nac) ==="
python "$REPO/10_data_prep/make_popsign_manifests.py" \
  --features 16 --prefix popsign16nac \
  --pkl "$NACLASS_PKL" --vocab "$NACLASS_VOCAB" \
  --threads "$SLURM_CPUS_PER_TASK"

echo "=== packed-blob symlinks (features are label-independent) ==="
# Pack file names are <prefix>_<split>_<stream>.<dtype>.bin + _index.json:
for split in train val test; do
  for suffix in _face.f16.bin _hand1.f16.bin _hand2.f16.bin _body.f32.bin _index.json; do
    src="$PACKED/popsign16_${split}${suffix}"
    dst="$PACKED/popsign16nac_${split}${suffix}"
    [ -e "$src" ] || { echo "MISSING pack $src" >&2; exit 1; }
    ln -sfn "$src" "$dst"
    echo "  $dst -> $src"
  done
done

echo "=== majority floors ==="
python "$REPO/10_data_prep/make_popsign_floors.py" --prefix popsign16nac

echo "POPSIGN NAC PREP DONE"
