"""Driver for upstream sign_gendata.py — no upstream code modified.

The upstream script hardcodes `part = 'test'` in its __main__ block, so we
import its gendata() function and call it per split with our paths instead of
patching it. Everything else (27-joint selection, cyclic padding to 150
frames, the 150-frame crop, the packed (N, C, T, V, 1) output format) is
upstream's own code.

NOTE: ~19.5% of PopSign clips exceed 150 frames and are cropped by upstream
(measured over 400 random clips). That is upstream
behaviour and we keep it per the least-invasive rule — but it is a much
larger fraction than on SEM-LEX, so the exact number is recorded in
results/frame_stats.json rather than left implicit.

Output: <GENDATA_DIR>/{split}_data_joint.npy + {split}_label.pkl
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_config import GENDATA_DIR, LABELS_DIR, SL_GCN_DIR, WHOLEBODY_NPY_DIR

sys.path.insert(0, str(SL_GCN_DIR / "data_gen"))
from sign_gendata import gendata  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"])
    args = parser.parse_args()

    GENDATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        print(f"=== gendata {split} ===", flush=True)
        gendata(
            str(WHOLEBODY_NPY_DIR / split),
            str(LABELS_DIR / f"{split}_labels.csv"),
            str(GENDATA_DIR),
            part=split,
            config="27",
        )


if __name__ == "__main__":
    main()
