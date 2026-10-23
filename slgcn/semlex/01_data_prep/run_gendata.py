"""Driver for upstream sign_gendata.py — no upstream code modified.

The upstream script hardcodes `part = 'test'` in its __main__ block, so we
import its gendata() function and call it per split with our paths instead of
patching it. Everything else (27-joint selection, cyclic padding to 150
frames, the packed (N, C, T, V, 1) output format) is upstream's own code.

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
    parser.add_argument("--wholebody-dir", type=Path, default=None,
                        help="override WHOLEBODY_NPY_DIR (e.g. the re-posed tree)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override GENDATA_DIR; pair with --wholebody-dir "
                             "so a re-posed run never overwrites phase 1's "
                             "packed tensors")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="override LABELS_DIR. Needed by the re-posed run: "
                             "2,401 train clips have no source video, and "
                             "upstream gendata np.loads every path in the CSV "
                             "and dies on the first missing file.")
    args = parser.parse_args()

    wholebody = args.wholebody_dir or WHOLEBODY_NPY_DIR
    out_dir = args.out_dir or GENDATA_DIR
    labels_dir = args.labels_dir or LABELS_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        print(f"=== gendata {split} ===", flush=True)
        gendata(
            str(wholebody / split),
            str(labels_dir / f"{split}_labels.csv"),
            str(out_dir),
            part=split,
            config="27",
        )


if __name__ == "__main__":
    main()
