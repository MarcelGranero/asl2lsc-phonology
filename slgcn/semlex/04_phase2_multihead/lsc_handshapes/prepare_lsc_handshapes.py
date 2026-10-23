"""Build SL-GCN gendata tensors for the LSC benchmark handshape clips.

Chain, mirroring 01_data_prep for a foreign split:

    .pose ──pose_to_553──> (T,553,3) ──convert──> (T,133,3) ──gendata──>
    (N,3,150,27,1) + label pkl

The gloss label is IRRELEVANT here — the LSC handshape classes are a different
vocabulary from the SEM-LEX gloss head and nothing reads them. They are written
as 0 purely to satisfy upstream's `name,label_int` CSV contract. What matters
is the ORDER: the pkl's sample_name list is the join key back to the manifest,
and downstream scoring matches on the clip stem.

Filenames get the `_color.mp4.npy` suffix upstream's gendata expects; the stem
before it is the clip id (e.g. `06_hs_1729263_1731650_6-extended`).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01_data_prep"))
sys.path.insert(0, str(HERE))

from data_config import GENDATA_DIR, PROCESSED_DIR, SL_GCN_DIR  # noqa: E402
from map_553_to_wholebody133 import convert  # noqa: E402

LSC_ROOT = PROCESSED_DIR / "lsc_handshapes"

# LSC handshape benchmark root (a separate release).
# --- site defaults (optional per-machine values) ---
_SITE_DEFAULTS = {}
# --- end site defaults ---


def _bench_dir(var):
    value = os.environ.get(var) or _SITE_DEFAULTS.get(var)
    if not value:
        raise SystemExit(
            f"{var} is not set. Point it at the LSC handshape benchmark "
            f"(a separate release; see the repository README).")
    return Path(value)


BENCH = _bench_dir("ASL2LSC_LSC_BENCH")

SPLITS = {
    "test": BENCH / "handshape_poses",
    "examples": BENCH / "handshape_example_poses",
}


def build_split(split: str, force: bool, variant: str = "") -> int:
    """`variant` keeps parallel preprocessing runs (e.g. the 30fps resample)
    in their own directories instead of overwriting each other."""
    suffix = f"_{variant}" if variant else ""
    src = LSC_ROOT / f"pose553{suffix}" / split
    npy_dir = LSC_ROOT / f"wholebody133{suffix}" / split
    npy_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    if not files:
        raise SystemExit(f"no 553 npys in {src} — run pose_to_553.py first")

    stems = []
    for path in files:
        stem = path.stem
        stems.append(stem)
        out = npy_dir / f"{stem}_color.mp4.npy"
        if out.exists() and not force:
            continue
        np.save(out, convert(np.load(path)))

    # Upstream's `name,label_int` contract: the name is the BARE stem —
    # sign_gendata appends `_color.mp4.npy` itself. Label is a placeholder
    # (see module docstring): the LSC handshape id space is not the SEM-LEX
    # gloss space and nothing downstream reads it.
    labels_csv = LSC_ROOT / f"labels{suffix}" / f"{split}_labels.csv"
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for stem in stems:
            writer.writerow([stem, 0])

    sys.path.insert(0, str(SL_GCN_DIR / "data_gen"))
    from sign_gendata import gendata

    out_dir = LSC_ROOT / f"gendata_27{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gendata(str(npy_dir), str(labels_csv), str(out_dir), part=split, config="27")
    return len(stems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["test", "examples"])
    ap.add_argument("--variant", default="",
                    help="suffix for parallel preprocessing runs, e.g. '30fps'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    suffix = f"_{args.variant}" if args.variant else ""
    for split in args.splits:
        n = build_split(split, args.force, args.variant)
        print(f"{split}: {n} clips packed -> {LSC_ROOT / f'gendata_27{suffix}'}")


if __name__ == "__main__":
    main()
