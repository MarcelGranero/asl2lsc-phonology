"""Convert SEM-LEX 553-point pose npys to upstream-format per-video npys.

For every video_id in results/splits.json, writes
<WHOLEBODY_NPY_DIR>/<split>/<video_id>_color.mp4.npy — shape (T, 133, 3)
float32 in COCO-WholeBody layout (see map_553_to_wholebody133.py), the exact
filename pattern and array contract sign_gendata.py expects.

Resumable: existing outputs are skipped unless --force.
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_config import POSES_DIR, SPLITS_JSON, WHOLEBODY_NPY_DIR
from map_553_to_wholebody133 import convert


def convert_one(video_id: str, split: str, channels: str, force: bool):
    out = WHOLEBODY_NPY_DIR / split / f"{video_id}_color.mp4.npy"
    if out.exists() and not force:
        try:
            np.load(out, mmap_mode="r")  # header+size check, no data read
            return "skip"
        except Exception:
            pass  # truncated by an interrupted run — reconvert
    try:
        pose = np.load(POSES_DIR[split] / f"{video_id}.npy")
        arr = convert(pose, channels=channels)
        tmp = out.with_name(out.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, arr)
        os.replace(tmp, out)  # atomic — no partial file can be "skipped" later
        return "ok"
    except Exception as e:  # noqa: BLE001 — collect per-file failures
        return f"FAIL {video_id}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", default="presence",
                        choices=["presence", "xyz"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="first 40 clips per split")
    parser.add_argument("--poses-dir", type=Path, default=None,
                        help="override the source pose tree; expects "
                             "<dir>/<split>/<video_id>.npy. Use this for the "
                             "re-extracted poses so the released ones and "
                             "phase 1's outputs are left untouched.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override WHOLEBODY_NPY_DIR (pair with "
                             "--poses-dir so a re-posed run gets its own tree)")
    parser.add_argument("--skip-absent", action="store_true",
                        help="skip split ids with no source npy instead of "
                             "failing. For the re-posed tree, where 2,401 "
                             "train clips have no video to extract from.")
    args = parser.parse_args()

    global POSES_DIR, WHOLEBODY_NPY_DIR
    if args.poses_dir:
        POSES_DIR = {s: args.poses_dir / s for s in ("train", "val", "test")}
    if args.out_dir:
        WHOLEBODY_NPY_DIR = args.out_dir

    splits = json.loads(SPLITS_JSON.read_text())
    for split in ["train", "val", "test"]:
        ids = splits[split]
        if args.skip_absent:
            # The re-posed tree cannot contain the 2,401 train clips whose
            # source video is missing. Drop them HERE rather than relaxing the
            # failure check below, so a genuine conversion error still exits 1.
            present = {p.stem for p in POSES_DIR[split].glob("*.npy")}
            before = len(ids)
            ids = [v for v in ids if v in present]
            if before != len(ids):
                print(f"{split}: {before - len(ids)} clips absent from "
                      f"{POSES_DIR[split]} — skipped")
        if args.smoke:
            ids = ids[:40]
        (WHOLEBODY_NPY_DIR / split).mkdir(parents=True, exist_ok=True)
        fn = partial(convert_one, split=split, channels=args.channels,
                     force=args.force)
        with Pool(args.workers) as pool:
            results = pool.map(fn, ids, chunksize=64)
        fails = [r for r in results if r.startswith("FAIL")]
        print(f"{split}: {results.count('ok')} converted, "
              f"{results.count('skip')} skipped, {len(fails)} failed")
        for f in fails[:20]:
            print("  " + f)
        if fails:
            sys.exit(1)


if __name__ == "__main__":
    main()
