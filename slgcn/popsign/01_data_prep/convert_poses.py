"""Convert PopSign 543-point landmark parquet to upstream-format per-clip npys.

For every clip_id in results/splits.json, writes
<WHOLEBODY_NPY_DIR>/<split>/<clip_id>_color.mp4.npy — shape (T, 133, 3)
float32 in COCO-WholeBody layout (see map_543_to_wholebody133.py), the exact
filename pattern and array contract sign_gendata.py expects.

Also accumulates per-clip pose statistics into results/frame_stats.json:
the fraction of clips exceeding upstream's 150-frame crop, hand-detection
rates and out-of-unit-range coordinates. Those are the PopSign-specific data
facts worth knowing BEFORE training rather than after.

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
from common import utcnow, write_json
from data_config import (CONVERT_FAILURES_JSON, FRAME_STATS_JSON,
                         LANDMARKS_DIR, SPLITS_JSON, WHOLEBODY_NPY_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_543_to_wholebody133 import convert
from read_parquet_pose import pose_stats, read_pose


def convert_one(clip_id: str, split: str, channels: str, force: bool):
    out = WHOLEBODY_NPY_DIR / split / f"{clip_id}_color.mp4.npy"
    if out.exists() and not force:
        try:
            np.load(out, mmap_mode="r")  # header+size check, no data read
            return ("skip", None)
        except Exception:
            pass  # truncated by an interrupted run — reconvert
    try:
        pose = read_pose(LANDMARKS_DIR / f"{clip_id}.parquet")
        stats = pose_stats(pose)
        arr = convert(pose, channels=channels)
        tmp = out.with_name(out.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, arr)
        os.replace(tmp, out)  # atomic — no partial file can be "skipped" later
        return ("ok", stats)
    except Exception as e:  # noqa: BLE001 — collect per-file failures
        return (f"FAIL {clip_id}: {e}", None)


def summarise(stats: list) -> dict:
    if not stats:
        return {}
    lens = np.array([s["n_frames"] for s in stats])
    return {
        "n_clips_measured": len(stats),
        "frames_min": int(lens.min()), "frames_max": int(lens.max()),
        "frames_p50": float(np.percentile(lens, 50)),
        "frames_p90": float(np.percentile(lens, 90)),
        "frames_p99": float(np.percentile(lens, 99)),
        # upstream sign_gendata crops >150 and cyclically pads <150
        "frac_over_150_cropped": float(np.mean([s["over_150"] for s in stats])),
        "mean_frac_nan_left_hand": float(
            np.mean([s["frac_nan_left_hand"] for s in stats])),
        "mean_frac_nan_right_hand": float(
            np.mean([s["frac_nan_right_hand"] for s in stats])),
        "mean_frac_frames_no_hand": float(
            np.mean([s["no_hand_frames"] for s in stats])),
        "mean_frac_xy_out_of_unit": float(
            np.mean([s["frac_xy_out_of_unit"] for s in stats])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", default="presence",
                        choices=["presence", "xyz"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="first 40 clips per split")
    parser.add_argument("--max-failures", type=int, default=50,
                        help="per split: tolerate this many unreadable source "
                             "files (recorded in results/convert_failures.json "
                             "and excluded from the label CSVs); above it, "
                             "assume a systematic fault and exit non-zero")
    args = parser.parse_args()

    splits = json.loads(SPLITS_JSON.read_text())
    all_stats, summary = {}, {"created_at": utcnow(), "channels": args.channels}
    failed_ids = {}
    for split in ["train", "val", "test"]:
        ids = splits[split]
        if args.smoke:
            ids = ids[:40]
        (WHOLEBODY_NPY_DIR / split).mkdir(parents=True, exist_ok=True)
        fn = partial(convert_one, split=split, channels=args.channels,
                     force=args.force)
        with Pool(args.workers) as pool:
            results = pool.map(fn, ids, chunksize=32)
        codes = [r[0] for r in results]
        stats = [r[1] for r in results if r[1] is not None]
        all_stats[split] = stats
        fails = [c for c in codes if c.startswith("FAIL")]
        print(f"{split}: {codes.count('ok')} converted, "
              f"{codes.count('skip')} skipped, {len(fails)} failed")
        for f in fails[:20]:
            print("  " + f)
        # A handful of unreadable source files is a DATA defect (PopSign ships
        # at least one truncated parquet — valid PAR1 header, missing footer
        # magic, from an interrupted export upstream of us). Those clips are
        # excluded and recorded, not crashed on: failing the whole 174k-clip
        # corpus over one bad file would be the wrong trade. A LARGE failure
        # count means something systematic (wrong path, bad env, broken
        # mapper) and must still stop the run loudly.
        if len(fails) > args.max_failures:
            print(f"FATAL: {len(fails)} failures in {split} exceeds "
                  f"--max-failures={args.max_failures}; this looks systematic, "
                  f"not a few corrupt source files.")
            sys.exit(1)
        failed_ids[split] = [c.split(":")[0].removeprefix("FAIL ")
                             for c in fails]
        summary[split] = summarise(stats)

    # Unreadable clips must not reach upstream: sign_gendata.py would try to
    # np.load a file that was never written and die. Record them so
    # build_splits.py --exclude-failures can drop them from the label CSVs.
    n_failed = sum(len(v) for v in failed_ids.values())
    if n_failed:
        write_json(CONVERT_FAILURES_JSON,
                   {"created_at": utcnow(), "n_failed": n_failed,
                    "note": "unreadable source parquet — excluded from the "
                            "label CSVs by build_splits.py --exclude-failures",
                    **failed_ids})
        print(f"\n{n_failed} clip(s) unreadable -> {CONVERT_FAILURES_JSON}")
        print("  re-run build_splits.py --exclude-failures before run_gendata")

    # A resumed run only measures the clips it actually converted; say so
    # rather than letting a partial summary look like a full-corpus statistic.
    summary["note"] = ("statistics cover clips converted in THIS run "
                       "(skipped//already-present clips are not re-measured)")
    if any(summary.get(s) for s in ("train", "val", "test")):
        write_json(FRAME_STATS_JSON, summary)
        print(f"\nwrote {FRAME_STATS_JSON}")
        for s in ("train", "val", "test"):
            if summary.get(s):
                print(f"  {s}: p50={summary[s]['frames_p50']:.0f} frames, "
                      f"{summary[s]['frac_over_150_cropped']:.1%} cropped "
                      f"at upstream's 150-frame limit")


if __name__ == "__main__":
    main()
