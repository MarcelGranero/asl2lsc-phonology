"""Deduplicate semlex_metadata.csv (leekezar/SemLex issue #1).

Two known defects in the released metadata:
  1. Duplicate rows: some video_ids carry multiple rows (up to 15x), e.g. the
     same clip annotated under several label_types.
  2. Cross-split duplicate clips: ~922 byte-identical videos appear in more
     than one split (train/val), which leaks eval items into training.

Policy:
  - Row-level: keep exactly one row per video_id — prefer the `asllex` row
    (it carries the phonological columns phase 2 needs), tie-break on original
    CSV order (first wins). Deterministic.
  - Content-level: hash each pose .npy (sha256 of file bytes); when identical
    content appears under different video_ids or in multiple splits, keep the
    copy in the highest-priority split (test > val > train, protecting the
    eval protocol) and drop the rest; within one split keep the first
    video_id alphabetically.

Outputs: results/metadata_deduped.csv (+ pose_path/pose_ok columns),
results/dedupe_report.json.
"""
import argparse
import hashlib
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import (DEDUPE_REPORT_JSON, METADATA_DEDUPED_CSV, POSES_DIR,
                         SEMLEX_METADATA_CSV)

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
LABEL_TYPE_PRIORITY = {"asllex": 0, "signbank": 1, "freetext": 2}


def pose_path(row) -> Path:
    return POSES_DIR[row.split] / f"{row.video_id}.npy"


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_one(pair):
    video_id, path = pair
    return video_id, content_hash(Path(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="hash only the first 200 pose files per split")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--force", action="store_true",
                        help="recompute even if outputs exist")
    args = parser.parse_args()

    if (METADATA_DEDUPED_CSV.exists() and DEDUPE_REPORT_JSON.exists()
            and not args.force):
        print(f"{METADATA_DEDUPED_CSV} exists — skipping (use --force)")
        return

    df = pd.read_csv(SEMLEX_METADATA_CSV, index_col=0)
    n_rows_raw = len(df)
    report = {"started_at": utcnow(), "n_rows_raw": n_rows_raw,
              "n_video_ids_raw": df.video_id.nunique()}

    # --- 1. row-level dedupe -------------------------------------------------
    df = df.reset_index(drop=True)
    df["_orig_order"] = df.index
    df["_lt_prio"] = df.label_type.map(LABEL_TYPE_PRIORITY).fillna(9)
    df = (df.sort_values(["video_id", "_lt_prio", "_orig_order"])
            .drop_duplicates("video_id", keep="first")
            .sort_values("_orig_order")
            .drop(columns=["_lt_prio", "_orig_order"]))
    report["n_rows_after_row_dedupe"] = len(df)
    report["n_duplicate_rows_dropped"] = n_rows_raw - len(df)

    # --- 2. pose presence ----------------------------------------------------
    df["pose_path"] = [str(pose_path(r)) for r in df.itertuples()]
    df["pose_ok"] = [Path(p).exists() for p in df.pose_path]
    report["n_missing_pose_files"] = int((~df.pose_ok).sum())
    report["missing_pose_by_split"] = (
        df[~df.pose_ok].split.value_counts().to_dict())

    # --- 3. content-level dedupe (pose byte hash) ---------------------------
    hashable = df[df.pose_ok]
    if args.smoke:
        hashable = hashable.groupby("split").head(200)
    print(f"hashing {len(hashable)} pose files ({args.workers} workers) ...",
          flush=True)
    pairs = [(r.video_id, r.pose_path) for r in hashable.itertuples()]
    hashes = {}
    with Pool(args.workers) as pool:
        for i, (vid, h) in enumerate(
                pool.imap_unordered(_hash_one, pairs, chunksize=256)):
            hashes[vid] = h
            if i % 10000 == 0:
                print(f"  {i}/{len(pairs)}", flush=True)
    df["pose_sha256"] = df.video_id.map(hashes)

    by_hash = defaultdict(list)
    for r in df[df.pose_sha256.notna()].itertuples():
        by_hash[r.pose_sha256].append((SPLIT_PRIORITY[r.split], r.video_id))
    drop_ids = set()
    n_groups_dup = 0
    n_cross_split = 0
    for _, members in by_hash.items():
        if len(members) < 2:
            continue
        n_groups_dup += 1
        members.sort()
        splits_in_group = {p for p, _ in members}
        if len(splits_in_group) > 1:
            n_cross_split += 1
        drop_ids.update(vid for _, vid in members[1:])
    report["n_content_dup_groups"] = n_groups_dup
    report["n_content_dup_groups_cross_split"] = n_cross_split
    report["n_content_dup_clips_dropped"] = len(drop_ids)
    dropped_by_split = df[df.video_id.isin(drop_ids)].split.value_counts()
    report["content_dup_dropped_by_split"] = dropped_by_split.to_dict()

    df = df[~df.video_id.isin(drop_ids)]
    report["n_rows_final"] = len(df)
    report["final_by_split"] = df.split.value_counts().to_dict()
    report["finished_at"] = utcnow()
    report["smoke"] = args.smoke

    METADATA_DEDUPED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_DEDUPED_CSV, index=False)
    write_json(DEDUPE_REPORT_JSON, report)
    print("\n=== dedupe report ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
