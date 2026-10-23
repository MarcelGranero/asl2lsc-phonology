"""Regenerate the stage input lists (upstream gzip-pickle format) from a
videos work list ∩ what actually exists on disk.

We do not trust the <dir>.list.gz files the upstream scripts drop into their
output dirs (concurrent array tasks race on them); the gate job runs this
right before each dependent stage so lists always reflect reality, filtered
to the videos of the current (full or smoke) work list.

Writes, next to the videos list (work_lists/):
  {prefix}_kpe_poses.list.gz    <stem>_pose.json         (input for body)
  {prefix}_face_crops.list.gz   <stem>_face.mp4          (input for dinov2 face)
  {prefix}_hand_crops.list.gz   <stem>_hand1/2.mp4       (input for dinov2 hands)
where {prefix} is the videos-list name minus "_videos.list.gz".
"""

import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import SPLITS, stage_dir  # noqa: E402


def dump(paths: list[Path], out: Path) -> None:
    with gzip.open(out, "wb") as f:
        pickle.dump([str(p) for p in paths], f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(paths)} entries -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=list(SPLITS))
    parser.add_argument("--videos-list", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(args.videos_list, "rb") as f:
        stems = [Path(p).stem for p in pickle.load(f)]
    name = args.videos_list.name
    assert name.endswith("_videos.list.gz"), name
    prefix = name[: -len("_videos.list.gz")]
    out_dir = args.videos_list.parent

    poses_dir = stage_dir(args.split, "kpe_poses")
    faces_dir = stage_dir(args.split, "pose_face_crops")
    hands_dir = stage_dir(args.split, "pose_hand_crops")

    dump(
        [p for s in stems if (p := poses_dir / f"{s}_pose.json").exists()],
        out_dir / f"{prefix}_kpe_poses.list.gz",
    )
    dump(
        [p for s in stems if (p := faces_dir / f"{s}_face.mp4").exists()],
        out_dir / f"{prefix}_face_crops.list.gz",
    )
    hand_paths = []
    for s in stems:
        for h in ("hand1", "hand2"):
            p = hands_dir / f"{s}_{h}.mp4"
            if p.exists():
                hand_paths.append(p)
    dump(hand_paths, out_dir / f"{prefix}_hand_crops.list.gz")


if __name__ == "__main__":
    main()
