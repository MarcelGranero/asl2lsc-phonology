"""Per-stage verification gate. Exits non-zero when a stage's outputs are
incomplete or malformed, which holds the downstream `afterok` dependency in
submit_pipeline.sh.

Stages:
  transcode  mp4 coverage vs webm dir (+ --limit), frame-exact spot-check
             (decord count == ffprobe -count_frames) on 20 random outputs
  kpe        pose-json coverage vs videos list, no .tmp residue, 10-sample
             parse + per-frame-entry count == video frame count
  crops      face/hand crop coverage vs kpe count; QA grid PNG to docs/qa/
  features   dinov2 face/hands npy coverage vs crops, body npy coverage vs
             poses; dtype/shape/NaN spot-checks

Writes results/{prefix}_verify_{stage}.json. Runs under
shubert_feature_extraction (decord/cv2/numpy/ffprobe).
"""

import argparse
import gzip
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    PIPELINE_LOGS_DIR,
    QA_DIR,
    RESULTS_DIR,
    SEMLEX_VIDEO_DIRS,
    SPLITS,
    VIDEOS_MP4_DIR,
    stage_dir,
)

COVERAGE_MIN = 0.995
RNG = random.Random(0)


def load_list(path: Path) -> list[Path]:
    with gzip.open(path, "rb") as f:
        return [Path(p) for p in pickle.load(f)]


def decord_len(path: Path) -> int:
    import decord

    return len(decord.VideoReader(str(path)))


def ffprobe_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=300,
    )
    return int(out.stdout.strip())


def check(report: dict, name: str, ok: bool, detail) -> None:
    report["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")


def verify_transcode(split: str, limit: int, report: dict) -> None:
    webms = sorted(SEMLEX_VIDEO_DIRS[split].glob("*.webm"))
    if limit:
        webms = webms[:limit]
    mp4_dir = VIDEOS_MP4_DIR / split
    present = [w for w in webms if (mp4_dir / f"{w.stem}.mp4").exists()]
    cov = len(present) / len(webms)
    check(report, "coverage", cov >= COVERAGE_MIN,
          f"{len(present)}/{len(webms)} = {cov:.4f}")

    # Losslessness vs the SOURCE: webm ffprobe == mp4 ffprobe == mp4 decord.
    # (An earlier -vsync 0 recipe dropped a frame on ~60% of clips while
    # decord masked it by over-reporting — hence the three-way check.)
    sample = RNG.sample(present, min(20, len(present)))
    mismatches = []
    for w in sample:
        mp4 = mp4_dir / f"{w.stem}.mp4"
        n_src = ffprobe_frames(w)
        n_mp4 = ffprobe_frames(mp4)
        n_dec = decord_len(mp4)
        if not (n_src == n_mp4 == n_dec):
            mismatches.append((w.stem, n_src, n_mp4, n_dec))
    check(report, "frame_exact_sample", not mismatches,
          f"{len(sample)} sampled (webm/mp4/decord), mismatches: {mismatches}")

    flog = PIPELINE_LOGS_DIR / f"{split}_transcode_failures.log"
    n_fail = sum(1 for _ in flog.open()) if flog.exists() else 0
    check(report, "failures_logged", True, f"{n_fail} lines in {flog.name}")


def verify_kpe(split: str, videos: list[Path], report: dict) -> None:
    poses_dir = stage_dir(split, "kpe_poses")
    present = [v for v in videos if (poses_dir / f"{v.stem}_pose.json").exists()]
    cov = len(present) / len(videos)
    check(report, "coverage", cov >= COVERAGE_MIN,
          f"{len(present)}/{len(videos)} = {cov:.4f}")

    tmps = list(poses_dir.glob("*.tmp*"))
    check(report, "no_tmp_residue", not tmps, f"{len(tmps)} .tmp files")

    bad = []
    for v in RNG.sample(present, min(10, len(present))):
        try:
            d = json.loads((poses_dir / f"{v.stem}_pose.json").read_text())
            n_video = decord_len(v)
            if len(d) != n_video:
                bad.append((v.stem, len(d), n_video))
        except Exception as e:
            bad.append((v.stem, "parse-error", str(e)[:80]))
    check(report, "json_sample", not bad, f"10 sampled, bad: {bad}")


def verify_crops(split: str, videos: list[Path], prefix: str, report: dict) -> None:
    poses_dir = stage_dir(split, "kpe_poses")
    faces_dir = stage_dir(split, "pose_face_crops")
    hands_dir = stage_dir(split, "pose_hand_crops")
    with_pose = [v for v in videos if (poses_dir / f"{v.stem}_pose.json").exists()]

    have_face = [v for v in with_pose if (faces_dir / f"{v.stem}_face.mp4").exists()]
    have_hands = [v for v in with_pose
                  if (hands_dir / f"{v.stem}_hand1.mp4").exists()
                  and (hands_dir / f"{v.stem}_hand2.mp4").exists()]
    for name, have in (("face_coverage", have_face), ("hands_coverage", have_hands)):
        cov = len(have) / len(with_pose)
        check(report, name, cov >= COVERAGE_MIN,
              f"{len(have)}/{len(with_pose)} = {cov:.4f}")

    # QA grid: middle frame of face/hand1/hand2 for 8 random stems.
    import cv2
    import decord

    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in RNG.sample(have_face, min(8, len(have_face))):
        row = []
        for path in (faces_dir / f"{v.stem}_face.mp4",
                     hands_dir / f"{v.stem}_hand1.mp4",
                     hands_dir / f"{v.stem}_hand2.mp4"):
            try:
                vr = decord.VideoReader(str(path))
                frame = vr[len(vr) // 2].asnumpy()
            except Exception:
                frame = np.zeros((224, 224, 3), np.uint8)
            row.append(frame)
        rows.append(np.concatenate(row, axis=1))
    if rows:
        png = QA_DIR / f"{prefix}_crops_grid.png"
        cv2.imwrite(str(png), np.concatenate(rows, axis=0))
        check(report, "qa_grid", True,
              f"{png} — REVIEW: faces must be eyes+mouth on grey (masked)")


def verify_features(split: str, videos: list[Path], report: dict) -> None:
    poses_dir = stage_dir(split, "kpe_poses")
    faces_dir = stage_dir(split, "pose_face_crops")
    hands_dir = stage_dir(split, "pose_hand_crops")
    dino_face = stage_dir(split, "dinov2_face")
    dino_hands = stage_dir(split, "dinov2_hands")
    body_dir = stage_dir(split, "body_features")

    face_crops = [v.stem for v in videos
                  if (faces_dir / f"{v.stem}_face.mp4").exists()]
    face_npys = [s for s in face_crops if (dino_face / f"{s}_face.npy").exists()]
    cov = len(face_npys) / len(face_crops)
    check(report, "dinov2_face_coverage", cov >= COVERAGE_MIN,
          f"{len(face_npys)}/{len(face_crops)} = {cov:.4f}")

    hand_crops = [v.stem for v in videos
                  if (hands_dir / f"{v.stem}_hand1.mp4").exists()
                  and (hands_dir / f"{v.stem}_hand2.mp4").exists()]
    hand_npys = [s for s in hand_crops
                 if (dino_hands / f"{s}_hand1.npy").exists()
                 and (dino_hands / f"{s}_hand2.npy").exists()]
    cov = len(hand_npys) / len(hand_crops)
    check(report, "dinov2_hands_coverage", cov >= COVERAGE_MIN,
          f"{len(hand_npys)}/{len(hand_crops)} = {cov:.4f}")

    poses = [v.stem for v in videos
             if (poses_dir / f"{v.stem}_pose.json").exists()]
    body_npys = [s for s in poses if (body_dir / f"{s}_pose.npy").exists()]
    cov = len(body_npys) / len(poses)
    check(report, "body_coverage", cov >= COVERAGE_MIN,
          f"{len(body_npys)}/{len(poses)} = {cov:.4f}")

    bad = []
    for s in RNG.sample(face_npys, min(100, len(face_npys))):
        a = np.load(dino_face / f"{s}_face.npy", mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != 384 or not np.isfinite(
                np.asarray(a[: min(4, a.shape[0])])).all():
            bad.append(("face", s, a.shape, str(a.dtype)))
    for s in RNG.sample(hand_npys, min(50, len(hand_npys))):
        for h in ("hand1", "hand2"):
            a = np.load(dino_hands / f"{s}_{h}.npy", mmap_mode="r")
            if a.ndim != 2 or a.shape[1] != 384:
                bad.append((h, s, a.shape, str(a.dtype)))
    for s in RNG.sample(body_npys, min(100, len(body_npys))):
        a = np.load(body_dir / f"{s}_pose.npy", mmap_mode="r")
        arr = np.asarray(a)
        # -9999 sentinels are expected; NaN/Inf are not
        if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(arr).all():
            bad.append(("body", s, a.shape, str(a.dtype)))
    check(report, "shape_sample", not bad, f"bad entries: {bad[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["transcode", "kpe", "crops", "features"])
    parser.add_argument("--split", required=True, choices=list(SPLITS))
    parser.add_argument("--videos-list", type=Path,
                        help="required for kpe/crops/features")
    parser.add_argument("--limit", type=int, default=0,
                        help="transcode only: smoke limit")
    args = parser.parse_args()

    if args.stage == "transcode":
        prefix = f"{args.split}_smoke" if args.limit else args.split
        videos = None
    else:
        assert args.videos_list, "--videos-list required for this stage"
        name = args.videos_list.name
        prefix = name[: -len("_videos.list.gz")]
        videos = load_list(args.videos_list)

    report = {"stage": args.stage, "split": args.split, "prefix": prefix,
              "checks": {}}
    if args.stage == "transcode":
        verify_transcode(args.split, args.limit, report)
    elif args.stage == "kpe":
        verify_kpe(args.split, videos, report)
    elif args.stage == "crops":
        verify_crops(args.split, videos, prefix, report)
    else:
        verify_features(args.split, videos, report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{prefix}_verify_{args.stage}.json"
    out.write_text(json.dumps(report, indent=2))
    failed = [k for k, v in report["checks"].items() if not v["ok"]]
    print(f"report -> {out}")
    if failed:
        print(f"GATE FAILED: {failed}")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
