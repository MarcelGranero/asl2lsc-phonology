"""Stage 0: frame-exact re-encode of the Sem-Lex .webm releases to mp4.

The released webms (VP9, browser-MediaRecorder timestamps, avg_frame_rate
0/0) are unreadable by decord, which every downstream upstream script uses.
`ffmpeg -c copy` remux DROPS frames (non-monotonic DTS) — never use it.
`-vsync 0` passthrough ALSO silently drops a frame on ~60% of clips (broken
PTS collide on the output timebase; decord then masks it by over-reporting
with a duplicated last frame — caught by the smoke-run transcode gate).
Frame-exact recipe: rewrite timestamps to frame-index CFR *before* encoding
(`-vf setpts=N/(25*TB) -r 25`), verified webm==mp4==decord on both a
previously-lossy and a previously-clean clip. The 25 fps stamp is arbitrary;
the whole pipeline is frame-based.

Idempotent: skips outputs that already exist and decode; writes via a
PID-unique .tmp + rename. Failures land in
pipeline_logs/{split}_transcode_failures.log (one "path\treason" per line).

Array-sliced like the upstream scripts: --index selects one contiguous batch
of --batch-size files from the sorted webm listing; omit --index to process
everything. --limit N restricts to the first N files (smoke mode).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    PIPELINE_LOGS_DIR,
    SEMLEX_VIDEO_DIRS,
    VIDEOS_MP4_DIR,
)


def decodable_frames(path: Path) -> int:
    """Frame count via decord; 0 if unreadable."""
    try:
        import decord

        return len(decord.VideoReader(str(path)))
    except Exception:
        return 0


def transcode_one(src: Path, dst: Path, failures_log: Path) -> str:
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-i", str(src),
        "-vf", "setpts=N/(25*TB)", "-r", "25",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-an",
        "-y", str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg rc={proc.returncode}: {proc.stderr[-300:]}")
        n = decodable_frames(tmp)
        if n <= 0:
            raise RuntimeError("output not decodable by decord")
        tmp.replace(dst)
        return f"ok {n} frames"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        with failures_log.open("a") as f:
            f.write(f"{src}\t{e}\n")
        return f"FAIL {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=list(SEMLEX_VIDEO_DIRS))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only the first N files (smoke mode); 0 = all")
    args = parser.parse_args()

    src_dir = SEMLEX_VIDEO_DIRS[args.split]
    out_dir = VIDEOS_MP4_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    PIPELINE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    failures_log = PIPELINE_LOGS_DIR / f"{args.split}_transcode_failures.log"

    files = sorted(src_dir.glob("*.webm"))
    if args.limit:
        files = files[: args.limit]
    if args.index is not None:
        files = files[args.index * args.batch_size:(args.index + 1) * args.batch_size]
        if not files:
            print(f"index {args.index}: empty slice, nothing to do")
            return

    n_ok = n_skip = n_fail = 0
    for src in files:
        status = transcode_one(src, out_dir / f"{src.stem}.mp4", failures_log)
        print(f"{src.name}: {status}", flush=True)
        n_ok += status.startswith("ok")
        n_skip += status == "skip"
        n_fail += status.startswith("FAIL")
    print(f"done: {n_ok} transcoded, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
