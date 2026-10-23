"""Extract one static hand frame per PopSign clip, in the same 63-float
format the baseline MLP already consumes for Sem-Lex.

This is the sibling of `extract_frames.py`. Frame SELECTION and coordinate
NORMALISATION are ported verbatim from it (which follows the Handshape-GNN
preprocessing), so the two corpora differ only in where the hand
landmarks come from -- not in how a frame is chosen or scaled. Anything else
would make a PopSign-vs-Sem-Lex comparison meaningless.

Input   ${ASL2LSC_SLGCN_POPSIGN_DATA}/wholebody133/{split}/
            <clip_id>_color.mp4.npy   (T, 133, 3), x/y scaled x512,
                                       channel 2 = presence (not depth)
        feature_labels16.pkl          clip_id -> 16 label ids (-1 = NA)
Output  <DATA_ROOT>/popsign_frames/{split}.npz
            X (N,63) float32, ids (N,) str, n_frames, n_hand

Three deltas from the Sem-Lex extractor, all forced by the data:

 #1 PopSign poses are wholebody-133, so the two hands are fixed slots
    91:112 (left) and 112:133 (right) rather than an unlabelled list. We
    still pick the DOMINANT hand by bounding-box diagonal, matching the
    Sem-Lex rule, instead of assuming the right hand as the Handshape-GNN
    PopSign loader does -- otherwise the two corpora would use
    different articulator-selection rules.

 #2 Channel 2 is a PRESENCE flag (0/1), not a z coordinate: the 543->133
    conversion writes x, y, presence. Sem-Lex kpe gives real z. A hand is
    "detected" when its presence channel is non-zero; the third coordinate
    is passed through as-is so the vector stays 63-d and the architecture is
    untouched. This is a genuine input difference between the corpora.

 #3 x/y arrive scaled by 512 (upstream gendata convention). normalize() is
    scale-invariant -- it divides by the max centred norm -- so no rescaling
    is applied; doing so would change nothing and risk drifting from the
    Sem-Lex path.

Usage:
    python 10_data_prep/extract_frames_popsign.py --split test [--limit 200]
    python 10_data_prep/extract_frames_popsign.py --all
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

# Pin BLAS to one thread per process BEFORE numpy is imported: the per-clip
# work is tiny vector ops, so a threaded BLAS in every worker would only
# contend for the same cores. Must precede `import numpy`.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_config import DATA_ROOT, POPSIGN_ROOT  # noqa: E402
from extract_frames import normalize, _bbox_diag  # noqa: E402

POSE_DIR = POPSIGN_ROOT / "wholebody133"
LABELS16 = POPSIGN_ROOT / "feature_labels16.pkl"
OUT_DIR = DATA_ROOT / "popsign_frames"

N_LANDMARKS = 21
LEFT_HAND = slice(91, 112)    # wholebody-133 hand blocks
RIGHT_HAND = slice(112, 133)


def pick_hand_frame(frame: np.ndarray):
    """Dominant hand of one (133,3) frame as (21,3), or None if undetected.

    Same rule as the Sem-Lex extractor: larger bounding-box diagonal wins.
    """
    best, best_score = None, -1.0
    for sl in (LEFT_HAND, RIGHT_HAND):
        pts = frame[sl]
        if pts.shape != (N_LANDMARKS, 3) or not np.isfinite(pts).all():
            continue
        # Channel 2 is presence: an all-zero-presence hand was not detected.
        if not pts[:, 2].any():
            continue
        # An undetected hand can also arrive as all-zero x/y.
        if not np.abs(pts[:, :2]).any():
            continue
        score = _bbox_diag(pts[:, :2])
        if score > best_score:
            best_score, best = score, pts.astype(np.float64)
    return best


def select_low_motion(arr: np.ndarray):
    """Least-moving frame that has a usable hand.

    Ported from extract_frames.select_low_motion: motion is the displacement
    to the NEXT frame that also has a hand, measured on NORMALISED coords so
    a hand moving toward the camera does not read as a configuration change.
    Returns (normalised (21,3), frame_index, n_frames, n_hand) or None.
    """
    n_frames = int(arr.shape[0])
    hands = []
    for idx in range(n_frames):
        pts = pick_hand_frame(arr[idx])
        if pts is not None:
            hands.append((idx, pts))
    if not hands:
        return None
    if len(hands) == 1:
        idx, pts = hands[0]
        return normalize(pts), idx, n_frames, 1

    normed = [(idx, normalize(pts)) for idx, pts in hands]
    scores = []
    for i in range(len(normed)):
        if i == len(normed) - 1:
            scores.append(float("inf"))
            continue
        scores.append(float(np.linalg.norm(normed[i][1] - normed[i + 1][1])))
    best = int(np.argmin(scores))
    idx, pts = normed[best]
    return pts, idx, n_frames, len(hands)


def clip_id_of(path: Path) -> str:
    """`<clip_id>_color.mp4.npy` -> `<clip_id>`."""
    return path.name[: -len("_color.mp4.npy")]


def process_clip(path_str: str):
    """One clip -> (clip_id, 63-float vector, n_frames, n_hand) or None.

    Module-level and self-contained so multiprocessing can pickle it. Returns
    None for a clip with no usable hand or an unreadable/misshapen array; the
    caller counts those.
    """
    p = Path(path_str)
    try:
        arr = np.load(p)
    except (OSError, ValueError):
        return None
    if arr.ndim != 3 or arr.shape[1] != 133:
        return None
    got = select_low_motion(arr)
    if got is None:
        return None
    pts, _idx, n_frames, n_hand = got
    return clip_id_of(p), pts.reshape(-1).astype(np.float32), n_frames, n_hand


def run_split(split: str, labelled: set, limit=None, workers: int = 1) -> dict:
    src = POSE_DIR / split
    if not src.is_dir():
        raise SystemExit(f"missing pose dir: {src}")
    files = sorted(src.glob("*_color.mp4.npy"))
    if limit:
        files = files[:limit]

    # Only clips with phonological labels are usable for training; the rest
    # are dropped here rather than carried and masked, so the npz row count
    # matches the label count exactly.
    todo = [p for p in files if clip_id_of(p) in labelled]
    n_unlabelled = len(files) - len(todo)

    # The work is one independent file per clip, so it parallelises directly.
    # imap (ordered) rather than imap_unordered: `files` is sorted, so the
    # output rows stay in clip-id order regardless of which worker finishes
    # first. Completion-order output would make the npz differ run to run --
    # unacceptable when the point of these runs is reproducibility.
    results = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for r in pool.imap(process_clip, [str(p) for p in todo],
                               chunksize=64):
                results.append(r)
    else:
        results = [process_clip(str(p)) for p in todo]

    X, ids = [], []
    n_no_hand = 0
    frames_tot = hands_tot = 0
    for r in results:
        if r is None:
            n_no_hand += 1
            continue
        cid, vec, n_frames, n_hand = r
        X.append(vec)
        ids.append(cid)
        frames_tot += n_frames
        hands_tot += n_hand

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{split}.npz"
    np.savez_compressed(out, X=np.asarray(X, dtype=np.float32),
                        ids=np.asarray(ids, dtype=object))
    rep = {
        "split": split,
        "n_pose_files": len(files),
        "n_written": len(X),
        "n_dropped_unlabelled": n_unlabelled,
        "n_dropped_no_hand": n_no_hand,
        "hand_frame_rate": round(hands_tot / frames_tot, 4) if frames_tot else 0.0,
        "out": str(out),
    }
    print(json.dumps(rep, indent=2))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "val", "test"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override the output dir (used to diff a parallel "
                         "run against a serial one without clobbering the "
                         "real frames)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes; 0 (default) uses the "
                         "SLURM CPU allocation, 1 forces serial. Output is "
                         "id-ordered either way, so results do not depend on "
                         "this value")
    args = ap.parse_args()
    if not args.split and not args.all:
        raise SystemExit("pass --split or --all")

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = args.out_dir.expanduser()

    workers = args.workers
    if workers == 0:
        # Honour the allocation rather than the node's core count: taking
        # more than SLURM granted would oversubscribe a shared node.
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (
            os.cpu_count() or 1)
    workers = max(1, workers)
    print(f"workers: {workers}")

    with open(LABELS16, "rb") as fh:
        labelled = set(pickle.load(fh))
    print(f"labelled clips available: {len(labelled)}")

    splits = ("train", "val", "test") if args.all else (args.split,)
    reports = [run_split(s, labelled, args.limit, workers) for s in splits]
    (OUT_DIR / "extract_report.json").write_text(
        json.dumps(reports, indent=2) + "\n")
    print(f"\nwrote {OUT_DIR}/extract_report.json")


if __name__ == "__main__":
    main()
