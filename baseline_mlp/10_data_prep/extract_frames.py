"""Extract one static hand frame per Sem-Lex clip, as the baseline MLP
expects it: 21 MediaPipe hand landmarks x 3 coords, wrist-centred and
scale-normalised, flattened to 63 floats.

Input  {split}/kpe_poses/<video_id>_pose.json  (the SHuBERT leaf pipeline output)
       {"<frame_idx>": {"hand_landmarks": [[ [x,y,z] x21 ], ...] | null, ...}}
Output frames/{split}.npz  X (N,63) float32, ids (N,) str, n_frames, n_hand

Frame selection and normalisation follow the Handshape-GNN preprocessing
(Carbo & Nalisnick, arXiv:2509.18309): pick the lowest-motion frame,
centre on the wrist, normalise scale. Two deltas, both forced by the data:

 #1 The reference pipeline reads x/y/z_right_hand columns (PopSign CSV
    layout, always the signer's right hand). SemLex kpe JSON gives an
    unlabelled list of 0-2 detected hands. We take the hand with the larger
    bounding-box diagonal when two are present -- the dominant/active
    articulator -- which for these one-handed citation-form clips is the
    signing hand.

 #2 The reference pipeline scores motion as the displacement to the NEXT
    frame and leaves the last frame at inf. With ~45% of SemLex frames
    carrying no detected hand, "next frame" is often absent; we score
    against the next frame that HAS a hand and record the gap, so a clip is
    never forced onto a bad frame just because its neighbour dropped out.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import FRAMES_DIR, SPLITS, kpe_dir  # noqa: E402

N_LANDMARKS = 21


def _bbox_diag(pts: np.ndarray) -> float:
    """Bounding-box diagonal of one hand, used to pick the dominant hand."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def pick_hand(hand_landmarks) -> np.ndarray | None:
    """Return the (21,3) array for the dominant hand in one frame, or None."""
    if not hand_landmarks:
        return None
    best = None
    best_score = -1.0
    for hand in hand_landmarks:
        if hand is None or len(hand) != N_LANDMARKS:
            continue
        pts = np.asarray(hand, dtype=np.float64)
        if pts.shape != (N_LANDMARKS, 3) or not np.isfinite(pts).all():
            continue
        score = _bbox_diag(pts)
        if score > best_score:
            best_score = score
            best = pts
    return best


def normalize(coords: np.ndarray) -> np.ndarray:
    """Wrist-centred, max-norm-scaled (the Handshape-GNN preprocessing)."""
    wrist = coords[0]
    centered = coords - wrist
    scale = np.clip(np.linalg.norm(centered, axis=1).max(), 1e-6, None)
    return centered / scale


def select_low_motion(pose_json: dict):
    """Pick the least-moving frame that has a usable hand.

    Returns (normalised (21,3), frame_index, n_frames, n_hand_frames) or None.
    """
    # kpe JSON keys are stringified frame indices; sort numerically.
    frame_keys = sorted(pose_json.keys(), key=int)
    n_frames = len(frame_keys)

    hands = []          # (frame_index, raw (21,3))
    for key in frame_keys:
        frame = pose_json[key]
        if not isinstance(frame, dict):
            continue
        pts = pick_hand(frame.get("hand_landmarks"))
        if pts is not None:
            hands.append((int(key), pts))

    if not hands:
        return None
    if len(hands) == 1:
        idx, pts = hands[0]
        return normalize(pts), idx, n_frames, 1

    # Motion = displacement to the next frame that also has a hand, measured
    # on NORMALISED coords so a hand moving toward the camera (which scales
    # the raw landmarks) does not read as a configuration change.
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


def run_split(split: str, limit: int | None = None) -> dict:
    src = kpe_dir(split)
    files = sorted(src.glob("*_pose.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no pose JSONs under {src}")

    ids, X, meta = [], [], []
    n_no_hand = 0
    for n, path in enumerate(files):
        video_id = path.name[: -len("_pose.json")]
        try:
            with path.open() as fh:
                pose = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] unreadable {path.name}: {exc}", flush=True)
            n_no_hand += 1
            continue

        res = select_low_motion(pose)
        if res is None:
            n_no_hand += 1
            continue

        pts, frame_idx, n_frames, n_hand = res
        ids.append(video_id)
        X.append(pts.reshape(-1).astype(np.float32))
        meta.append((frame_idx, n_frames, n_hand))

        if (n + 1) % 2000 == 0:
            print(f"  {split}: {n + 1}/{len(files)}", flush=True)

    X = np.stack(X) if X else np.zeros((0, N_LANDMARKS * 3), dtype=np.float32)
    meta = np.asarray(meta, dtype=np.int32).reshape(-1, 3)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out = FRAMES_DIR / f"{split}.npz"
    np.savez_compressed(
        out,
        X=X,
        ids=np.asarray(ids),
        selected_frame=meta[:, 0],
        n_frames=meta[:, 1],
        n_hand_frames=meta[:, 2],
    )

    report = {
        "split": split,
        "clips_seen": len(files),
        "clips_kept": len(ids),
        "clips_no_hand": n_no_hand,
        "hand_frame_fraction": (
            float(np.mean(meta[:, 2] / np.maximum(meta[:, 1], 1)))
            if len(ids) else 0.0
        ),
        "out": str(out),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke mode: only the first N clips per split")
    args = ap.parse_args()

    reports = [run_split(s, args.limit) for s in args.splits]
    if not args.limit:
        (FRAMES_DIR / "extract_report.json").write_text(
            json.dumps(reports, indent=2)
        )


if __name__ == "__main__":
    main()
