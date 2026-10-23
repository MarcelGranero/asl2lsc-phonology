"""Bridge PopSign's SHuBERT features into this repo's trainer format.

PopSign's SHuBERT input features live under
$ASL2LSC_POPSIGN_PREPROC/popsign (dinov2_face, dinov2_hands, body_features),
extracted with the same SHuBERT pipeline (20_preprocess/) and in exactly the
layout this trainer expects -- (T,384) face and hands, (T,14) body. What is missing is only the two index files the dataset
reads:

    manifests/{split}.tsv        face  hand1  hand2  body  num_frames
    results/{split}_labels.csv   video_id,signer_id,f00..fNN

so this writes those and nothing else. No feature is recomputed.

Labels come from the SL-GCN PopSign leaf's sidecar pickles, NOT from Sem-Lex
(slgcn/popsign/01_data_prep/build_feature_labels16.py; the NA-as-class build
is passed with --pkl/--vocab, see run_popsign_nac_prep.sbatch). The pickles map
clip_id -> list[int] already encoded, with -1 meaning "no supervision for this
feature on this clip" (unmatched gloss). -1 passes straight through to
CrossEntropyLoss(ignore_index=-1) -- it is NOT remapped to a class here.

num_frames is min() over the four streams, matching how the Sem-Lex manifests
are built (streams can differ by a frame after independent extraction).

Clips are emitted only when ALL four feature files exist AND a label row is
present; anything else is counted and reported rather than silently skipped.
"""

import argparse
import csv
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_config import MANIFESTS_DIR, RESULTS_DIR  # noqa: E402

# --- site defaults (optional per-machine values) ---
_SITE_DEFAULTS = {}
# --- end site defaults ---


def _env_dir(var):
    value = os.environ.get(var) or _SITE_DEFAULTS.get(var)
    if not value:
        raise SystemExit(f"Environment variable {var} is not set "
                         f"(see the repository README, 'Data').")
    return Path(value)


POPSIGN_FEATURES = _env_dir("ASL2LSC_POPSIGN_PREPROC") / "popsign"
POPSIGN_REPO = REPO.parent / "slgcn" / "popsign"   # sibling leaf
POPSIGN_PROCESSED = _env_dir("ASL2LSC_SLGCN_POPSIGN_DATA")
SPLITS_JSON = POPSIGN_REPO / "results" / "splits.json"

PKL = {6: POPSIGN_PROCESSED / "feature_labels.pkl",
       16: POPSIGN_PROCESSED / "feature_labels16.pkl"}
VOCAB = {6: POPSIGN_REPO / "results" / "feature_vocab.json",
         16: POPSIGN_REPO / "results" / "feature_vocab16.json"}


def npy_len(path: Path) -> int:
    """First dimension of a .npy WITHOUT reading the array.

    np.load even with mmap_mode still opens and maps the file; over ~700k
    files (4 streams x 174k clips) that dominates the runtime on GPFS. The
    header alone carries the shape, so this reads a few hundred bytes per
    file instead.
    """
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _fortran, _dtype = np.lib.format._read_array_header(f, version)
    return int(shape[0]) if shape else 0


def stream_paths(clip_id: str) -> dict:
    return {
        "face": POPSIGN_FEATURES / "dinov2_face" / f"{clip_id}_face.npy",
        "hand1": POPSIGN_FEATURES / "dinov2_hands" / f"{clip_id}_hand1.npy",
        "hand2": POPSIGN_FEATURES / "dinov2_hands" / f"{clip_id}_hand2.npy",
        "body": POPSIGN_FEATURES / "body_features" / f"{clip_id}_pose.npy",
    }


def signer_of(clip_id: str) -> str:
    """PopSign ids look like [gtsignstudy]4a.8040-<word>-<timestamp>-0.
    The participant token is everything before the first '-'."""
    return clip_id.split("-", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=int, choices=(6, 16), default=16)
    ap.add_argument("--prefix", default=None,
                    help="split-name prefix (default popsign{N})")
    ap.add_argument("--pkl", type=Path, default=None,
                    help="label pickle override (default PKL[features]); use "
                         "the NA-as-class build for new runs")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="vocab json override (default VOCAB[features]); must "
                         "match --pkl")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N clips")
    ap.add_argument("--threads", type=int, default=32,
                    help="concurrent header reads; the scan is GPFS-latency "
                         "bound, so this is the main throughput knob")
    args = ap.parse_args()

    prefix = args.prefix or f"popsign{args.features}"
    pkl_path = args.pkl or PKL[args.features]
    vocab_path = args.vocab or VOCAB[args.features]
    raw_pkl = pickle.loads(pkl_path.read_bytes())
    # The two sidecars differ in shape: PopSign's original 6-feature pickle
    # wraps the map as {"features": [...], "labels": {clip: vec}}, while the
    # 16-feature build writes {clip: vec} flat. Accept both rather than
    # silently reading the wrapper's 2 top-level keys as if they were clips.
    if "labels" in raw_pkl and isinstance(raw_pkl.get("labels"), dict):
        labels = raw_pkl["labels"]
        pkl_order = raw_pkl.get("features")
    else:
        labels = raw_pkl
        pkl_order = None
    vocab = json.loads(vocab_path.read_text())
    feature_order = vocab["features"]
    if pkl_order is not None and list(pkl_order) != list(feature_order):
        sys.exit(f"pickle feature order {pkl_order} != vocab order {feature_order}")
    if not labels:
        sys.exit(f"no clip labels found in {pkl_path}")
    n_feat = len(feature_order)
    print(f"{args.features}-feature labels: {len(labels)} clips, "
          f"order={feature_order}")

    splits = json.loads(SPLITS_JSON.read_text())
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {}
    for split in ("train", "val", "test"):
        clips = splits[split]
        if args.limit:
            clips = clips[: args.limit]
        rows, label_rows = [], []
        miss_feat = miss_label = 0
        todo = [c for c in clips if labels.get(c) is not None]
        miss_label = len(clips) - len(todo)

        # The header reads are LATENCY-bound on GPFS, not CPU-bound: a serial
        # pass measured 3.5% CPU over 34 min because every one of the ~4x174k
        # opens waits on a metadata round-trip. Threads overlap those waits
        # (the work is in the kernel, so the GIL is not the limit).
        def measure(cid):
            p = stream_paths(cid)
            try:
                return cid, p, min(npy_len(v) for v in p.values())
            except (FileNotFoundError, ValueError, OSError):
                return cid, p, -1

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for cid, p, n in pool.map(measure, todo, chunksize=64):
                if n < 1:
                    miss_feat += 1
                    continue
                rows.append("\t".join([str(p["face"]), str(p["hand1"]),
                                       str(p["hand2"]), str(p["body"]), str(n)]))
                label_rows.append([cid, signer_of(cid)] +
                                  [str(int(x)) for x in labels[cid]])

        name = f"{prefix}_{split}"
        (MANIFESTS_DIR / f"{name}.tsv").write_text("\n".join(rows) + "\n")
        with (RESULTS_DIR / f"{name}_labels.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "signer_id"] +
                       [f"f{i:02d}" for i in range(n_feat)])
            w.writerows(label_rows)
        report[name] = {"clips": len(rows), "requested": len(clips),
                        "missing_features": miss_feat, "missing_labels": miss_label}
        print(f"{name}: {len(rows)}/{len(clips)} clips "
              f"(missing features {miss_feat}, missing labels {miss_label})")

    # A label_maps.json sibling so the trainer can size its heads from the same
    # source of truth it uses for Sem-Lex.
    maps_path = RESULTS_DIR / f"{prefix}_label_maps.json"
    maps_path.write_text(json.dumps({
        "feature_order": feature_order,
        "maps": {f: vocab["vocab"][f] for f in feature_order},
        "source": str(vocab_path),
        "note": ("-1 in the label CSV means no supervision "
                 "(ignore_index): unmatched gloss, plus structural NA "
                 "under the legacy ignore-policy build. NA-as-class "
                 "builds encode NA as a real class (null_policy in "
                 "the source vocab)."),
        "coverage": report,
    }, indent=2))
    print(f"-> {maps_path}")


if __name__ == "__main__":
    main()
