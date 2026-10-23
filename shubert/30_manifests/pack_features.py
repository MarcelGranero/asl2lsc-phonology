"""Pack per-clip feature npys into one memmap blob per stream.

125 epochs x 38k clips x 4 files would be ~19M small-file GPFS opens; the
trainer instead memmaps 4 blobs and slices. Layout per split (packed/):
  {split}_face.f16.bin   concat of (num_frames, 384) f16, manifest order
  {split}_hand1.f16.bin  ditto
  {split}_hand2.f16.bin  ditto
  {split}_body.f32.bin   concat of (num_frames, 14) f32
  {split}_index.json     {"dims": {...}, "clips": [{"id", "offset", "length"}]}
Streams are truncated to the manifest's num_frames, so all four share one
offset/length table. DINOv2 features are O(1) so f16 quantisation (<=5e-4)
is harmless; the body stream stays f32 because its -9999.0 sentinel is NOT
f16-representable (spacing is 8 there -> would shift to -10000).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import MANIFESTS_DIR, PACKED_DIR, SPLITS  # noqa: E402

STREAM_DIMS = {"face": 384, "hand1": 384, "hand2": 384, "body": 14}
STREAM_DTYPES = {"face": np.float16, "hand1": np.float16,
                 "hand2": np.float16, "body": np.float32}
STREAM_SUFFIX = {"face": "f16", "hand1": "f16", "hand2": "f16", "body": "f32"}


def pack_split(split: str) -> None:
    manifest = MANIFESTS_DIR / f"{split}.tsv"
    rows = [line.rstrip("\n").split("\t") for line in manifest.open()]
    assert rows, f"empty manifest {manifest}"

    PACKED_DIR.mkdir(parents=True, exist_ok=True)
    outs = {
        stream: (PACKED_DIR
                 / f"{split}_{stream}.{STREAM_SUFFIX[stream]}.bin").open("wb")
        for stream in STREAM_DIMS
    }
    clips = []
    offset = 0
    for face, hand1, hand2, body, n_str in rows:
        n = int(n_str)
        for stream, path in zip(STREAM_DIMS, (face, hand1, hand2, body)):
            arr = np.load(path, mmap_mode="r")[:n]
            assert arr.shape == (n, STREAM_DIMS[stream]), (path, arr.shape, n)
            outs[stream].write(
                np.ascontiguousarray(arr, dtype=STREAM_DTYPES[stream]).tobytes()
            )
        stem = Path(face).name[: -len("_face.npy")]
        clips.append({"id": stem, "offset": offset, "length": n})
        offset += n
        if len(clips) % 2000 == 0:
            print(f"{split}: packed {len(clips)}/{len(rows)}", flush=True)
    for f in outs.values():
        f.close()

    index = {"dims": STREAM_DIMS,
             "dtypes": {s: np.dtype(d).name for s, d in STREAM_DTYPES.items()},
             "total_frames": offset, "clips": clips}
    (PACKED_DIR / f"{split}_index.json").write_text(json.dumps(index))
    gb = offset * sum(
        STREAM_DIMS[s] * np.dtype(STREAM_DTYPES[s]).itemsize
        for s in STREAM_DIMS
    ) / 1e9
    print(f"{split}: {len(clips)} clips, {offset} frames, ~{gb:.1f} GB packed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True,
                        help="split name; any name with a manifest works "
                        "(semlex train/val/test, popsign16_train, ...) — the "
                        "manifest-existence check below is the validation")
    args = parser.parse_args()
    for split in args.split:
        manifest = MANIFESTS_DIR / f"{split}.tsv"
        if not manifest.exists():
            raise SystemExit(f"no manifest for split '{split}': {manifest}")
        pack_split(split)


if __name__ == "__main__":
    main()
