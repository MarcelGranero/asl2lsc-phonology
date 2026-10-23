"""Datasets + collate for the phonological fine-tune.

Two backends with identical items:
  SemLexPackedDataset  slices the float16 memmap blobs from pack_features.py
                       (default; avoids ~19M small-file GPFS opens/run)
  SemLexNpyDataset     reads the per-clip npys via the manifest (--no-pack;
                       used before packing exists, e.g. pipeline smoke)

Item: face (T,384) f32, hand1, hand2, body (T,14), labels (16,) int64, id.
Collate pads to batch-max T; padding_mask (B,T) bool True at pads.
Stdlib + numpy + torch only (shubert_train_env has no pandas).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    MANIFESTS_DIR,
    PACKED_DIR,
    RESULTS_DIR,
)

N_FEATURES = 16


def load_labels(split: str) -> dict[str, np.ndarray]:
    """video_id -> int label vector.

    The width is read from the header rather than assumed to be N_FEATURES:
    Sem-Lex ships 16 columns but PopSign's 6-feature split ships 6, and
    hardcoding 16 makes that split die on a KeyError for 'f06'.
    """
    out = {}
    with (RESULTS_DIR / f"{split}_labels.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or [])
                if len(c) == 3 and c[0] == "f" and c[1:].isdigit()]
        if not cols:
            raise SystemExit(f"{split}_labels.csv has no fNN feature columns")
        for row in reader:
            out[row["video_id"]] = np.array(
                [int(row[c]) for c in cols], dtype=np.int64)
    return out


def load_n_classes(maps_name: str = "label_maps.json") -> tuple[list[str], list[int]]:
    """Feature order + per-head class counts.

    maps_name selects the corpus: Sem-Lex uses the default label_maps.json,
    PopSign runs pass popsign{6,16}_label_maps.json. Getting this wrong is
    silent and destructive -- the heads would be sized for one corpus and fed
    the other's label ids -- so the trainer derives it from the split name
    rather than letting it default.
    """
    maps = json.loads((RESULTS_DIR / maps_name).read_text())
    order = maps["feature_order"]
    return order, [len(maps["maps"][name]) for name in order]


class SemLexPackedDataset(Dataset):
    STREAMS = ("face", "hand1", "hand2", "body")

    def __init__(self, split: str):
        index = json.loads((PACKED_DIR / f"{split}_index.json").read_text())
        self.dims = index["dims"]
        self.dtypes = index["dtypes"]  # body is f32 (its -9999 sentinel is
        # not f16-representable); DINOv2 streams are f16
        labels = load_labels(split)
        self.clips = [c for c in index["clips"] if c["id"] in labels]
        assert self.clips, f"no labeled clips in {split} packed index"
        dropped = len(index["clips"]) - len(self.clips)
        if dropped:
            print(f"{split}: {dropped} packed clips without labels (dropped)")
        self.labels = labels
        self.split = split
        self._mm = None  # opened lazily per worker

    def _memmaps(self):
        if self._mm is None:
            self._mm = {}
            for s in self.STREAMS:
                dtype = np.dtype(self.dtypes[s])
                suffix = "f16" if dtype == np.float16 else "f32"
                self._mm[s] = np.memmap(
                    PACKED_DIR / f"{self.split}_{s}.{suffix}.bin",
                    dtype=dtype, mode="r",
                ).reshape(-1, self.dims[s])
        return self._mm

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        o, n = clip["offset"], clip["length"]
        mm = self._memmaps()
        streams = [
            torch.from_numpy(np.array(mm[s][o:o + n], dtype=np.float32))
            for s in self.STREAMS
        ]
        return (*streams, torch.from_numpy(self.labels[clip["id"]]), clip["id"])


class SemLexNpyDataset(Dataset):
    def __init__(self, split: str):
        labels = load_labels(split)
        self.rows = []
        with (MANIFESTS_DIR / f"{split}.tsv").open() as f:
            for line in f:
                face, h1, h2, body, n = line.rstrip("\n").split("\t")
                stem = Path(face).name[: -len("_face.npy")]
                if stem in labels:
                    self.rows.append((face, h1, h2, body, int(n), stem))
        assert self.rows, f"no labeled rows in {split} manifest"
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        face, h1, h2, body, n, stem = self.rows[i]
        streams = [
            torch.from_numpy(
                np.array(np.load(p, mmap_mode="r")[:n], dtype=np.float32))
            for p in (face, h1, h2, body)
        ]
        return (*streams, torch.from_numpy(self.labels[stem]), stem)


def collate(batch):
    faces, h1s, h2s, bodies, labels, ids = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in faces], dtype=torch.long)
    T = int(lengths.max())
    B = len(batch)

    def pad_stack(tensors, dim):
        out = torch.zeros(B, T, dim, dtype=torch.float32)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    padding_mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)
    return {
        "face": pad_stack(faces, 384),
        "hand1": pad_stack(h1s, 384),
        "hand2": pad_stack(h2s, 384),
        "body": pad_stack(bodies, 14),
        "padding_mask": padding_mask,
        "lengths": lengths,
        "labels": torch.stack(labels),
        "ids": list(ids),
    }
