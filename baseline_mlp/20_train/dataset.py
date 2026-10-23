"""Join extracted static hand frames to the Sem-Lex phonological labels.

Frames come from 10_data_prep/extract_frames.py; labels are reused verbatim
from the SHuBERT leaf (results/{split}_labels.csv + label_maps.json), so the
class spaces are literally the same objects the SHuBERT run was scored on.
A clip is kept only if it appears in BOTH -- the intersection is reported.

Stdlib + numpy + torch only (shubert_train_env has no pandas).
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    FEATURE_ORDER_16, FRAMES_DIR, HEADS6, LABEL_MAPS_JSON,
    POPSIGN_FRAMES_DIR, POPSIGN_LABELS16, POPSIGN_LABELS16_NACLASS,
    POPSIGN_VOCAB16, POPSIGN_VOCAB16_NACLASS, SHUBERT_LABELS_CSV,
)

# "class" (NA as a real class) is the default everywhere below; "ignore"
# selects the legacy NA-masked build so older checkpoints keep decoding
# against the maps they were trained with.
def _popsign_files(na_policy: str):
    if na_policy == "class":
        return POPSIGN_LABELS16_NACLASS, POPSIGN_VOCAB16_NACLASS
    if na_policy == "ignore":
        return POPSIGN_LABELS16, POPSIGN_VOCAB16
    raise ValueError(f"unknown na_policy {na_policy!r}")


def load_label_maps() -> dict:
    """{feature: {value_string: class_id}} plus the canonical feature order."""
    with LABEL_MAPS_JSON.open() as fh:
        blob = json.load(fh)
    maps = blob["maps"]
    order = blob.get("feature_order", FEATURE_ORDER_16)
    if order != FEATURE_ORDER_16:
        raise SystemExit(
            "label_maps.json feature_order differs from data_config -- the "
            "head indices would silently shift.\n"
            f"  json: {order}\n  ours: {FEATURE_ORDER_16}"
        )
    return maps


def load_label_maps_popsign(na_policy: str = "class") -> dict:
    """PopSign's OWN {feature: {value: class_id}} maps.

    These are NOT interchangeable with Sem-Lex's: PopSign's vocabulary is
    derived from its 250-gloss subset, so 11 of the 16 heads have a different
    class-id space (e.g. Minor Location 25 classes vs Sem-Lex's 37). Training
    PopSign frames against Sem-Lex ids would decode every one of those heads
    through the wrong table. The feature ORDER is identical in both, and that
    is asserted below.
    """
    _, vocab_path = _popsign_files(na_policy)
    with vocab_path.open() as fh:
        blob = json.load(fh)
    if blob["features"] != FEATURE_ORDER_16:
        raise SystemExit(
            "popsign feature_vocab16.json order differs from data_config -- "
            "head indices would silently shift.\n"
            f"  json: {blob['features']}\n  ours: {FEATURE_ORDER_16}"
        )
    return blob["vocab"]


def _read_labels_popsign(split: str, na_policy: str = "class") -> dict:
    """{clip_id: np.int64 array of 16 class ids}, in FEATURE_ORDER_16 order.

    Already encoded (feature_labels16*.pkl); -1 = genuinely unknown
    (unmatched gloss), skipped by CrossEntropyLoss. Under the mandatory
    "class" policy NA is a real class id; the legacy "ignore" build also
    used -1 for NA. The pickle is corpus-wide, not per split; the split is
    defined by which clips the frame extractor wrote, so every id is
    returned and the Dataset's own intersection does the filtering.
    """
    labels_path, _ = _popsign_files(na_policy)
    with labels_path.open("rb") as fh:
        raw = pickle.load(fh)
    return {k: np.asarray(v, dtype=np.int64) for k, v in raw.items()}


def _read_labels(split: str) -> dict:
    """{video_id: np.int64 array of 16 class ids}, in FEATURE_ORDER_16 order."""
    path = SHUBERT_LABELS_CSV[split]
    out = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = [f"f{i:02d}" for i in range(len(FEATURE_ORDER_16))]
        missing = [c for c in cols if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        for row in reader:
            out[row["video_id"]] = np.array(
                [int(row[c]) for c in cols], dtype=np.int64
            )
    return out


def head_indices(head_set: str) -> list:
    """Positions within the 16-feature vector for the requested head set."""
    if head_set == "all16":
        return list(range(len(FEATURE_ORDER_16)))
    if head_set == "heads6":
        return [FEATURE_ORDER_16.index(name) for name in HEADS6]
    raise ValueError(f"unknown head set {head_set!r}")


def head_names(head_set: str) -> list:
    return [FEATURE_ORDER_16[i] for i in head_indices(head_set)]


def head_class_counts(head_set: str, corpus: str = "semlex",
                      na_policy: str = "class") -> list:
    # Must follow the CORPUS: PopSign's heads are smaller (its vocabulary
    # comes from a 250-gloss subset), so sizing them from Sem-Lex would build
    # heads with unreachable classes and decode ids through the wrong table.
    maps = (load_label_maps() if corpus == "semlex"
            else load_label_maps_popsign(na_policy))
    return [len(maps[name]) for name in head_names(head_set)]


class SemLexFrames(Dataset):
    """(63-d normalised hand vector, per-head class ids) for one split."""

    def __init__(self, split: str, head_set: str = "all16",
                 corpus: str = "semlex", na_policy: str = "class"):
        if corpus not in ("semlex", "popsign"):
            raise ValueError(f"unknown corpus {corpus!r}")
        self.na_policy = na_policy
        frames_dir = FRAMES_DIR if corpus == "semlex" else POPSIGN_FRAMES_DIR
        script = ("extract_frames.py" if corpus == "semlex"
                  else "extract_frames_popsign.py")
        npz_path = frames_dir / f"{split}.npz"
        if not npz_path.exists():
            raise SystemExit(
                f"{npz_path} not found -- run 10_data_prep/{script}"
            )
        blob = np.load(npz_path, allow_pickle=True)
        frame_ids = [str(v) for v in blob["ids"]]
        X = blob["X"].astype(np.float32)

        labels = (_read_labels(split) if corpus == "semlex"
                  else _read_labels_popsign(split, na_policy))
        cols = head_indices(head_set)

        keep_x, keep_y, keep_ids = [], [], []
        for i, vid in enumerate(frame_ids):
            row = labels.get(vid)
            if row is None:
                continue      # clip has a hand frame but no asllex labels
            keep_x.append(X[i])
            keep_y.append(row[cols])
            keep_ids.append(vid)

        self.split = split
        self.corpus = corpus
        self.head_set = head_set
        self.head_names = head_names(head_set)
        self.num_classes = head_class_counts(head_set, corpus)
        self.ids = keep_ids
        self.X = torch.from_numpy(
            np.stack(keep_x) if keep_x else np.zeros((0, 63), np.float32)
        )
        self.y = torch.from_numpy(
            np.stack(keep_y) if keep_y else
            np.zeros((0, len(cols)), np.int64)
        )
        self.stats = {
            "split": split,
            "frames_available": len(frame_ids),
            "labels_available": len(labels),
            "kept": len(keep_ids),
            "dropped_no_label": len(frame_ids) - len(keep_ids),
        }

        # A class id outside its head's map means the label space drifted.
        for h, name in enumerate(self.head_names):
            if len(self.y):
                hi = int(self.y[:, h].max())
                if hi >= self.num_classes[h]:
                    raise SystemExit(
                        f"{split}/{name}: class id {hi} >= "
                        f"{self.num_classes[h]} classes"
                    )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def majority_floors(train_ds, eval_ds) -> dict:
    """Per-head Recall@1 of always predicting the train-majority class.

    The floor every reported number must be read against -- most of the 16
    heads describe movement/location a single still hand frame cannot show.
    """
    floors = {}
    for h, name in enumerate(train_ds.head_names):
        # PopSign masks an absent value with -1 (Sem-Lex has no negative
        # ids). np.bincount rejects negatives outright, and a masked row is
        # not evidence for any class, so both the majority vote and the hit
        # rate are computed over LABELLED rows only.
        tr = train_ds.y[:, h].numpy()
        tr = tr[tr >= 0]
        if tr.size == 0:
            floors[name] = 0.0
            continue
        counts = np.bincount(tr, minlength=train_ds.num_classes[h])
        majority = int(counts.argmax())
        ev = eval_ds.y[:, h].numpy() if len(eval_ds) else np.empty(0, np.int64)
        ev = ev[ev >= 0]
        floors[name] = float((ev == majority).mean()) if ev.size else 0.0
    return floors


if __name__ == "__main__":
    for head_set in ("heads6", "all16"):
        print(f"--- {head_set}: {head_names(head_set)}")
        print("    classes:", head_class_counts(head_set))
    for split in ("test",):
        ds = SemLexFrames(split, "all16")
        print(json.dumps(ds.stats, indent=2))
        print("X", tuple(ds.X.shape), "y", tuple(ds.y.shape))
