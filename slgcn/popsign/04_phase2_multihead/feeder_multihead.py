"""Feeder that adds phonological-feature targets to upstream's batches.

Least-invasive by construction: this subclasses upstream's `Feeder` and adds
NOTHING to the data pipeline.  All loading, augmentation (random_choose,
mirror, normalisation, shift) and the (C, T, V, M) tensor layout are inherited
verbatim — `__getitem__` calls `super()` and only appends a label vector
looked up by sample name.

Upstream returns `(data, label, index)`; we return
`(data, label, feature_vector, index)`.  The extra element is inserted BEFORE
`index` so that any code doing `data, label = batch[:2]` keeps working.

The join is by `video_id`: upstream's `{split}_label.pkl` stores
`(sample_name, label)` where sample_name IS the video_id, and the sidecar
built by `01_data_prep/build_feature_labels.py` is keyed the same way.  A
sample with no sidecar entry gets an all -1 vector (fully masked) rather than
raising, so a partially-built sidecar degrades to gloss-only training instead
of crashing mid-epoch — the count is reported once at load time.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

# Upstream's package dir must be importable; the sbatch files run from
# external/SAM-SLR-v2/SL-GCN so `feeders` resolves there.
from feeders.feeder import Feeder


class FeederMultihead(Feeder):
    """Upstream Feeder + a (n_features,) int64 target vector per sample."""

    def __init__(self, *args, feature_label_path=None, **kwargs):
        if feature_label_path is None:
            raise ValueError("feature_label_path is required")
        self.feature_label_path = feature_label_path
        super().__init__(*args, **kwargs)
        self._load_feature_labels()

    def _load_feature_labels(self):
        with open(self.feature_label_path, "rb") as f:
            payload = pickle.load(f)
        # Two sidecar schemas exist: build_feature_labels.py (6 features)
        # wraps the map as {"features": [...], "labels": {clip: vec}};
        # build_feature_labels16.py writes the bare {clip: vec} map (the
        # schema baseline_mlp and shubert consume). Accept both; feature
        # NAMES are authoritative in the vocab json either way (the trainer
        # reads them from there), so here they are informational only.
        if isinstance(payload.get("labels"), dict):
            self.features = payload["features"]
            lookup = payload["labels"]
        else:
            self.features = None
            lookup = payload
        self.n_features = (len(self.features) if self.features is not None
                           else len(next(iter(lookup.values()))))
        missing = 0
        table = np.empty((len(self.sample_name), self.n_features), dtype=np.int64)
        for i, name in enumerate(self.sample_name):
            vector = lookup.get(name)
            if vector is None:
                missing += 1
                table[i] = -1
            else:
                table[i] = vector
        self.feature_label = table
        self.n_missing_feature_labels = missing
        if missing:
            print(
                f"[FeederMultihead] WARNING: {missing}/{len(self.sample_name)} "
                f"samples have no phonological labels; masked with -1"
            )

    def __getitem__(self, index):
        data_numpy, label, index = super().__getitem__(index)
        return data_numpy, label, self.feature_label[index], index
