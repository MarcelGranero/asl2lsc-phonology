# common/lsc_eval — the unified LSC zero-shot handshape evaluation

One model-independent evaluation for the 37-clip LSC handshape benchmark.
Models predict the five ASL-LEX phonological streams; everything after
that — canonicalisation, snapping onto the LSC inventory, ranking,
metrics — happens here, identically for every model family.

```
model (any leaf) ──> predictions/<arm>.tsv ──> evaluate.py ──> results/<arm>.metrics.json
```

The benchmark's recordings and gold labels are not part of this release
(see the root README), so scoring cannot run without them; the code, the
inventory and the frozen mapping are complete.

## The predictions TSV (the only contract a model must meet)

One row per clip, six tab-separated columns, header included:

```
clip_id	selected_fingers	flexion	spread	thumb_position	thumb_contact
```

Values are in the model's own spelling (`evaluate.py` canonicalises both
sides into one shared string space); never emit raw integer class ids.
The constrained decoding variant restricts each head's argmax to values
occurring in the LSC inventory — a target-label-space restriction only,
still zero-shot.

## Running

```bash
python3 evaluate.py predictions/<arm>.tsv     # stdlib python3, no env
bash run_inference_all.sh                     # SLURM: inference for every
                                              # registry arm + scoring gate
```

`inference/checkpoints.json` lists one entry per released checkpoint;
`family` picks the sbatch wrapper (`inference/<family>_arm.sbatch`), which
drives that leaf's own eval script. Checkpoints resolve through
`ASL2LSC_MODELS_DIR`. To evaluate your own checkpoint, add one entry.

## Metrics, in brief

- **Attribute Recall@1** per stream, macro over the 5 streams.
- **Handshape**: the predicted 5-tuple is snapped onto the nearest valid
  LSC handshape under a 5-channel feature distance (`lsc_map.py`); the
  reported series is *expected* accuracy / recall@k under uniform
  tie-breaking. Unmappable predictions count as wrong.
- **Candidate space** = the 37 benchmark-recorded handshapes
  (`inventory/lsc_handshapes.csv`).

`inventory/map_ASL_to_LSC.csv` is the frozen ASL-LEX→LSC mapping
(43-class inventory; naming per SignHub 1.1.1.2, Navarrete-González 2020);
`lsc_map.py` implements the distance and ranking.
