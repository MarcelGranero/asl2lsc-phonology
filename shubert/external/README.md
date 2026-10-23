# external/ — vendored upstream SHuBERT code

`external/shubert/` holds the SHuBERT preprocessing scripts and model
definition, copied from the upstream release. Two files differ from upstream
(see the table): `kpe_mediapipe.py` comes from a lab fork with robustness
fixes, and the repo-root `data_config.py` has its default data root replaced
(nothing reads that default: every vendored script gets its paths as CLI
flags). The directory mirrors the upstream repo shape (`dataset/`,
`features/`, repo-root `data_config.py`) so the scripts'
`sys.path.insert(repo_root)` + `from dataset.utils import …` /
`from data_config import …` resolve exactly as upstream.

`shubert_model.py` is the **pretraining** SHuBERT model
(`fairseq/examples/shubert/models/shubert.py`) under a non-colliding module
name. Import it directly (`sys.path` + `import shubert_model`), never as
`examples.shubert.models.shubert`: an environment whose fairseq install
exposes a modified `examples` package can resolve that name to a `shubert.py`
rewritten into a supervised classification variant, which cannot load the
pretraining checkpoint.

## Provenance

sha256 prefixes of the upstream source files at copy time (2026-08-03).

| File | Upstream source | sha256 (source) |
|---|---|---|
| `shubert/data_config.py` | `data_config.py` (default data root replaced in this copy) | `8e38a747…a5b5b02` |
| `shubert/dataset/utils.py` | `dataset/utils.py` | `553126eb…9c22d2` |
| `shubert/dataset/kpe_mediapipe.py` | lab fork of `dataset/kpe_mediapipe.py` (CLI-identical; adds streamed per-frame JSON with PID-unique tmp+rename, per-video detector re-init, per-video exception guard — avoids upstream's OOM on long videos) | `9012d0de…65a5be` |
| `shubert/dataset/crop_face.py` | `dataset/crop_face.py` | `2c1f5a5a…9713c3` |
| `shubert/dataset/crop_hands.py` | `dataset/crop_hands.py` | `c0db43d8…554432b` |
| `shubert/features/dinov2_features.py` | `features/dinov2_features.py` | `8da25645…08c36a3` |
| `shubert/features/body_features.py` | `features/body_features.py` | `fa0f8b95…9f23d1b` |
| `shubert/shubert_model.py` | `fairseq/examples/shubert/models/shubert.py` (pretraining variant) | `9c03a518…70c227` |
| `shubert/models/face_landmarker_v2_with_blendshapes.task` | `models/` (MediaPipe) | `64184e22…0bc9ff` |
| `shubert/models/hand_landmarker.task` | `models/` (MediaPipe) | `fbc2a300…607cde1` |

## Not redistributed (obtain upstream, place under `$ASL2LSC_SHUBERT_CKPTS`)

| Asset | File | sha256 |
|---|---|---|
| SHuBERT pretrained (400k updates, YouTube-ASL, random masking) | `checkpoint_836_400000.pt` | `9087f476…20cd7d` |
| DINOv2 face (fine-tuned, from the SHuBERT release) | `dinov2face.pth` | `ebf07e57…be59bcb` |
| DINOv2 hand (fine-tuned, from the SHuBERT release) | `dinov2hand.pth` | `c6ffaf9f…640d85` |

Also asserted at runtime: the deduped Sem-Lex metadata
(`slgcn/semlex/results/metadata_deduped.csv`, sha256 `5d471306…858985`),
produced by `slgcn/semlex/01_data_prep/dedupe_metadata.py`. Full hashes of
both asserted files are in `../data_config.py`.

## Known upstream warts we work around (never edit)

- `kpe_mediapipe.py` calls `dump_file_list(files_list.parent)` at startup →
  work lists live in a dedicated `work_lists/` dir so the rewrite is harmless.
- `--write-output-list` flags use `type=bool` → any string is truthy; the
  scripts drop `<dir>.list.gz` files into their output dirs. We ignore those
  and regenerate stage lists ourselves (`20_preprocess/make_stage_lists.py`).
- Crop scripts convert decord's RGB output with `COLOR_BGR2RGB` before writing
  with cv2 → crops are channel-swapped relative to "correct". Faithful = keep:
  the DINOv2 fine-tuned weights were trained on crops produced exactly this way.
- `crop_face.py`/`crop_hands.py` skip leading frames until the first pose
  detection → crop videos can be shorter than the source; handled by
  truncate-to-min at manifest time.
- `shubert_model.py` `forward()` requires dummy `label_*` tensors and logs
  stream shapes at INFO on every call (the trainer silences that logger).
