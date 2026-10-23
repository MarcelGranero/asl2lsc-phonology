# Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes

<p align="center">
  <a href="https://arxiv.org/abs/2609.18772"><img src="https://img.shields.io/badge/arXiv-2609.18772-b31b1b.svg" alt="Paper on arXiv"></a>
  <a href="https://marcelgranero.github.io/cross-lingual-handshapes/"><img src="https://img.shields.io/badge/Project%20Page-cross--lingual--handshapes-2c7cb0.svg" alt="Project page"></a>
  <a href="https://huggingface.co/MarcelGranero/asl2lsc-phonology"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffd21e.svg" alt="Models on Hugging Face"></a>
</p>

Code and trained models for the paper
**"Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes"**
(2nd Workshop on Sign Language Processing, WSLP @ EMNLP 2026).
Phonological-feature recognizers are trained on two American Sign Language
corpora (Sem-Lex, PopSign ASL v1.0) and evaluated zero-shot on Catalan Sign
Language (LSC) handshapes, decoding through a shared ASL-LEX-based
phonological feature space.

## Repository layout

| Path | Contents |
|---|---|
| `shubert/` | SHuBERT fine-tuning: 16 phonological-feature heads on the pretrained SSL encoder (paper protocol), for Sem-Lex and PopSign |
| `slgcn/semlex/`, `slgcn/popsign/` | SL-GCN (upstream SAM-SLR-v2, cloned by recipe — see `slgcn/semlex/external/README.md`): gloss pretraining and phonological multi-head fine-tuning |
| `baseline_mlp/` | Single-frame, single-hand landmark MLP baseline |
| `common/lsc_eval/` | The unified LSC zero-shot evaluation: ASL-LEX→LSC mapping, constrained decoding, scoring |
| `tools/` | `smoke_test.py` (env gating, syntax, optional checkpoint load), run in CI |

Each leaf is an independently runnable pipeline (own SLURM scripts, own
`data_config.py`, own README with the exact commands). Conda environments:
`shubert/environment.yml` (SHuBERT, baseline MLP, LSC inference),
`shubert/environment_features.yml` (SHuBERT preprocessing),
`slgcn/semlex/environment.yml` and `slgcn/popsign/environment.yml`
(SL-GCN; the latter also reads the LSC `.pose` files). The SLURM scripts
source `~/miniconda3/etc/profile.d/conda.sh` and use generic partition
names (`short`, `medium`, `high`); adapt both to your cluster.

## Data

No dataset is redistributed here — obtain the corpora upstream:

| Dataset | Where | License / access |
|---|---|---|
| Sem-Lex Benchmark | [github.com/leekezar/SemLex](https://github.com/leekezar/SemLex) | CC BY-NC-SA 4.0, gated |
| PopSign ASL v1.0 | [signdata.cc.gatech.edu](https://signdata.cc.gatech.edu) | CC BY 4.0 |
| Kaggle `asl-signs` landmarks | [kaggle.com/competitions/asl-signs](https://www.kaggle.com/competitions/asl-signs) (only the class-index map is read) | Kaggle terms — not redistributable |
| ASL-LEX 2.0 | [asl-lex.org](https://asl-lex.org) (reaches this pipeline through the Sem-Lex metadata) | CC BY-NC 4.0 |
| SHuBERT pretrained checkpoints | The SHuBERT authors' release (expected SHA-256s in `shubert/external/README.md`) | see Third-party below |

Then point these environment variables at your local copies — every script
resolves paths through them (per-leaf `data_config.py`):

| Variable | Points at |
|---|---|
| `ASL2LSC_SEMLEX_ROOT` | Sem-Lex release (videos, poses, metadata) |
| `ASL2LSC_SHUBERT_DATA` | SHuBERT working data (features, manifests, runs) |
| `ASL2LSC_SHUBERT_CKPTS` | SHuBERT pretrained checkpoint + DINOv2 face/hand weights |
| `ASL2LSC_SLGCN_SEMLEX_DATA` / `ASL2LSC_SLGCN_POPSIGN_DATA` | SL-GCN working data (poses, gendata, checkpoints) |
| `ASL2LSC_MLP_DATA` | MLP frames + runs |
| `ASL2LSC_POPSIGN_PREPROC` | PopSign preprocessing tree: videos and landmark parquet, plus the SHuBERT feature streams under `popsign/` |
| `ASL2LSC_POPSIGN_JSON` | PopSign per-clip label JSONs |
| `ASL2LSC_KAGGLE_ASL_SIGNS` | Kaggle `asl-signs` release |
| `ASL2LSC_MODELS_DIR` | Directory holding the released checkpoints |
| `ASL2LSC_LSC_BENCH` | LSC handshape benchmark root (forthcoming release; see below) |
| `ASL2LSC_LSC_LABELS` | The benchmark's 37-clip gold labels and per-split feature manifests |

The only data files included are the label-space definitions the code
reads (`shubert/results/*label_maps*.json`, `slgcn/*/results/*vocab*.json`);
their class names derive from ASL-LEX 2.0 codes (CC BY-NC 4.0) and the
PopSign gloss set (CC BY 4.0). Everything else regenerates from your own
copies via the data-prep stages.

## Models

The trained checkpoints, with per-checkpoint licenses and citation
requirements, are on Hugging Face:
**https://huggingface.co/MarcelGranero/asl2lsc-phonology**.
Download them into one directory, point `ASL2LSC_MODELS_DIR` at it, and
each leaf's evaluation entry point (or
`common/lsc_eval/run_inference_all.sh`) consumes them directly. See the
paper's Tables 1–2 for the headline results.

## The LSC benchmark

The 37-handshape LSC benchmark (videos and gold labels) is unpublished
collaborator data, forthcoming as its own release; it is not redistributed
here and `common/lsc_eval` scoring requires it. What IS here: the full
evaluation code, the LSC handshape inventory and the frozen ASL-LEX→LSC
mapping (`common/lsc_eval/inventory/`). The code expects, under
`ASL2LSC_LSC_BENCH`: `handshape_{,examples_}manifest.csv`, the raw `.pose`
files (`handshape_poses/`, `handshape_example_poses/`), the SHuBERT input
features (`features/`) and `labels/lsc_handshape_distance.py`; under
`ASL2LSC_LSC_LABELS`: `test_labels_audit_v2.csv` and
`shubert_inference_handshape_{test,examples}_v2.csv`.

## Licensing

**Code**: MIT (see `LICENSE`). **Model weights**: per-checkpoint licenses
on the Hugging Face card (CC BY-NC-SA 4.0 / CC BY-NC 4.0 / research-only —
each inherits the most restrictive license among its training inputs).
**Included label-space files**: ASL-LEX-derived, CC BY-NC 4.0 attribution
applies.

### Third-party code and licenses

| Component | License | Notes |
|---|---|---|
| SHuBERT (Gueuwou et al., ACL 2025, [arXiv:2411.16765](https://arxiv.org/abs/2411.16765)) | MIT (per the upstream README) | Preprocessing + model definition vendored in `shubert/external/` with per-file SHA-256 provenance; the pretrained checkpoint and fine-tuned DINOv2 weights are NOT redistributed — obtain upstream |
| SAM-SLR-v2 / SL-GCN (Jiang et al., CVPRW 2021) | `SL-GCN/` component: CC BY-NC 4.0, academic use | Never vendored — pinned-commit clone by recipe + a 9-line compat patch (`slgcn/semlex/external/`) |
| MediaPipe (Google) | Apache-2.0 (code and models) | Two `.task` landmarker bundles redistributed under this notice |
| DINOv2 / fairseq (Meta AI) | Apache-2.0 / MIT | Runtime dependencies |
| Handshape-GNN baseline MLP (Carbo & Nalisnick, EMNLP 2025, [arXiv:2509.18309](https://arxiv.org/abs/2509.18309)) | no public code release | Architecture source for `baseline_mlp/20_train/model.py`, reimplemented from the paper; no upstream code included |
| Sem-Lex Benchmark (Kezar et al., ASSETS 2023) | CC BY-NC-SA 4.0, gated | Not redistributed; label CSVs regenerate via `shubert/10_data_prep/make_labels.py` |
| ASL-LEX 2.0 (Sehyr et al., 2021) | CC BY-NC 4.0 | Feature codings reach this repo through the Sem-Lex/PopSign metadata; class names appear in the shipped label maps |
| PopSign ASL v1.0 (Starner et al., NeurIPS 2023) | CC BY 4.0 | Not redistributed; the Kaggle dump is not redistributable — only its 250-class ordering is reflected in `slgcn/popsign/results/vocab.json` (rebuild: `derive_vocab.py`) |
| LSC handshape benchmark | All rights reserved by its creators | Withheld until its own release |
| SignHub LSC grammar ([thesignhub.eu](https://thesignhub.eu)) | © SignHub | Inventory naming per SignHub 1.1.1.2 (Navarrete-González, 2020); no SignHub images included |

### Ethics

Both ASL corpora contain identifiable signers and are used under their
release terms; this repository redistributes no video, image, or per-frame
data of any signer. The LSC recordings were made with the signer's informed
consent and are not redistributed. The released models predict phonological
feature classes; they are not signer-identification systems and must not be
used to identify individuals.

## Citation

```bibtex
@inproceedings{granero-moya-etal-2026-zero-shot,
  title     = {Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes},
  author    = {Granero-Moya, Marcel and del Corral Farrar{\'o}s, Carolina
               and Haro, Gloria and Ballester, Coloma and Marques, Ricardo},
  booktitle = {Proceedings of the Second Workshop on Sign Language Processing (WSLP)},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary (Co-located with EMNLP 2026)},
  publisher = {Association for Computational Linguistics},
  eprint    = {2609.18772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

If you use a released checkpoint, please also cite the work it builds on
(table above): SHuBERT checkpoints → Gueuwou et al. (2025); SL-GCN →
Jiang et al. (2021); MLP → Carbo & Nalisnick (2025); and the corpora that
supervised them — Kezar et al. (2023), Starner et al. (2023), Sehyr et
al. (2021).

Contact: Marcel Granero-Moya (marcel.granero@upf.edu).
