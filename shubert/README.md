# SHuBERT fine-tuning (Sem-Lex + PopSign)

Phonological feature classification following the protocol of *SHuBERT:
Self-Supervised Sign Language Representation Learning via Multi-Stream
Cluster Prediction* (Gueuwou et al., ACL 2025,
[arXiv:2411.16765](https://arxiv.org/abs/2411.16765)): 16 ASL-LEX feature
heads on time-averaged SHuBERT representations (BatchNorm → one linear head
per feature), full encoder fine-tuning, trained on the Sem-Lex Benchmark
and, with the `popsign16nac_*` splits, on PopSign ASL v1.0.

The pretrained SHuBERT checkpoint and DINOv2 face/hand weights are not
redistributed — obtain them upstream and point `ASL2LSC_SHUBERT_CKPTS` at
them; expected SHA-256s and the vendored preprocessing code are in
[`external/README.md`](external/README.md).

## Environments

```bash
conda env create -f environment.yml             # shubert_train_env: train/eval
pip install --no-deps -e <SHuBERT checkout>/fairseq   # inside that env
conda env create -f environment_features.yml    # shubert_feature_extraction
```

Submit every job from this directory, after `mkdir -p logs`.

## Evaluate a released checkpoint

With the packed test features and label CSVs in place (steps 1–4 below):

```bash
python 50_eval/evaluate.py --ckpt $ASL2LSC_MODELS_DIR/shubert-semlex-main.pt \
    --split test --run-label semlex_main
python 50_eval/evaluate.py --ckpt $ASL2LSC_MODELS_DIR/shubert-popsign-main.pt \
    --split popsign16nac_test --run-label popsign_main
```

Metrics land in `results/metrics_<split>_<label>.json`. The LSC transfer
evaluation runs through `../common/lsc_eval/run_inference_all.sh`.

## Train (Sem-Lex)

```bash
# 1. labels -- needs the deduped Sem-Lex metadata from the SL-GCN leaf:
#    (cd ../slgcn/semlex/01_data_prep && python dedupe_metadata.py)  # slgcn-env
python 10_data_prep/make_labels.py
# 2. features: transcode -> MediaPipe -> crops -> DINOv2 + body, per split
bash 20_preprocess/submit_pipeline.sh --split test     # then val, train
# 3. manifests, 4. packed blobs
python 30_manifests/build_manifests.py --split val --split test --split train
python 30_manifests/pack_features.py   --split val --split test --split train
# 5. fine-tune, then evaluate (results/metrics_test_main.json)
sbatch 40_train/run_train.sbatch
sbatch 50_eval/run_eval.sbatch
```

## Train (PopSign)

The PopSign labels come from the SL-GCN PopSign leaf
(`../slgcn/popsign/`: data prep, then
`01_data_prep/run_labels16_naclass.sbatch`). The PopSign SHuBERT features
are the same four streams, extracted with the same stages as step 2 into
`$ASL2LSC_POPSIGN_PREPROC/popsign/{dinov2_face,dinov2_hands,body_features}`
(`submit_pipeline.sh` is wired for Sem-Lex's video layout).

```bash
python 10_data_prep/make_popsign_manifests.py      # popsign16 manifests
sbatch 30_manifests/pack_popsign.sbatch            # packed blobs
sbatch 10_data_prep/run_popsign_nac_prep.sbatch    # NA-as-class splits
sbatch 40_train/run_train_popsign_nac.sbatch       # self-chaining, 48 GB GPU
```

Paths resolve through `data_config.py` (`ASL2LSC_*` variables — see the
root README). Label derivation and the NA-as-class rule:
[`docs/LABELS.md`](docs/LABELS.md). What the PopSign arm's numbers can and
cannot mean: [`docs/POPSIGN_NOTES.md`](docs/POPSIGN_NOTES.md).
