# Baseline MLP (Sem-Lex + PopSign)

A single-frame, single-hand landmark baseline: one 63-float MediaPipe hand
frame (21 landmarks × 3, wrist-centred, scale-normalised) through two
Linear→BatchNorm→ReLU→Dropout blocks, one linear head per phonological
feature. Architecture after Carbo & Nalisnick (EMNLP 2025,
[arXiv:2509.18309](https://arxiv.org/abs/2509.18309)), reimplemented in
[`20_train/model.py`](20_train/model.py) from the published description.

Trained on Sem-Lex and, as a second arm, on PopSign ASL v1.0 (same
architecture and trainer, different frame extractor and label space).
Runs in `shubert_train_env` (`../shubert/environment.yml`; fairseq is not
needed here).

## Inputs from the other leaves

- Sem-Lex labels: `../shubert/results/{split}_labels.csv`
  (`../shubert/10_data_prep/make_labels.py`).
- Sem-Lex hand landmarks: the SHuBERT pipeline's MediaPipe stage
  (`$ASL2LSC_SHUBERT_DATA/<split>/kpe_poses/`).
- PopSign landmarks and labels: the SL-GCN PopSign leaf
  (`$ASL2LSC_SLGCN_POPSIGN_DATA/wholebody133/`, `feature_labels16_naclass.pkl`).
- LSC clips: `(T,553,3)` npys from the SL-GCN Sem-Lex leaf
  (`04_phase2_multihead/lsc_handshapes/pose_to_553.py` →
  `$ASL2LSC_SLGCN_SEMLEX_DATA/lsc_handshapes/pose553/{test,examples}`).

## Evaluate a released checkpoint (CPU, seconds)

```bash
python 20_train/model.py                      # architecture self-test
python 30_eval/evaluate_semlex.py --run-name all16 --split test \
    --checkpoint $ASL2LSC_MODELS_DIR/mlp-semlex-all16.pt
python 30_eval/evaluate_semlex.py --run-name popsign_all16_nac_s42 --split test \
    --checkpoint $ASL2LSC_MODELS_DIR/mlp-popsign-all16-nac.pt
```

The checkpoint's `corpus` field selects the frames and label maps.

## Train

Submit from this directory after `mkdir -p logs`.

```bash
# Sem-Lex arm
sbatch 10_data_prep/run_extract.sbatch                        # one frame per clip
sbatch --job-name=all16 20_train/run_train.sbatch all16 all16 multi 42 semlex
sbatch --job-name=all16 30_eval/run_eval.sbatch all16         # test + LSC

# PopSign arm (NA-as-class labels)
sbatch 10_data_prep/run_extract_popsign.sbatch
sbatch --job-name=popsign_all16_nac_s42 20_train/run_train.sbatch \
    all16 popsign_all16_nac_s42 multi 42 popsign
sbatch --job-name=popsign_all16_nac_s42 30_eval/run_eval_popsign.sbatch \
    popsign_all16_nac_s42
```

`20_train/submit_popsign_grid.sh` submits the PopSign seed grid.

Caveat: this model sees one still frame of one hand — no face, body, other
hand, or motion. Of the 16 trained heads, 11 describe properties it cannot
observe; their scores measure a learned class prior, not recognition, and
the eval prints every head next to its train-majority floor for that
reason.

Labels are reused verbatim from the `shubert/` leaf so all three model
families score in identical class spaces. Paths resolve through
`data_config.py` (`ASL2LSC_*` variables — see the root README).
