# SL-GCN on Sem-Lex

The SL-GCN isolated-sign-recognition baseline of *The Sem-Lex Benchmark*
(Kezar et al., ASSETS 2023,
[arXiv:2310.00196](https://arxiv.org/abs/2310.00196)), run with the
upstream reference implementation
([jackyjsy/SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2))
unmodified, adapted only at the data boundary; phase 2 fine-tunes the gloss
encoder into phonological-feature heads. Upstream is a pinned-commit clone
recreated by the recipe in [`external/README.md`](external/README.md) — a
9-line compatibility patch is the entire code delta.

Our run does **not** reproduce the paper's gloss accuracy from the released
pose files, whose hands are largely missing. The released phonology
checkpoint (`slgcn-semlex-pfr-repose.pt`) is therefore trained on poses we
re-extracted from the Sem-Lex videos.

Submit every job from this directory, after `mkdir -p logs`.

## Released-pose pipeline

```bash
conda env create -f environment.yml          # slgcn-env
# clone + patch upstream per external/README.md
sbatch 01_data_prep/run_data_prep.sbatch     # dedupe → vocab → splits → poses → pack
sbatch 02_train/run_train_smoke.sbatch       # short smoke first
sbatch 02_train/run_train.sbatch             # gloss pretraining (self-chaining)
sbatch 03_eval/run_eval.sbatch               # test top-1/3/5 + per-signer
```

## Re-posed pipeline (the released checkpoint)

```bash
# 1. re-extract poses from the videos (MediaPipe Holistic, CPU array; the
#    env needs mediapipe + opencv), one job per split:
sbatch --export=ALL,SPLIT=train 01_data_prep/run_extract_semlex_poses.sbatch
#    (same for val, test)
# 2. convert + pack into gendata_27_reposed/
sbatch 01_data_prep/run_data_prep_reposed.sbatch
# 3. gloss pretraining on the re-posed tensors (250 epochs)
CONFIG=$PWD/02_train/config/train_joint_reposed.yaml sbatch --export=ALL 02_train/run_train.sbatch
# 4. phonology fine-tune from the best-val gloss checkpoint (100 epochs, seed 1)
INIT=$(python 03_eval/score_semlex.py find-best --exp semlex_joint_reposed | tail -1)
sbatch --export=ALL,REGIME=pfr,EXP_NAME=semlex_pfr_repose,NUM_EPOCH=100,\
CONFIG=$PWD/02_train/config/train_joint_reposed.yaml,INIT_CKPT=$INIT \
       04_phase2_multihead/run_multihead.sbatch
```

Evaluate a phase-2 checkpoint (e.g. the released one) on the re-posed
Sem-Lex test split:

```bash
sbatch --export=ALL,REGIME=pfr,CKPT=$ASL2LSC_MODELS_DIR/slgcn-semlex-pfr-repose.pt,\
RUN_NAME=semlex_pfr_repose,\
EXTRA_ARGS="--gendata-dir $ASL2LSC_SLGCN_SEMLEX_DATA/gendata_27_reposed" \
       04_phase2_multihead/run_eval_multihead.sbatch
```

A re-posed checkpoint must be scored on re-posed tensors; scoring it on the
released-pose `gendata_27/` gives silently lower numbers.

## LSC inputs

The LSC benchmark's `.pose` files become Sem-Lex-layout npys
(`pose_to_553.py`, needs `pose_format`, e.g. in `../popsign`'s env), then
SL-GCN tensors. The baseline MLP reads the same `pose553/` npys.

```bash
L=$ASL2LSC_SLGCN_SEMLEX_DATA/lsc_handshapes/pose553
python 04_phase2_multihead/lsc_handshapes/pose_to_553.py \
    --pose-dir $ASL2LSC_LSC_BENCH/handshape_poses --out-dir $L/test
python 04_phase2_multihead/lsc_handshapes/pose_to_553.py \
    --pose-dir $ASL2LSC_LSC_BENCH/handshape_example_poses --out-dir $L/examples
python 04_phase2_multihead/lsc_handshapes/prepare_lsc_handshapes.py
```

Inference and scoring then run through
`../../common/lsc_eval/run_inference_all.sh`.

Known Sem-Lex release defects (duplicate metadata rows, byte-identical
clips across splits — [leekezar/SemLex#1](https://github.com/leekezar/SemLex/issues/1))
are handled by `01_data_prep/dedupe_metadata.py`. Paths resolve through
`data_config.py` (`ASL2LSC_*` variables — see the root README).
