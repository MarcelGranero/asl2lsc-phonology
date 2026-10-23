# SL-GCN on PopSign

The same unmodified upstream SL-GCN
([jackyjsy/SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2)) as
[`slgcn/semlex`](../semlex), trained on the PopSign ASL v1.0 game corpus
(250 classes, signer-independent splits). There is no published SL-GCN
PopSign number, so this is a new baseline, not a reproduction. Phase 2
fine-tunes the gloss encoder into phonological-feature heads; the LSC
zero-shot arm emits predictions for `common/lsc_eval`.

Submit every job from this directory, after `mkdir -p logs`.

## Run

```bash
conda env create -f environment.yml       # popsign-slgcn-env
# reuse the sibling's pinned upstream clone (a separate copy on purpose:
# main.py writes save_models/ and work_dir/ inside the clone):
mkdir -p external && cp -r ../semlex/external/SAM-SLR-v2 external/
python -m pytest tests/ -q                # data-boundary tests

sbatch 01_data_prep/run_data_prep.sbatch  # manifest → vocab → splits → poses → pack → QA
# inspect the docs/overlay_qa/*.png renders it produces, THEN:
sbatch 02_train/run_train.sbatch          # gloss pretraining (self-chaining)
sbatch 03_eval/run_eval.sbatch            # test top-1/3/5 + per-signer

# phase 2: 16 ASL-LEX heads, NA-as-class labels (the released checkpoint;
# 250 epochs, seed 1) from the best-val gloss checkpoint
python 01_data_prep/build_feature_labels.py
sbatch 01_data_prep/run_labels16_naclass.sbatch
INIT=$(python 03_eval/score_popsign.py find-best --exp popsign_joint | tail -1)
sbatch --export=ALL,REGIME=pfr,EXP_NAME=popsign_pfr16_nac,NUM_EPOCH=250,INIT_CKPT=$INIT,\
EXTRA_ARGS="--feature-labels $ASL2LSC_SLGCN_POPSIGN_DATA/feature_labels16_naclass.pkl --feature-vocab $PWD/results/feature_vocab16_naclass.json" \
       04_phase2_multihead/run_multihead.sbatch
```

Evaluate a phase-2 checkpoint on the PopSign test split (for a 6-head
checkpoint trained with the default labels, drop `EXTRA_ARGS`):

```bash
sbatch --export=ALL,REGIME=pfr,CKPT=$ASL2LSC_MODELS_DIR/slgcn-popsign-pfr16-nac.pt,\
RUN_NAME=popsign_pfr16_nac,\
EXTRA_ARGS="--feature-labels $ASL2LSC_SLGCN_POPSIGN_DATA/feature_labels16_naclass.pkl --feature-vocab $PWD/results/feature_vocab16_naclass.json" \
       04_phase2_multihead/run_eval_multihead.sbatch
```

The LSC transfer arm (`05_lsc_eval/eval_lsc_handshapes.py`, run through
`../../common/lsc_eval/run_inference_all.sh`) reads the benchmark's raw
`.pose` files from `$ASL2LSC_LSC_BENCH`.

Caveats that bound what the numbers mean:

- **PopSign feature labels are type-level** (joined from ASL-LEX by gloss),
  so high in-domain feature recall partly reflects gloss recognition — see
  [`../../shubert/docs/POPSIGN_NOTES.md`](../../shubert/docs/POPSIGN_NOTES.md).
- PopSign results are **not comparable** to Sem-Lex ones: different
  dataset, class count, label spaces and framing.

Paths resolve through `data_config.py` (`ASL2LSC_*` variables — see the
root README).
