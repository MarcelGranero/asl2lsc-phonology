# external/

## SAM-SLR-v2 (the SL-GCN reference implementation)

The Sem-Lex paper's own repo (github.com/leekezar/SemLex) is a stub — no code,
configs, or checkpoints have ever been published. The paper's SL-GCN
description ("decoupled GCN layer... spatial and temporal attention...
temporal convolution", 10 blocks, FC decoder) matches
`SL-GCN/model/decouple_gcn_attn.Model` from SAM-SLR (CVPR'21 challenge
winner), so that implementation is what we run — training, preprocessing,
feeder, graph and evaluation all upstream, unmodified.

```bash
git clone https://github.com/jackyjsy/SAM-SLR-v2 external/SAM-SLR-v2
cd external/SAM-SLR-v2
git checkout ef0ab363f27691dcd775169184a5ecd736cb9b38   # pinned 2026-08-03
git apply ../patches/0001-torch2x-pyyaml6-compat.patch
```

The clone is gitignored; re-run the block above to recreate it.

### Patches (correctness only — never style/refactor)

`patches/0001-torch2x-pyyaml6-compat.patch` (9 changed lines, 2 files):

- `SL-GCN/model/decouple_gcn_attn.py`: `nn.init.normal/constant/kaiming_normal`
  → the `_`-suffixed in-place variants. The positional aliases were removed
  from PyTorch long before 2.x; the pinned torch 2.3 (needed for the
  cluster's newest GPUs) has no other option.
- `SL-GCN/main.py`: `yaml.load(f)` → `yaml.load(f, Loader=yaml.FullLoader)`
  (mandatory argument since PyYAML 6).

### What we call and how

| Upstream file | Used as |
|---|---|
| `SL-GCN/data_gen/sign_gendata.py` | imported by `01_data_prep/run_gendata.py` (its `__main__` hardcodes `part='test'`, so we call `gendata()` directly — no patch) |
| `SL-GCN/main.py` | run as-is for training (`02_train/run_train.sbatch`) and test inference (`03_eval/run_eval.sbatch`), CWD = `SL-GCN/` |
| `SL-GCN/{model,feeders,graph}/` | imported by main.py itself |

`main.py` hardcodes `./save_models` and `./work_dir` relative to CWD; the
sbatch scripts symlink those to `${ASL2LSC_SLGCN_SEMLEX_DATA}/`
so heavy artifacts stay off home (no code change).

### License

SAM-SLR-v2 ships a CC-BY-NC-4.0 LICENSE inside `SL-GCN/`. Fine for running
and reporting numbers; do not redistribute derived weights without checking.
