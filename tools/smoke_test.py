#!/usr/bin/env python3
"""Fresh-clone smoke test for the assembled release tree.

Usage: smoke_test.py <release_dir>

1. Each leaf's data_config must fail with a clear ASL2LSC_* message (not a
   dead absolute path) when the env is unset, and import cleanly when set.
2. Every shipped python file must parse (ast only — no imports, no writes),
   and no ${ASL2LSC_*} placeholder may sit where nothing expands it: in an
   #SBATCH line (SLURM does not expand variables there) or as a python
   string literal.
3. Optional, when ASL2LSC_SMOKE_CKPT_DIR points at the released checkpoints
   and torch is importable: the two MLP checkpoints must strict-load into
   the shipped architecture and produce the head shapes their config
   declares. (The other families need their heavy envs/data; the MLP is the
   one whose full load path fits a bare python + CPU torch.)

Runs with stdlib python3 only — mirrors what an anonymous cloner can do;
step 3 upgrades the check when torch is available.
"""

import ast
import os
import pickle
import subprocess
import sys
from pathlib import Path

LEAF_CONFIGS = [
    "shubert/data_config.py",
    "slgcn/semlex/data_config.py",
    "slgcn/popsign/data_config.py",
    "baseline_mlp/data_config.py",
]

CLEAN_ENV = {k: v for k, v in os.environ.items()
             if not k.startswith("ASL2LSC_")}
# The import checks must not write __pycache__ into the certified tree —
# neither the subprocess probes nor the in-process checkpoint check.
CLEAN_ENV["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=str(cwd), env=env or CLEAN_ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def check_mlp_checkpoints(release, ckpt_dir, failures):
    try:
        import torch
    except ImportError:
        print("  (checkpoint check skipped: torch not installed)")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "release_mlp_model", release / "baseline_mlp/20_train/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for name in ("mlp-semlex-all16.pt", "mlp-popsign-all16-nac.pt"):
        path = Path(ckpt_dir) / name
        if not path.is_file():
            failures.append(f"{name}: not under ASL2LSC_SMOKE_CKPT_DIR")
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, pickle.UnpicklingError):
            # torch < 1.13 has no weights_only; some checkpoints stash a
            # numpy scalar in their metrics, which strict weights_only
            # rejects. These are the release's own sha256-manifested files.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = mod.build_model(
            cfg["head_set"], cfg["num_classes"], arch=cfg["arch"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"])
        model.load_state_dict(ckpt["model_state_dict"])  # strict by default
        model.eval()
        with torch.no_grad():
            outs = model(torch.randn(4, 63))
        want = [int(n) for n in cfg["num_classes"]]
        if [o.shape[1] for o in outs] != want:
            failures.append(f"{name}: head shapes do not match its config")
        else:
            print(f"  {name}: strict load + forward OK ({len(want)} heads)")


def main():
    release = Path(sys.argv[1]).resolve()
    failures = []

    # 1. data_config behavior without and with env.
    dummy_env = dict(CLEAN_ENV)
    for var in ("ASL2LSC_SEMLEX_ROOT", "ASL2LSC_SHUBERT_DATA",
                "ASL2LSC_SHUBERT_CKPTS", "ASL2LSC_SLGCN_SEMLEX_DATA",
                "ASL2LSC_SLGCN_POPSIGN_DATA", "ASL2LSC_POPSIGN_PREPROC",
                "ASL2LSC_POPSIGN_JSON", "ASL2LSC_KAGGLE_ASL_SIGNS",
                "ASL2LSC_MLP_DATA", "ASL2LSC_LSC_BENCH"):
        dummy_env[var] = "/nonexistent/" + var.lower()
    for cfg in LEAF_CONFIGS:
        leaf = release / Path(cfg).parent
        r = run([sys.executable, "-c", "import data_config"], cwd=leaf)
        if r.returncode == 0:
            failures.append(f"{cfg}: imported with NO env set "
                            f"(site defaults not stripped?)")
        elif "ASL2LSC_" not in r.stderr:
            failures.append(f"{cfg}: unset-env failure does not name the "
                            f"ASL2LSC_* variable:\n{r.stderr[-500:]}")
        r = run([sys.executable, "-c", "import data_config"], cwd=leaf,
                env=dummy_env)
        if r.returncode != 0:
            failures.append(f"{cfg}: does not import with env set:\n"
                            f"{r.stderr[-500:]}")

    # 2. Every shipped python file parses.
    checked = 0
    for py in sorted(release.rglob("*.py")):
        try:
            ast.parse(py.read_text(), filename=str(py))
            checked += 1
        except SyntaxError as e:
            failures.append(f"syntax error: {py.relative_to(release)}: {e}")
    print(f"  parsed {checked} python files")

    placeholder = "${" + "ASL2LSC_"   # split so this line does not match itself
    for f in sorted(release.rglob("*")):
        if not f.is_file() or ".git" in f.parts or f.suffix not in (
                ".py", ".sbatch", ".slurm", ".sh"):
            continue
        for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if placeholder not in line:
                continue
            if line.lstrip().startswith("#SBATCH") or (
                    f.suffix == ".py"
                    and any(q + placeholder in line for q in "\"'")):
                failures.append(f"unexpanded {placeholder}...}} at "
                                f"{f.relative_to(release)}:{n}")

    # 3. Released MLP checkpoints load into the shipped architecture.
    ckpt_dir = os.environ.get("ASL2LSC_SMOKE_CKPT_DIR")
    if ckpt_dir:
        check_mlp_checkpoints(release, ckpt_dir, failures)

    if failures:
        print(f"smoke_test: {len(failures)} failure(s)")
        for f in failures:
            print("  FAIL:", f)
        sys.exit(1)
    print("smoke_test: PASS")


if __name__ == "__main__":
    main()
