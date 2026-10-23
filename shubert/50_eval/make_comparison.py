"""Render results/comparison.md: our metrics vs paper Table 9 (Sem-Lex).

Tolerance verdict: reproduction accepted when test macro Recall@1 is within
±0.015 of 0.7938 AND no feature is more than 0.03 below its paper value.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import (  # noqa: E402
    PAPER_TABLE9_SEMLEX,
    PAPER_TABLE9_SEMLEX_AVG,
    RESULTS_DIR,
)

MACRO_TOL = 0.015
FEATURE_TOL = 0.03

# The handshape-family features the input-modality ablation targets: the ones
# that should survive on the hand streams alone. The rest act as the control.
HANDSHAPE_FAMILY = (
    "Thumb Contact", "Selected Fingers", "Thumb Position",
    "Flexion", "Spread", "Handshape",
)


def _ablation_section(baseline: dict) -> list[str]:
    """Table every input-modality ablation that has test metrics, vs baseline.

    Discovered from results/metrics_test_*.json rather than hardcoded, so a
    newly added ablation run shows up without editing this file.
    """
    runs = []
    for path in sorted(RESULTS_DIR.glob("metrics_test_*.json")):
        label = path.stem[len("metrics_test_"):]
        if label == "main":
            continue
        m = json.loads(path.read_text())
        if not m.get("keep_channels") or m["keep_channels"] == "all":
            continue  # not an ablation
        runs.append((label, m))
    if not runs:
        return []

    lines = ["", "## Input-modality ablations (test split)", "",
             "Dropped channels are filled with the pretrained `mask_emb`. "
             "Δ is vs the full 4-channel baseline.", ""]
    header = "| Feature | baseline |" + "".join(f" {r} | Δ |" for r, _ in runs)
    lines += [header, "|---|---|" + "---|---|" * len(runs)]

    def row(name: str, bold: bool = False) -> str:
        b = baseline[name]
        cells = f"| {'**' if bold else ''}{name}{'**' if bold else ''} | {b:.4f} |"
        for _, m in runs:
            o = m["per_feature_recall1"][name]
            cells += f" {o:.4f} | {o - b:+.4f} |"
        return cells

    lines.append("| *handshape family* | |" + " |" * (2 * len(runs)))
    for name in HANDSHAPE_FAMILY:
        lines.append(row(name))
    lines.append("| *control* | |" + " |" * (2 * len(runs)))
    for name in PAPER_TABLE9_SEMLEX:
        if name not in HANDSHAPE_FAMILY:
            lines.append(row(name))

    # group means + macro
    def mean_row(label: str, names) -> str:
        names = list(names)
        b = sum(baseline[n] for n in names) / len(names)
        cells = f"| **{label}** | **{b:.4f}** |"
        for _, m in runs:
            o = sum(m["per_feature_recall1"][n] for n in names) / len(names)
            cells += f" **{o:.4f}** | **{o - b:+.4f}** |"
        return cells

    lines.append(mean_row("mean handshape", HANDSHAPE_FAMILY))
    lines.append(mean_row("mean control",
                          [n for n in PAPER_TABLE9_SEMLEX
                           if n not in HANDSHAPE_FAMILY]))
    lines.append(mean_row("macro (all 16)", PAPER_TABLE9_SEMLEX))
    lines += [
        "",
        "Reading it: the ablation supports *\"handshape features do not need "
        "face/body\"* only if the handshape family degrades MEANINGFULLY LESS "
        "than the control. If both fall by a similar amount, the streams are "
        "redundant for every feature (or the heads exploit label correlation) "
        "— a weaker claim than the ablation set out to make.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-metrics", type=Path,
                        default=RESULTS_DIR / "metrics_test_main.json")
    parser.add_argument("--val-metrics", type=Path,
                        default=RESULTS_DIR / "metrics_val_main.json")
    args = parser.parse_args()

    test = json.loads(args.test_metrics.read_text())
    val = (json.loads(args.val_metrics.read_text())
           if args.val_metrics.exists() else None)
    ours = test["per_feature_recall1"]

    lines = [
        "# Comparison vs SHuBERT paper Table 9 (Sem-Lex)",
        "",
        f"Checkpoint: `{test['ckpt']}` (epoch {test['ckpt_epoch']}), "
        f"test n={test['n']}.",
        "",
        "| Feature | Paper | Ours (test) | Δ |" + (" Ours (val) |" if val else ""),
        "|---|---|---|---|" + ("---|" if val else ""),
    ]
    below = []
    for name, paper in PAPER_TABLE9_SEMLEX.items():
        o = ours[name]
        d = o - paper
        if d < -FEATURE_TOL:
            below.append((name, d))
        row = f"| {name} | {paper:.4f} | {o:.4f} | {d:+.4f} |"
        if val:
            row += f" {val['per_feature_recall1'][name]:.4f} |"
        lines.append(row)
    macro = test["macro_recall1"]
    d_macro = macro - PAPER_TABLE9_SEMLEX_AVG
    row = (f"| **Average** | **{PAPER_TABLE9_SEMLEX_AVG:.4f}** | "
           f"**{macro:.4f}** | **{d_macro:+.4f}** |")
    if val:
        row += f" **{val['macro_recall1']:.4f}** |"
    lines.append(row)

    ok = abs(d_macro) <= MACRO_TOL and not below
    lines += [
        "",
        f"**Verdict: {'REPRODUCED' if ok else 'OUTSIDE TOLERANCE'}** — "
        f"macro Δ {d_macro:+.4f} (tol ±{MACRO_TOL}); "
        f"features >{FEATURE_TOL} below paper: "
        f"{[f'{n} ({d:+.3f})' for n, d in below] or 'none'}.",
    ]
    if not ok:
        lines += ["", "Next ablations: pooling layer, raw-vs-deduped "
                  "metadata, layerdrop 0.05→0."]

    lines += _ablation_section(ours)

    out = RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
