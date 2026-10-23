"""Derive the gloss vocabulary following the paper's stated protocol.

Read of the paper (arXiv 2310.00196 §3) — the two
filters that define its experimental set:
  - "we only use the videos which have been aligned with ASL-LEX in order to
     maintain consistency among the target gloss labels and complete coverage
     of phonological feature annotations"      -> label_type == asllex only
     (the 5,732 SignBank-aligned videos are EXCLUDED)
  - "Signs with fewer than 5 instances are not given a split"
     -> drop labels with <5 instances; 945 of the 3,149 lemmatized signs,
        leaving ~2,204 classes for the experiments

Frozen default: `paper_protocol`. Reproducing the paper's exact split sizes
is not possible — the release carries ~4% more data than the paper's 84,568
videos (see docs/VOCAB_DERIVATION.md for the measured deltas), so this rule
reproduces its *definition*, not its row counts.

An earlier default (`absb_nofilter`, asllex+signbank, no frequency filter)
predates the paper text; it is retained in RULES as the audit trail. It included SignBank videos the paper excludes
and kept sub-5-instance signs the paper drops.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import utcnow, write_json
from data_config import DOCS_DIR, METADATA_DEDUPED_CSV, PAPER, VOCAB_JSON

# rule -> (label_types considered, min instance count, count on pose_ok only)
#
# `paper_protocol` implements §3 of the paper verbatim:
#   "we only use the videos which have been aligned with ASL-LEX in order to
#    maintain consistency among the target gloss labels" (label_type=asllex)
#   "Signs with fewer than 5 instances are not given a split"  (min count 5)
# The other rules are kept as the audit trail of what else was considered.
RULES = {
    "paper_protocol": (["asllex"], 5, True),
    "absb_nofilter": (["asllex", "signbank"], 1, True),
    "absb_min5": (["asllex", "signbank"], 5, True),
    "asllex_nofilter": (["asllex"], 1, True),
    "asllex_min5_all": (["asllex"], 5, False),
}


def apply_rule(df: pd.DataFrame, rule: str):
    label_types, min_count, pose_only = RULES[rule]
    base = df[df.label_type.isin(label_types)]
    if pose_only:
        base = base[base.pose_ok]
    counts = base.label.value_counts()
    vocab = sorted(counts[counts >= min_count].index)
    n_videos = int(base.label.isin(vocab).sum())
    return vocab, n_videos


def md_table(rows) -> str:
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", default="paper_protocol", choices=RULES)
    args = parser.parse_args()

    df = pd.read_csv(METADATA_DEDUPED_CSV)
    rows = []
    for rule in RULES:
        vocab, n_videos = apply_rule(df, rule)
        rows.append({
            "rule": rule,
            "n_classes": len(vocab),
            "n_videos": n_videos,
            "delta_classes": len(vocab) - PAPER["n_signs_with_split"],
            "delta_videos": n_videos - PAPER["n_asllex_videos"],
        })
        print(rows[-1])

    vocab, n_videos = apply_rule(df, args.rule)
    label_to_id = {label: i for i, label in enumerate(vocab)}
    vocab_sha = hashlib.sha256(
        json.dumps(vocab, ensure_ascii=False).encode()).hexdigest()
    write_json(VOCAB_JSON, {
        "derivation_rule": args.rule,
        "rule_def": {"label_types": RULES[args.rule][0],
                     "min_count": RULES[args.rule][1],
                     "pose_ok_only": RULES[args.rule][2]},
        "n_classes": len(vocab),
        "n_videos": n_videos,
        "delta_vs_paper": {"classes": len(vocab) - PAPER["n_signs_with_split"],
                           "videos": n_videos - PAPER["n_asllex_videos"]},
        "vocab_sha256": vocab_sha,
        "classes": vocab,
        "label_to_id": label_to_id,
        "created_at": utcnow(),
    })
    print(f"\nfrozen rule={args.rule}: {len(vocab)} classes / {n_videos} "
          f"videos -> {VOCAB_JSON} (sha {vocab_sha[:12]})")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc = DOCS_DIR / "VOCAB_DERIVATION.md"
    entry = (f"\n## Run {utcnow()}\n\nPaper targets: {PAPER['n_signs_with_split']} "
             f"classes / {PAPER['n_asllex_videos']} videos. Frozen rule: "
             f"**{args.rule}**.\n\n{md_table(rows)}\n")
    header = ("# Vocabulary derivation\n\nCandidate rules vs the paper's "
              "reported counts; see derive_vocab.py for definitions and the "
              "reasoning behind the frozen default.\n")
    doc.write_text((doc.read_text() if doc.exists() else header) + entry)


if __name__ == "__main__":
    main()
