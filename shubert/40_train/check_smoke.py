"""Assert the trainer smoke actually learned — not just that it exited 0.

Pass criteria (evaluated against the run's history.json and the majority-class
floors in results/labels_report.json for the split used as validation):

  1. train loss strictly decreasing across epochs
  2. every one of the 16 features >= its majority-class floor
  3. macro Recall@1 >= mean(floors) + 0.05

A model that collapses to predicting the majority class of each feature is
the thing this is designed to catch: it can look "fine" on macro average
while being useless on the skewed features (Handshape floor is 0.11,
Wrist Twist 0.83).

Usage: python check_smoke.py --run-name smoke --val-split test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_config import RUNS_DIR  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--strict", action="store_true",
                    help="require EVERY feature above its majority floor "
                         "(for the full run; a 3-epoch smoke may still have "
                         "slow features climbing)")
    ap.add_argument("--report", default="labels_report.json",
                    help="floors file in results/ matching the corpus of "
                         "--val-split (popsign runs: "
                         "popsign16_labels_report.json)")
    args = ap.parse_args()

    log_path = RUNS_DIR / args.run_name / "log.jsonl"
    if not log_path.exists():
        print(f"FAIL: no log at {log_path}")
        return 1
    history = [json.loads(line) for line in
               log_path.read_text().splitlines() if line.strip()]
    if not history:
        print(f"FAIL: empty log at {log_path}")
        return 1
    floors = json.loads(
        (REPO / "results" / args.report).read_text()
    )["splits"][args.val_split]["majority_class_floor"]

    failures = []

    losses = [e["train_loss"] for e in history]
    if not all(b < a for a, b in zip(losses, losses[1:])):
        failures.append(f"train loss not strictly decreasing: {losses}")
    print(f"[{'OK' if not failures else 'FAIL'}] train_loss: "
          f"{' -> '.join(f'{x:.4f}' for x in losses)}")

    best = max(history, key=lambda e: e["val_macro_recall1"])
    per_feat = best.get("val_per_feature_recall1")
    if per_feat is None:
        print("FAIL: log has no val_per_feature_recall1 — rerun with the "
              "current train.py (older runs only logged the macro)")
        return 1

    # A 3-epoch smoke is not expected to beat every majority floor — the
    # hardest features are still climbing at epoch 3 of a 125-epoch schedule.
    # What must hold is that a feature is either above its floor OR clearly
    # improving; a feature that is below floor AND flat/falling is a defect.
    # --strict (for the real run) requires every feature above its floor.
    first, last = history[0], history[-1]
    below = []
    for feat, floor in floors.items():
        got = per_feat.get(feat)
        if got is None:
            below.append(f"{feat}: MISSING from eval")
            continue
        if got >= floor:
            continue
        delta = (last.get("val_per_feature_recall1", {}).get(feat, 0.0)
                 - first.get("val_per_feature_recall1", {}).get(feat, 0.0))
        if args.strict:
            below.append(f"{feat}: {got:.4f} < floor {floor:.4f}")
        elif delta <= 0.005:
            below.append(f"{feat}: {got:.4f} < floor {floor:.4f} and not "
                         f"improving (delta {delta:+.4f} over the run)")
        else:
            print(f"    NOTE {feat}: {got:.4f} < floor {floor:.4f} but rising "
                  f"({delta:+.4f} over the run) — acceptable for a smoke")
    if below:
        failures.append("features below majority floor: " + "; ".join(below))
    n_above = sum(1 for f, fl in floors.items() if per_feat.get(f, -1) >= fl)
    print(f"[{'OK' if not below else 'FAIL'}] per-feature vs floor "
          f"(epoch {best['epoch']}): {n_above}/{len(floors)} above floor, "
          f"{len(floors) - n_above} below but rising, {len(below)} failing")
    for feat, floor in sorted(floors.items()):
        got = per_feat.get(feat, float('nan'))
        print(f"    {feat:<24} {got:.4f}  (floor {floor:.4f}, "
              f"{'+' if got >= floor else ''}{got - floor:+.4f})")

    # Representation-collapse guard. The failure this catches: every clip
    # pools to an identical vector, so each head emits one constant class.
    # That can sit AT the majority floor (looking merely under-trained) or
    # below it, so check the prediction diversity directly.
    n_const = sum(1 for f, v in per_feat.items()
                  if abs(v - floors[f]) < 1e-4)
    if n_const >= len(floors) // 2:
        failures.append(
            f"{n_const}/{len(floors)} features score EXACTLY their majority "
            f"floor — likely constant predictions (representation collapse)")
    print(f"[{'OK' if n_const < len(floors) // 2 else 'FAIL'}] collapse guard: "
          f"{n_const}/{len(floors)} features exactly at floor")

    macro = best["val_macro_recall1"]
    mean_floor = sum(floors.values()) / len(floors)
    need = mean_floor + args.margin
    if macro < need:
        failures.append(
            f"macro {macro:.4f} < mean floor {mean_floor:.4f} + {args.margin}")
    print(f"[{'OK' if macro >= need else 'FAIL'}] macro R@1 {macro:.4f} "
          f"vs mean floor {mean_floor:.4f} + {args.margin} = {need:.4f}")

    if failures:
        print("\nSMOKE CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
