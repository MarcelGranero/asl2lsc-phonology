"""Evaluate a fine-tuned checkpoint: per-feature Recall@1 + macro on a split.

Writes results/metrics_{split}[_{run}].json. Run under shubert_train_env.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "40_train"))
sys.path.insert(1, str(REPO_ROOT))
from data_config import RESULTS_DIR, RUNS_DIR  # noqa: E402
from ablation import resolve_channels  # noqa: E402
from dataset import SemLexNpyDataset, SemLexPackedDataset, collate  # noqa: E402
from engine import run_eval  # noqa: E402
from model import PhonologyClassifier  # noqa: E402
from model_loading import load_backbone  # noqa: E402
from train import resolve_heads  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path,
                        default=RUNS_DIR / "main" / "checkpoints" / "best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-label", default="main")
    parser.add_argument("--no-pack", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    feature_order = state["feature_order"]
    # The channel ablation adds no parameters, so a hands-only checkpoint would
    # load cleanly into a full 4-channel model and silently be evaluated WITH
    # face+body. Recover it from the config saved alongside the weights.
    spec = (state.get("config") or {}).get("keep_channels")
    keep_channels = resolve_channels(spec) if spec else None
    if keep_channels:
        print(f"ABLATION (from ckpt config): keeping {list(keep_channels)}")
    model = PhonologyClassifier(load_backbone(), state["n_classes"],
                                keep_channels=keep_channels)
    model.load_state_dict(state["model"])
    device = torch.device("cuda")
    model.to(device)

    ds_cls = SemLexNpyDataset if args.no_pack else SemLexPackedDataset
    loader = DataLoader(ds_cls(args.split), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=collate)
    metrics = run_eval(model, loader, device, feature_order)
    # A --keep-heads checkpoint carries 10 heads that were never trained
    # (random init). Their per-feature numbers are noise, and the 16-head
    # macro would average them in — null the untrained entries and report the
    # macro over the trained heads instead.
    head_spec = (state.get("config") or {}).get("keep_heads")
    kept_heads = resolve_heads(head_spec, feature_order) if head_spec else None
    if kept_heads:
        print(f"HEAD ABLATION (from ckpt config): trained heads {list(kept_heads)}")
        per = metrics["per_feature_recall1"]
        metrics["macro_recall1"] = round(
            sum(per[h] for h in kept_heads) / len(kept_heads), 4)
        metrics["per_feature_recall1"] = {
            h: (per[h] if h in kept_heads else None) for h in feature_order}
    metrics.update({"split": args.split, "ckpt": str(args.ckpt),
                    "ckpt_epoch": state["epoch"],
                    "keep_channels": list(keep_channels) if keep_channels else "all",
                    "keep_heads": list(kept_heads) if kept_heads else "all",
                    "ckpt_val": state.get("val")})

    out = RESULTS_DIR / f"metrics_{args.split}_{args.run_label}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["per_feature_recall1"], indent=2))
    print(f"macro Recall@1 [{args.split}]: {metrics['macro_recall1']} "
          f"(n={metrics['n']}) -> {out}")


if __name__ == "__main__":
    main()
