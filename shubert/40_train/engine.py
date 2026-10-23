"""Shared eval loop (used by train.py each epoch and 50_eval/evaluate.py)."""

import torch
import torch.nn.functional as F


def autocast_dtype() -> torch.dtype:
    """bf16 only where it is NATIVE (Ampere sm_80+), else fp16.

    `torch.cuda.is_bf16_supported()` also returns True on pre-Ampere cards
    (e.g. Tesla T4, sm_75) where bf16 is emulated and painfully slow — the
    first smoke run crawled at 0.03 it/s because of exactly this.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def to_device(batch: dict, device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def run_eval(model, loader, device, feature_order):
    """Per-feature Recall@1 (= accuracy incl. NA-class rows) + macro avg.

    Label -1 means "no supervision for this feature on this clip" (PopSign
    clips whose gloss is absent from ASL-LEX, or structurally-absent values
    under the ignore policy). Such entries are excluded from BOTH the loss and
    the denominator -- scoring them as errors would understate accuracy on
    exactly the features that carry them. Sem-Lex has no -1, so its numbers are
    unchanged.
    """
    model.eval()
    n_feat = len(feature_order)
    correct = torch.zeros(n_feat, dtype=torch.long)
    valid = torch.zeros(n_feat, dtype=torch.long)
    total = 0
    loss_sum = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=autocast_dtype(),
                            enabled=device.type == "cuda"):
            logits = model(batch["face"], batch["hand1"], batch["hand2"],
                           batch["body"], batch["padding_mask"],
                           batch["lengths"])
            losses = [
                F.cross_entropy(lg.float(), batch["labels"][:, i],
                                ignore_index=-1)
                for i, lg in enumerate(logits)
            ]
        # an all-ignored feature yields NaN from cross_entropy; drop those
        finite = [l for l in losses if torch.isfinite(l)]
        if finite:
            loss_sum += (float(sum(finite).item() / len(finite))
                         * batch["labels"].shape[0])
        for i, lg in enumerate(logits):
            tgt = batch["labels"][:, i]
            m = tgt != -1
            correct[i] += ((lg.argmax(-1) == tgt) & m).sum().cpu()
            valid[i] += m.sum().cpu()
        total += batch["labels"].shape[0]
    per_feature = {
        name: round(correct[i].item() / max(1, valid[i].item()), 4)
        for i, name in enumerate(feature_order)
    }
    macro = round(sum(per_feature.values()) / n_feat, 4)
    out = {"per_feature_recall1": per_feature, "macro_recall1": macro,
           "loss": round(loss_sum / total, 4), "n": total}
    if int(valid.min()) != total:  # only when some supervision was missing
        out["n_valid_per_feature"] = {
            name: int(valid[i]) for i, name in enumerate(feature_order)}
    return out
