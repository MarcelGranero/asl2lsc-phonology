"""SL-GCN with parallel gloss + phonological-feature classification heads.

Paper §4.3/§4.4: "we train 16 classification heads ranging from size 2 to 58,
one for each phonological feature type ... a summed cross-entropy loss is
computed", and for ISR+PFR "the model architecture is identical ... only with
an extra classification head for gloss".

So the whole delta from upstream is the final `nn.Linear`.  This subclass
reuses upstream's `Model` — data_bn, l1–l10, the pooling — completely
untouched, and replaces `fc` with a `ModuleDict` of heads sharing the same
256-d pooled representation.

`self.fc` is deliberately KEPT (not deleted) when a gloss head is present: it
IS the gloss head, under upstream's name.  That means a phase-1 gloss
checkpoint loads into this model with `strict=True` for every shared weight,
which is exactly what the §5.2 "pre-trained to predict gloss" regime needs —
no key renaming, no partial-load guesswork.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

from model.decouple_gcn_attn import Model as UpstreamModel


class MultiheadModel(UpstreamModel):
    """Upstream SL-GCN, with `fc` replaced by parallel heads.

    Args mirror upstream's exactly, plus:
      feature_sizes: list[int] — number of classes per phonological feature,
                     in the order given by results/feature_vocab.json.
      with_gloss:    bool — keep the gloss head (`fc`).  False = PFR-only
                     (paper §5.2), True = ISR+PFR (paper §5.3).
    """

    def __init__(self, *args, feature_sizes=None, with_gloss=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not feature_sizes:
            raise ValueError("feature_sizes is required")

        self.feature_sizes = list(feature_sizes)
        self.with_gloss = with_gloss

        in_dim = self.fc.in_features  # 256, read from upstream rather than assumed
        if not with_gloss:
            # No gloss objective: drop upstream's head so its weights are not
            # carried around unused (and cannot silently receive gradients).
            self.fc = None

        self.feature_heads = nn.ModuleList(
            [nn.Linear(in_dim, n) for n in self.feature_sizes]
        )
        # Match upstream's init scheme for the gloss head (see its __init__).
        for head, n in zip(self.feature_heads, self.feature_sizes):
            nn.init.normal_(head.weight, 0, (2.0 / n) ** 0.5)
            nn.init.constant_(head.bias, 0)

    def encode(self, x, keep_prob=0.9):
        """Upstream's forward up to the pooled representation, verbatim.

        Kept in lockstep with `model.decouple_gcn_attn.Model.forward`; the only
        change is returning the pooled tensor instead of `self.fc(x)`.
        """
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)

        c_new = x.size(1)
        x = x.reshape(N, M, c_new, -1)
        return x.mean(3).mean(1)

    def forward(self, x, keep_prob=0.9):
        """Return (gloss_logits | None, [feature_logits per head])."""
        pooled = self.encode(x, keep_prob)
        gloss_logits = self.fc(pooled) if self.with_gloss else None
        feature_logits = [head(pooled) for head in self.feature_heads]
        return gloss_logits, feature_logits


def multihead_loss(gloss_logits, feature_logits, gloss_target, feature_target,
                   gloss_weight=1.0, feature_weight=1.0, ignore_index=-1):
    """Summed cross-entropy over the gloss head and every feature head.

    The paper gives no loss weighting ("a summed cross-entropy loss"), so both
    weights default to 1.0 — reproduce first, tune later.

    `ignore_index=-1` matters even under the default `--null-policy class`
    sidecar (where n/a is a real class and no -1 appears): it keeps the loss
    correct if the sidecar was built with `--null-policy ignore`, or if a
    sample was missing from it entirely and the feeder masked the row.
    """
    ce = nn.functional.cross_entropy
    losses = {}
    total = 0.0

    if gloss_logits is not None and gloss_target is not None:
        gloss_loss = ce(gloss_logits, gloss_target)
        losses["gloss"] = gloss_loss
        total = total + gloss_weight * gloss_loss

    feature_total = 0.0
    for i, logits in enumerate(feature_logits):
        target = feature_target[:, i]
        if (target != ignore_index).sum() == 0:
            continue  # whole batch masked for this feature — skip, no NaN
        loss_i = ce(logits, target, ignore_index=ignore_index)
        losses[f"feature_{i}"] = loss_i
        feature_total = feature_total + loss_i

    total = total + feature_weight * feature_total
    losses["total"] = total
    return total, losses
