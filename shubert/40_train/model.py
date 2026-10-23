"""PhonologyClassifier: SHuBERT backbone (fully trainable) -> padding-aware
time average -> BatchNorm1d(768) -> 16 linear heads.

Paper setup: "average SHuBERT representations across the time dimension and
add a batch-norm layer followed by a linear layer as the classification
head", 16 heads trained simultaneously, full fine-tuning.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import apply_channel_ablation  # noqa: E402
from model_loading import make_source  # noqa: E402


class PhonologyClassifier(nn.Module):
    def __init__(self, backbone, n_classes: list[int], keep_channels=None):
        """keep_channels: input-modality ablation. None (default) = the paper's
        full 4-channel input. Otherwise a tuple of channel names to keep; the
        rest have their channel slot filled with the pretrained mask_emb (see
        ablation.py). Changes no parameter, so the checkpoint still loads
        226/226 either way."""
        super().__init__()
        self.backbone = backbone
        self.keep_channels = keep_channels
        if keep_channels is not None:
            apply_channel_ablation(backbone, keep_channels)
        dim = backbone.cfg.encoder_embed_dim
        self.bn = nn.BatchNorm1d(dim)
        self.heads = nn.ModuleList(nn.Linear(dim, c) for c in n_classes)

    def forward(self, face, hand1, hand2, body, padding_mask, lengths):
        out = self.backbone(
            make_source(face, hand1, hand2, body),
            padding_mask=padding_mask,
            mask=False,
            features_only=True,
        )
        x = out["x"]  # (B, T, D) last encoder layer
        # encoder may have internally padded T to a multiple; padded frames
        # are excluded via the mask either way
        T = padding_mask.shape[1]
        x = x[:, :T]
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / lengths.unsqueeze(-1).to(x.dtype)
        pooled = self.bn(pooled)
        return [head(pooled) for head in self.heads]
