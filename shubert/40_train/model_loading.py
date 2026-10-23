"""Construct the pretraining SHuBERT backbone and load the released
checkpoint, assert-verified.

Import trap this module exists to avoid: in shubert_train_env,
`import examples` resolves through a site-wide fairseq .pth path hook to a rewritten
classification shubert.py. We import the vendored PRETRAINING model
(external/shubert/shubert_model.py) directly instead; its own imports are
fairseq *core* only, which resolve fine in that env.

Hardening over upstream's shubert_inference.py loading pattern:
  1. the checkpoint's cfg.model is overlaid onto SHubertConfig() field by
     field, so architecture AND pretrain-time regularisation (dropout 0.1,
     layerdrop 0.05) are reproduced exactly rather than trusted to defaults;
  2. load_state_dict(strict=False) is followed by a hard assert that
     missing == unexpected == [] (226/226 tensors).

--self-test additionally sha256-checks the checkpoint and verifies that a
clip run alone equals the same clip inside a padded batch (padding_mask +
pooling correctness).
"""

import hashlib
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "shubert"))
sys.path.insert(0, str(REPO_ROOT))

from shubert_model import SHubertConfig, SHubertModel  # noqa: E402
from data_config import SHUBERT_CHECKPOINT, SHUBERT_CHECKPOINT_SHA256  # noqa: E402

# forward() logs stream shapes at INFO on every call — silence, don't edit.
logging.getLogger("shubert_model").setLevel(logging.WARNING)

# Fields whose checkpoint value must not be copied verbatim:
#   max_update is an omegaconf II("optimization.max_update") interpolation.
_SKIP_FIELDS = {"max_update"}


def load_backbone(checkpoint_path: Path = SHUBERT_CHECKPOINT,
                  checkpoint_activations: bool = False,
                  random_init: bool = False) -> SHubertModel:
    """Load the pretrained backbone.

    random_init=True builds the SAME architecture and skips the weight load:
    the checkpoint is still read for its cfg.model, so layer count, width,
    dropout and layerdrop are identical and the ONLY difference from the
    pretrained arm is the parameter values. That is what makes the
    pretraining ablation a one-variable comparison; constructing a default
    SHubertConfig() instead would silently change the architecture too.

    checkpoint_activations recomputes encoder activations during backward
    instead of storing them: same math, ~30% slower, but the worst-case
    128-clip batch drops from 13.66 GB to 3.84 GB peak — which is what lets
    the paper's true batch size fit the T4s in the `high` partition. It must
    be set on the config BEFORE constructing the model (the encoder wraps its
    layers at build time); it changes no parameter, so the checkpoint still
    loads 226/226.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    mcfg = ckpt["cfg"]["model"]

    cfg = SHubertConfig()
    overlaid = []
    for name in cfg.__dataclass_fields__:
        if name in _SKIP_FIELDS:
            continue
        if name in mcfg:
            setattr(cfg, name, mcfg[name])
            overlaid.append(name)
    assert "encoder_layers" in overlaid and "conv_pos" in overlaid, (
        "checkpoint cfg.model missing core fields — wrong checkpoint format?"
    )

    # AFTER the overlay — the checkpoint's cfg has its own (pretrain-time)
    # value for this and would otherwise clobber the caller's choice.
    cfg.checkpoint_activations = checkpoint_activations

    model = SHubertModel(cfg)
    if random_init:
        # Sanity-check the shapes we are NOT loading, so a silently divergent
        # architecture cannot masquerade as "pretraining didn't help".
        shapes = {k: tuple(v.shape) for k, v in ckpt["model"].items()}
        own = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        mismatch = [k for k in shapes if own.get(k) != shapes[k]]
        assert not mismatch, f"random-init arch differs from checkpoint: {mismatch[:5]}"
        print(f"RANDOM INIT: architecture from {checkpoint_path.name} "
              f"({cfg.encoder_layers}L/{cfg.encoder_embed_dim}d, "
              f"dropout={cfg.dropout}, layerdrop={cfg.encoder_layerdrop}), "
              f"{len(shapes)} tensors left at random init")
        return model
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert missing == [] and unexpected == [], (
        f"checkpoint/model mismatch: missing={missing} unexpected={unexpected}"
    )
    n = len(ckpt["model"])
    print(f"loaded {n}/{n} tensors from {checkpoint_path.name} "
          f"({cfg.encoder_layers}L/{cfg.encoder_embed_dim}d, "
          f"dropout={cfg.dropout}, layerdrop={cfg.encoder_layerdrop})")
    return model


def make_source(face, hand1, hand2, body):
    """The forward contract: list of per-sample dicts (equal, padded T) with
    mandatory dummy label_* tensors that forward() reads unconditionally."""
    B, T = face.shape[0], face.shape[1]
    z = face.new_zeros(T, 1)
    return [
        {
            "face": face[i],
            "left_hand": hand1[i],
            "right_hand": hand2[i],
            "body_posture": body[i],
            "label_face": z,
            "label_left_hand": z,
            "label_right_hand": z,
            "label_body_posture": z,
        }
        for i in range(B)
    ]


def _self_test() -> None:
    digest = hashlib.sha256()
    with SHUBERT_CHECKPOINT.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == SHUBERT_CHECKPOINT_SHA256, "checkpoint sha256 mismatch"
    print("checkpoint sha256 OK")

    model = load_backbone().eval()
    torch.manual_seed(0)
    T1, T2 = 37, 61
    clips = [
        (torch.randn(t, 384), torch.randn(t, 384), torch.randn(t, 384),
         torch.randn(t, 14))
        for t in (T1, T2)
    ]

    def run(face, h1, h2, body, pad_mask):
        with torch.no_grad():
            out = model(make_source(face, h1, h2, body),
                        padding_mask=pad_mask, mask=False, features_only=True)
        return out["x"]

    # clip 0 alone (no padding)
    f, h1, h2, b = (t.unsqueeze(0) for t in clips[0])
    solo = run(f, h1, h2, b, torch.zeros(1, T1, dtype=torch.bool))

    # clip 0 inside a padded batch with clip 1
    def pad(t):
        return torch.nn.functional.pad(t, (0, 0, 0, T2 - t.shape[0]))
    face = torch.stack([pad(clips[0][0]), clips[1][0]])
    h1b = torch.stack([pad(clips[0][1]), clips[1][1]])
    h2b = torch.stack([pad(clips[0][2]), clips[1][2]])
    bb = torch.stack([pad(clips[0][3]), clips[1][3]])
    pm = torch.zeros(2, T2, dtype=torch.bool)
    pm[0, T1:] = True
    batched = run(face, h1b, h2b, bb, pm)

    diff = (solo[0] - batched[0, :T1]).abs().max().item()
    print(f"solo-vs-batched max abs diff over real frames: {diff:.2e}")
    assert diff < 1e-4, "padding is leaking into real-frame representations"
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
