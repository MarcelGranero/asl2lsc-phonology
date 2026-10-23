"""Input-modality ablation: keep only a subset of SHuBERT's 4 channels.

Motivation
----------
Ablation asked of the phonological heads: do the handshape-family features
(Thumb Contact, Selected Fingers, Thumb Position, Flexion, Spread, Handshape)
actually need the face and body streams, or do the two hand streams carry all
the signal? Answering it requires removing face+body from the INPUT while
changing nothing else about the Table-9 setup.

Why mask_emb substitution (and not zeros, and not deleting the slots)
---------------------------------------------------------------------
The backbone fuses its 4 channels positionally: each stream gets its own
LayerNorm + Linear -> intermediate_dim//4 (=256), the four are stacked and
flattened to intermediate_dim, then post_extract_proj maps that to the encoder
width. The slots are fixed-width and order-dependent, so a channel cannot be
"removed" without either changing the architecture or supplying *something* in
its slot.

SHuBERT pretraining already defines what "this channel is absent" looks like.
`apply_mask` computes

    x = x * mask + self.mask_emb * (1 - mask)

and with `mask_strategy="channel"` it masks entire channels across all frames
-- exactly the operation this ablation needs. So the encoder has seen
mask_emb-in-a-channel-slot as in-distribution input during pretraining.
We reuse that same learned parameter.

The alternatives are worse:
  * zeros in the raw stream -> LayerNorm maps an all-zero vector to a nonzero
    arbitrary constant, an input the encoder never saw. That would understate
    hands-only performance and confound "face/body unnecessary" with
    "face/body slot fed garbage".
  * deleting the projections and shrinking post_extract_proj -> discards
    pretrained weights and breaks the 226/226 checkpoint load, confounding
    modality removal with re-initialisation.

Implementation
--------------
Forward hooks on the per-channel projection Linears, so `external/` stays
verbatim (repo rule: upstream runs AS-IS). A hook on e.g. `face_proj` replaces
its (B,T,256) output with mask_emb broadcast to the same shape -- bit-identical
to what apply_mask would have written into that channel slot.

Two consequences are deliberate:
  * the ablated stream's LayerNorm/Linear still run, but their output is
    discarded, so they receive no gradient and cannot leak information;
  * the ablated features are still READ from disk. Wasteful, but it keeps the
    dataloader, manifests and packed blobs byte-identical to the baseline, so
    the two runs differ in exactly one variable.
"""

import torch

# Channel name -> the projection module whose output fills that channel slot.
CHANNEL_PROJ = {
    "face": "face_proj",
    "left_hand": "left_hand_proj",
    "right_hand": "right_hand_proj",
    "body": "body_posture_proj",
}
ALL_CHANNELS = tuple(CHANNEL_PROJ)

# Named presets. "hands" is the ablation this module was written for.
PRESETS = {
    "all": ALL_CHANNELS,
    "hands": ("left_hand", "right_hand"),
    "hands_body": ("left_hand", "right_hand", "body"),
    "face": ("face",),
    "body": ("body",),
}


def resolve_channels(spec: str) -> tuple[str, ...]:
    """'hands' | 'all' | ... | a comma list like 'left_hand,right_hand'."""
    if spec in PRESETS:
        return PRESETS[spec]
    chans = tuple(c.strip() for c in spec.split(",") if c.strip())
    bad = [c for c in chans if c not in CHANNEL_PROJ]
    assert not bad, (
        f"unknown channel(s) {bad}; expected a preset {sorted(PRESETS)} "
        f"or a comma list of {sorted(CHANNEL_PROJ)}"
    )
    assert chans, "no channels kept -- the model would have no input"
    return chans


def apply_channel_ablation(backbone, keep: tuple[str, ...]) -> list:
    """Replace every non-kept channel slot with the pretrained mask_emb.

    Returns the hook handles (kept so a caller could undo it; the trainer
    never does). Asserts the model still has 4 channel slots, so this fails
    loudly rather than silently no-op'ing if upstream is ever re-vendored.
    """
    keep = tuple(keep)
    drop = [c for c in ALL_CHANNELS if c not in keep]
    for name in keep:
        assert name in CHANNEL_PROJ, f"unknown channel {name}"
    assert drop, "ablation asked to keep all 4 channels -- nothing to do"

    slot_dim = backbone.mask_emb.shape[-1]
    for proj_name in CHANNEL_PROJ.values():
        proj = getattr(backbone, proj_name)
        assert proj.out_features == slot_dim, (
            f"{proj_name} out_features {proj.out_features} != mask_emb width "
            f"{slot_dim} -- channel layout changed, ablation is unsafe"
        )

    handles = []
    for name in drop:
        module = getattr(backbone, CHANNEL_PROJ[name])

        def hook(mod, inputs, output, _bb=backbone):
            # mask_emb is (1,1,1,D); the slot here is (B,T,D). Expand, and
            # match dtype/device so autocast sees no surprise.
            emb = _bb.mask_emb[0, 0, 0].to(output.dtype)
            return emb.expand_as(output)

        handles.append(module.register_forward_hook(hook))
    return handles


def _self_test() -> None:
    """Verify the hook does what the docstring claims, on the real model."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_loading import load_backbone, make_source

    torch.manual_seed(0)
    model = load_backbone().eval()
    B, T = 2, 11
    face = torch.randn(B, T, 384)
    h1 = torch.randn(B, T, 384)
    h2 = torch.randn(B, T, 384)
    body = torch.randn(B, T, 14)
    pm = torch.zeros(B, T, dtype=torch.bool)

    def run():
        with torch.no_grad():
            return model(make_source(face, h1, h2, body), padding_mask=pm,
                         mask=False, features_only=True)["x"]

    base = run()
    apply_channel_ablation(model.backbone if hasattr(model, "backbone") else model,
                           PRESETS["hands"])
    abl = run()

    # 1. the ablation must actually change the representation
    delta = (base - abl).abs().max().item()
    print(f"max |baseline - hands_only| = {delta:.4f}")
    assert delta > 1e-3, "ablation had no effect -- hooks not firing"

    # 2. changing a DROPPED stream must now be a no-op ...
    face = torch.randn(B, T, 384)
    body = torch.randn(B, T, 14)
    abl2 = run()
    d_dropped = (abl - abl2).abs().max().item()
    print(f"max change when face+body are resampled = {d_dropped:.2e}")
    assert d_dropped < 1e-5, "dropped channels still leak into the output"

    # 3. ... while changing a KEPT stream must still matter.
    h1 = torch.randn(B, T, 384)
    abl3 = run()
    d_kept = (abl2 - abl3).abs().max().item()
    print(f"max change when left_hand is resampled = {d_kept:.4f}")
    assert d_kept > 1e-3, "kept channel no longer influences the output"

    print("ABLATION SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
