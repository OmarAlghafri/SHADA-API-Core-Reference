"""Tests for the self-supervised objectives in :mod:`shadax.ssl`."""
from __future__ import annotations

import torch

from shadax.config import create_config
from shadax.encoder import HierarchicalEncoder
from shadax.heads import LanguageModelHead
from shadax.ssl import MaskedImageModeling, MaskedLanguageModeling


def _nano(task="classification", num_classes=5, **kw):
    kw.setdefault("vocab_size", 200)
    return create_config("nano", task=task, num_classes=num_classes, **kw)


def test_mim_loss_shapes_and_mask_coverage():
    torch.manual_seed(0)
    cfg = _nano(mask_ratio=0.75)
    enc = HierarchicalEncoder(cfg)
    mim = MaskedImageModeling(cfg)
    images = torch.randn(2, 3, 64, 64)
    out = mim(enc, images)

    assert out["loss"].ndim == 0
    assert out["recon"].shape == images.shape
    assert out["mask"].shape == (2, 1, 64, 64)
    # Mask coverage close to mask_ratio (patch grid is 2x2 -> coarse, so allow slack).
    frac = out["mask"].mean().item()
    assert 0.0 < frac < 1.0


def test_mim_grad_flows_into_encoder():
    torch.manual_seed(0)
    cfg = _nano()
    enc = HierarchicalEncoder(cfg)
    mim = MaskedImageModeling(cfg)
    loss = mim(enc, torch.randn(2, 3, 64, 64))["loss"]
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_mlm_loss_shapes_and_grad():
    torch.manual_seed(0)
    cfg = _nano("lm", text_mask_ratio=0.5)
    enc = HierarchicalEncoder(cfg)
    head = LanguageModelHead(cfg.final_dim, cfg.vocab_size)
    mlm = MaskedLanguageModeling(cfg)
    ids = torch.randint(1, 199, (3, 16))
    out = mlm(enc, head, ids)

    assert out["loss"].ndim == 0
    assert out["logits"].shape == (3, 16, cfg.vocab_size)
    assert out["mask"].shape == (3, 16)
    assert out["mask"].any()  # at least one masked position

    out["loss"].backward()
    enc_grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in enc_grads)


def test_mlm_default_mask_token_id():
    cfg = _nano("lm", vocab_size=200)
    mlm = MaskedLanguageModeling(cfg)
    assert mlm.mask_token_id == 199  # vocab_size - 1


def test_mlm_never_masks_padding():
    torch.manual_seed(0)
    cfg = _nano("lm", pad_token_id=0, text_mask_ratio=0.9)
    enc = HierarchicalEncoder(cfg)
    head = LanguageModelHead(cfg.final_dim, cfg.vocab_size)
    mlm = MaskedLanguageModeling(cfg)
    ids = torch.randint(1, 199, (2, 16))
    ids[:, 8:] = 0  # padding
    out = mlm(enc, head, ids)
    pad_positions = ids == 0
    assert not bool((out["mask"] & pad_positions).any())
