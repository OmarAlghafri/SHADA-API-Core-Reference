"""Tests for :class:`shadax.encoder.HierarchicalEncoder`."""
from __future__ import annotations

import pytest
import torch

from shadax.config import create_config
from shadax.encoder import HierarchicalEncoder


def _nano():
    return create_config("nano", task="classification", num_classes=5, vocab_size=200)


def test_image_forward_shapes_and_hierarchy():
    torch.manual_seed(0)
    cfg = _nano()
    enc = HierarchicalEncoder(cfg)
    x = torch.randn(2, 3, 64, 64)
    out = enc(x, modality="image")

    assert out["modality"] == "image"
    assert out["hw"] == (2, 2)  # 64 / 32
    assert len(out["feature_maps"]) == 4

    # Strides 4, 8, 16, 32 and increasing channel width.
    expected_hw = [16, 8, 4, 2]
    for i, fmap in enumerate(out["feature_maps"]):
        assert fmap.shape[0] == 2
        assert fmap.shape[1] == cfg.encoder_dims[i]
        assert fmap.shape[2] == expected_hw[i]
        assert fmap.shape[3] == expected_hw[i]

    assert out["tokens"].shape == (2, 2 * 2, cfg.final_dim)
    assert out["global_features"].shape == (2, cfg.final_dim)


def test_text_forward_shapes():
    torch.manual_seed(0)
    cfg = _nano()
    enc = HierarchicalEncoder(cfg)
    ids = torch.randint(1, 199, (3, 16))
    out = enc(ids, modality="text")

    assert out["modality"] == "text"
    assert out["hw"] is None
    assert len(out["feature_maps"]) == 4
    for i, fmap in enumerate(out["feature_maps"]):
        assert fmap.shape == (3, 16, cfg.encoder_dims[i])
    assert out["tokens"].shape == (3, 16, cfg.final_dim)
    assert out["global_features"].shape == (3, cfg.final_dim)


def test_adaptive_resolution():
    torch.manual_seed(0)
    enc = HierarchicalEncoder(_nano())
    for size in (32, 64, 96):
        out = enc(torch.randn(1, 3, size, size), modality="image")
        assert out["hw"] == (size // 32, size // 32)


def test_non_divisible_resolution_raises():
    enc = HierarchicalEncoder(_nano())
    with pytest.raises(ValueError):
        enc(torch.randn(1, 3, 50, 64), modality="image")


def test_unknown_modality_raises():
    enc = HierarchicalEncoder(_nano())
    with pytest.raises(ValueError):
        enc(torch.randn(1, 3, 64, 64), modality="audio")


def test_causal_mask_blocks_future_tokens():
    """Changing a future token must not affect an earlier position's output."""
    torch.manual_seed(0)
    enc = HierarchicalEncoder(_nano()).eval()
    ids = torch.randint(1, 199, (1, 12))

    with torch.no_grad():
        out_a = enc(ids, modality="text", causal=True)["tokens"]
        modified = ids.clone()
        modified[0, -1] = (modified[0, -1] + 5) % 199  # change last token only
        out_b = enc(modified, modality="text", causal=True)["tokens"]

    # Earlier positions (all but the last) must be identical under a causal mask.
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)
    # The last position is allowed to differ.
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)


def test_forward_features_returns_only_maps():
    enc = HierarchicalEncoder(_nano())
    maps = enc.forward_features(torch.randn(1, 3, 32, 32), modality="image")
    assert isinstance(maps, list) and len(maps) == 4
