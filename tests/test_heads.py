"""Tests for the four task heads in :mod:`shadax.heads`."""
from __future__ import annotations

import torch

from shadax.config import create_config
from shadax.heads import (
    ClassificationHead,
    DetectionHead,
    LanguageModelHead,
    SegmentationHead,
    build_head,
)
from shadax.encoder import HierarchicalEncoder


def _nano(task="classification", num_classes=5, **kw):
    kw.setdefault("vocab_size", 200)
    return create_config("nano", task=task, num_classes=num_classes, **kw)


def test_classification_head_shape_and_grad():
    head = ClassificationHead(512, 5, dropout=0.0)
    feats = torch.randn(4, 512, requires_grad=True)
    out = head(feats)
    assert out.shape == (4, 5)
    out.sum().backward()
    assert feats.grad is not None


def test_language_model_head_shape_and_grad():
    head = LanguageModelHead(512, 200)
    tokens = torch.randn(2, 16, 512, requires_grad=True)
    out = head(tokens)
    assert out.shape == (2, 16, 200)
    out.sum().backward()
    assert tokens.grad is not None


def test_segmentation_head_shape_and_grad():
    cfg = _nano("segmentation", num_classes=4)
    head = SegmentationHead(cfg.encoder_dims, 4)
    maps = [
        torch.randn(2, cfg.encoder_dims[0], 16, 16, requires_grad=True),
        torch.randn(2, cfg.encoder_dims[1], 8, 8),
        torch.randn(2, cfg.encoder_dims[2], 4, 4),
        torch.randn(2, cfg.encoder_dims[3], 2, 2),
    ]
    out = head(maps, output_size=(64, 64))
    assert out.shape == (2, 4, 64, 64)
    out.sum().backward()
    assert maps[0].grad is not None

    # Default output size = stride-4 map size x4.
    out_default = head(maps)
    assert out_default.shape == (2, 4, 64, 64)


def test_detection_head_shapes_and_grad():
    cfg = _nano("detection", num_classes=3)
    head = DetectionHead(cfg.final_dim, 3)
    maps = [
        torch.randn(2, cfg.encoder_dims[0], 16, 16),
        torch.randn(2, cfg.encoder_dims[1], 8, 8),
        torch.randn(2, cfg.encoder_dims[2], 4, 4),
        torch.randn(2, cfg.final_dim, 2, 2, requires_grad=True),
    ]
    out = head(maps)
    assert out["heatmap"].shape == (2, 3, 2, 2)
    assert out["wh"].shape == (2, 2, 2, 2)
    assert out["offset"].shape == (2, 2, 2, 2)
    (out["heatmap"].sum() + out["wh"].sum() + out["offset"].sum()).backward()
    assert maps[-1].grad is not None


def test_build_head_selects_correct_type():
    assert isinstance(build_head(_nano("classification")), ClassificationHead)
    assert isinstance(build_head(_nano("lm")), LanguageModelHead)
    assert isinstance(build_head(_nano("segmentation", num_classes=4)), SegmentationHead)
    assert isinstance(build_head(_nano("detection", num_classes=3)), DetectionHead)


def test_heads_consume_real_encoder_output():
    """Heads must accept the exact encoder output contract."""
    torch.manual_seed(0)
    cfg = _nano("segmentation", num_classes=4)
    enc = HierarchicalEncoder(cfg)
    enc_out = enc(torch.randn(2, 3, 64, 64), modality="image")
    seg_head = SegmentationHead(cfg.encoder_dims, 4)
    seg = seg_head(enc_out["feature_maps"], output_size=(64, 64))
    assert seg.shape == (2, 4, 64, 64)
