"""Tests for :mod:`shadax.config` -- tiers, validation, the factory."""
from __future__ import annotations

import pytest

from shadax.config import (
    NUM_STAGES,
    TOTAL_REDUCTION,
    ModelTier,
    SHADAConfig,
    TaskType,
    create_config,
)


def test_total_reduction_is_32():
    assert TOTAL_REDUCTION == 32
    assert NUM_STAGES == 4


@pytest.mark.parametrize("tier", ["nano", "base", "large", "xl"])
def test_each_tier_builds_and_is_consistent(tier):
    cfg = create_config(tier, task="classification", num_classes=10)
    assert cfg.tier == tier
    assert len(cfg.encoder_dims) == NUM_STAGES
    assert len(cfg.encoder_depths) == NUM_STAGES
    assert len(cfg.num_heads) == NUM_STAGES
    assert cfg.embed_dim == cfg.encoder_dims[0]
    assert cfg.final_dim == cfg.encoder_dims[-1]
    assert cfg.num_stages == NUM_STAGES


def test_create_config_accepts_enum_tier_and_task():
    cfg = create_config(ModelTier.NANO, task=TaskType.SEGMENTATION, num_classes=4)
    assert cfg.tier == "nano"
    assert cfg.task == "segmentation"
    assert cfg.task_type is TaskType.SEGMENTATION


def test_overrides_applied():
    cfg = create_config("nano", task="lm", num_classes=1, vocab_size=123, dropout=0.0)
    assert cfg.vocab_size == 123
    assert cfg.dropout == 0.0


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        create_config("gigantic", task="classification", num_classes=2)


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        SHADAConfig(task="not-a-task")


def test_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError):
        SHADAConfig(encoder_dims=[10, 256, 512, 1024])  # 10 % 4 != 0


def test_wrong_length_lists_raise():
    with pytest.raises(ValueError):
        SHADAConfig(encoder_dims=[64, 128, 256])  # only 3 stages


def test_bad_mask_ratio_raises():
    with pytest.raises(ValueError):
        SHADAConfig(mask_ratio=1.0)
    with pytest.raises(ValueError):
        SHADAConfig(text_mask_ratio=-0.1)


def test_bad_num_classes_raises():
    with pytest.raises(ValueError):
        create_config("nano", num_classes=0)


def test_stage_reduction_progression():
    cfg = create_config("nano")
    assert cfg.stage_reduction(0) == 4
    assert cfg.stage_reduction(1) == 8
    assert cfg.stage_reduction(2) == 16
    assert cfg.stage_reduction(3) == 32
