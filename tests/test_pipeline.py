"""Tests for losses and the four-phase pipeline in :mod:`shadax.training`."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from shadax.config import create_config, TrainingPhase
from shadax.network import SHADANet
from shadax.training import compute_task_loss, run_pipeline


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def _cls_net():
    return SHADANet(create_config("nano", task="classification", num_classes=5))


# --------------------------------------------------------------------------- #
# Task losses.
# --------------------------------------------------------------------------- #
def test_classification_loss_scalar_and_one_hot():
    cfg = create_config("nano", task="classification", num_classes=5)
    logits = torch.randn(4, 5, requires_grad=True)
    y = torch.randint(0, 5, (4,))
    loss = compute_task_loss(cfg, {"logits": logits}, None, y)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None

    one_hot = torch.eye(5)[y]
    loss_oh = compute_task_loss(cfg, {"logits": logits.detach()}, None, one_hot)
    assert loss_oh.ndim == 0


def test_lm_loss_uses_shifted_input():
    cfg = create_config("nano", task="lm", vocab_size=200)
    logits = torch.randn(2, 16, 200, requires_grad=True)
    x = torch.randint(1, 199, (2, 16))
    loss = compute_task_loss(cfg, {"logits": logits}, x, None)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_segmentation_loss():
    cfg = create_config("nano", task="segmentation", num_classes=4)
    seg = torch.randn(2, 4, 64, 64, requires_grad=True)
    y = torch.randint(0, 4, (2, 64, 64))
    loss = compute_task_loss(cfg, {"segmentation": seg}, None, y)
    assert loss.ndim == 0
    loss.backward()
    assert seg.grad is not None


def test_detection_loss():
    cfg = create_config("nano", task="detection", num_classes=3)
    outputs = {
        "heatmap": torch.randn(2, 3, 2, 2, requires_grad=True),
        "wh": torch.randn(2, 2, 2, 2, requires_grad=True),
        "offset": torch.randn(2, 2, 2, 2, requires_grad=True),
    }
    y = {
        "heatmap": torch.rand(2, 3, 2, 2),
        "wh": torch.rand(2, 2, 2, 2),
        "offset": torch.rand(2, 2, 2, 2),
        "reg_mask": (torch.rand(2, 1, 2, 2) > 0.5).float(),
    }
    loss = compute_task_loss(cfg, outputs, None, y)
    assert loss.ndim == 0
    loss.backward()
    assert outputs["heatmap"].grad is not None


# --------------------------------------------------------------------------- #
# Pipeline phases.
# --------------------------------------------------------------------------- #
def test_pretrain_only_runs_without_labels():
    net = _cls_net()
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    history = run_pipeline(
        net, X, None,
        phases=[TrainingPhase.PRETRAIN],
        epochs=2, lr=1e-4, weight_decay=0.0, batch_size=4,
        device="cpu", verbose=False,
    )
    assert history["phases"] == ["pretrain"]
    assert len(history["losses"][0]) == 2


def test_finetune_requires_labels():
    net = _cls_net()
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    with pytest.raises(ValueError):
        run_pipeline(
            net, X, None,
            phases=[TrainingPhase.FINETUNE],
            epochs=1, lr=1e-4, weight_decay=0.0, batch_size=4,
            device="cpu", verbose=False,
        )


def test_lm_finetune_runs_without_labels():
    # The language-model task derives next-token targets from X itself, so a
    # supervised FINETUNE phase must run even when y is None.
    net = SHADANet(create_config("nano", task="lm", vocab_size=200))
    X = np.random.randint(1, 199, size=(8, 16))
    history = run_pipeline(
        net, X, None,
        phases=[TrainingPhase.FINETUNE],
        epochs=2, lr=1e-4, weight_decay=0.0, batch_size=4,
        device="cpu", verbose=False,
    )
    assert history["phases"] == ["finetune"]
    assert len(history["losses"][0]) == 2
    assert all(np.isfinite(history["losses"][0]))


def test_multitask_combines_losses():
    net = _cls_net()
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.random.randint(0, 5, 8)
    history = run_pipeline(
        net, X, y,
        phases=[TrainingPhase.MULTITASK],
        epochs=2, lr=1e-4, weight_decay=0.0, batch_size=4,
        device="cpu", ssl_weight=0.5, verbose=False,
    )
    assert history["phases"] == ["multitask"]
    assert all(np.isfinite(history["losses"][0]))


def test_full_four_phase_run():
    net = _cls_net()
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.random.randint(0, 5, 8)
    history = run_pipeline(
        net, X, y,
        phases=[
            TrainingPhase.PRETRAIN,
            TrainingPhase.MULTITASK,
            TrainingPhase.FINETUNE,
            TrainingPhase.DEPLOY,
        ],
        epochs=1, lr=1e-4, weight_decay=0.0, batch_size=4,
        device="cpu", verbose=False,
    )
    # DEPLOY performs no optimisation, so only 3 phases are recorded.
    assert history["phases"] == ["pretrain", "multitask", "finetune"]


def test_deploy_sets_eval_and_no_grad_state():
    net = _cls_net()
    net.train()
    X = np.random.randn(4, 3, 64, 64).astype("float32")
    run_pipeline(
        net, X, None,
        phases=[TrainingPhase.DEPLOY],
        epochs=1, lr=1e-4, weight_decay=0.0, batch_size=4,
        device="cpu", verbose=False,
    )
    assert not net.training  # DEPLOY switched to eval mode


def test_pipeline_accepts_tensor_inputs():
    net = _cls_net()
    X = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 5, (8,))
    history = run_pipeline(
        net, X, y,
        phases=[TrainingPhase.FINETUNE],
        epochs=1, lr=1e-4, weight_decay=0.0, batch_size=4,
        device="cpu", verbose=False,
    )
    assert len(history["losses"][0]) == 1
