"""Tests for the sklearn-style :class:`shadax.model.SHADA` estimator."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from shadax import SHADA, SHADAConfig, create_config
from shadax.config import TaskType, TrainingPhase


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# --------------------------------------------------------------------------- #
# Construction.
# --------------------------------------------------------------------------- #
def test_init_accepts_config_object():
    cfg = create_config("nano", task="classification", num_classes=7)
    model = SHADA(cfg)
    assert isinstance(model.config, SHADAConfig)
    assert model.config.num_classes == 7
    assert model.num_classes == 7


def test_init_from_strings():
    model = SHADA("nano", task="segmentation", num_classes=4)
    assert model.config.tier == "nano"
    assert model.config.task_type is TaskType.SEGMENTATION
    assert not model.is_fitted


def test_repr_is_informative():
    model = SHADA("nano", task="lm", vocab_size=200)
    assert "tier='nano'" in repr(model)
    assert "task='lm'" in repr(model)


# --------------------------------------------------------------------------- #
# Classification.
# --------------------------------------------------------------------------- #
def test_classification_fit_predict_proba_score_and_epochs_kwarg():
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.random.randint(0, 5, 8)
    model = SHADA("nano", task="classification", num_classes=5, batch_size=4, epochs=99)
    # The epochs kwarg must be accepted (previously raised TypeError).
    model.fit(X, y, epochs=2, verbose=False)
    assert model.is_fitted

    preds = model.predict(X)
    assert preds.shape == (8,)
    preds2, probs = model.predict(X, return_probs=True)
    assert probs.shape == (8, 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert model.predict_proba(X).shape == (8, 5)
    assert 0.0 <= model.score(X, y) <= 1.0


def test_classification_accepts_one_hot_labels():
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.eye(5)[np.random.randint(0, 5, 8)]
    model = SHADA("nano", task="classification", num_classes=5, batch_size=4, epochs=1)
    model.fit(X, y, verbose=False)
    assert model.predict(X).shape == (8,)


def test_channels_last_input_is_handled():
    X = np.random.randn(6, 64, 64, 3).astype("float32")  # NHWC
    y = np.random.randint(0, 5, 6)
    model = SHADA("nano", task="classification", num_classes=5, batch_size=3, epochs=1)
    model.fit(X, y, verbose=False)
    assert model.predict(X).shape == (6,)


# --------------------------------------------------------------------------- #
# Segmentation.
# --------------------------------------------------------------------------- #
def test_segmentation_fit_predict_score():
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.random.randint(0, 4, (8, 64, 64))
    model = SHADA("nano", task="segmentation", num_classes=4, batch_size=4, epochs=1)
    model.fit(X, y, phases=[TrainingPhase.FINETUNE], verbose=False)
    assert model.predict(X).shape == (8, 64, 64)
    assert model.predict_proba(X).shape == (8, 4, 64, 64)
    assert 0.0 <= model.score(X, y) <= 1.0


# --------------------------------------------------------------------------- #
# Language model.
# --------------------------------------------------------------------------- #
def test_lm_pretrain_and_finetune_predict_score():
    X = np.random.randint(1, 199, (8, 16))
    model = SHADA("nano", task="lm", vocab_size=200, batch_size=4, epochs=1)
    # y=None -> default PRETRAIN.
    model.fit(X, verbose=False)
    assert model.is_fitted
    # Supervised next-token finetune.
    model.fit(X, y=X, phases=[TrainingPhase.FINETUNE], verbose=False)
    assert model.predict(X).shape == (8, 16)
    assert model.predict_proba(X).shape == (8, 16, 200)
    assert 0.0 <= model.score(X, X) <= 1.0


# --------------------------------------------------------------------------- #
# Detection.
# --------------------------------------------------------------------------- #
def _det_targets(n=8, c=3, hs=2, ws=2):
    return {
        "heatmap": np.random.rand(n, c, hs, ws).astype("float32"),
        "wh": np.random.rand(n, 2, hs, ws).astype("float32"),
        "offset": np.random.rand(n, 2, hs, ws).astype("float32"),
        "reg_mask": (np.random.rand(n, 1, hs, ws) > 0.5).astype("float32"),
    }


def test_detection_fit_predict_and_score_raises():
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = _det_targets()
    model = SHADA("nano", task="detection", num_classes=3, batch_size=4, epochs=1)
    model.fit(X, y, phases=[TrainingPhase.FINETUNE], verbose=False)
    det = model.predict(X)
    assert isinstance(det, list) and len(det) == 8
    for d in det:
        assert set(d) == {"boxes", "scores", "labels"}
        assert d["boxes"].shape[1] == 4
        assert d["boxes"].shape[0] == d["scores"].shape[0] == d["labels"].shape[0]
    with pytest.raises(NotImplementedError):
        model.score(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


# --------------------------------------------------------------------------- #
# Feature extraction + persistence.
# --------------------------------------------------------------------------- #
def test_extract_features_all_layers():
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.random.randint(0, 5, 8)
    model = SHADA("nano", task="classification", num_classes=5, batch_size=4, epochs=1)
    model.fit(X, y, verbose=False)
    fd = model.config.final_dim
    g = model.extract_features(X, "global")
    t = model.extract_features(X, "tokens")
    s = model.extract_features(X, "spatial")
    assert g.shape == (8, fd)
    assert t.shape == (8, 4, fd)
    assert s.shape == (8, fd, 2, 2)
    with pytest.raises(ValueError):
        model.extract_features(X, "bogus")


def test_extract_features_before_fit_raises():
    model = SHADA("nano", task="classification", num_classes=5)
    with pytest.raises(RuntimeError):
        model.extract_features(np.random.randn(2, 3, 64, 64).astype("float32"))


def test_predict_before_fit_raises():
    model = SHADA("nano", task="classification", num_classes=5)
    with pytest.raises(RuntimeError):
        model.predict(np.random.randn(2, 3, 64, 64).astype("float32"))


def test_save_load_round_trip_predictions_match():
    X = np.random.randn(8, 3, 64, 64).astype("float32")
    y = np.random.randint(0, 5, 8)
    model = SHADA("nano", task="classification", num_classes=5, batch_size=4, epochs=2)
    model.fit(X, y, verbose=False)
    before = model.predict(X)

    path = os.path.join(tempfile.gettempdir(), "shada_test_ckpt.pt")
    try:
        model.save(path)
        reloaded = SHADA("nano", task="classification", num_classes=5)
        reloaded.load(path)
        after = reloaded.predict(X)
        assert reloaded.is_fitted
        assert np.array_equal(before, after)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_save_before_fit_raises():
    model = SHADA("nano", task="classification", num_classes=5)
    with pytest.raises(RuntimeError):
        model.save(os.path.join(tempfile.gettempdir(), "nope.pt"))
