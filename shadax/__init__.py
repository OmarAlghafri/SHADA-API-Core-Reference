"""
SHADAX - Self-supervised Hierarchical Adaptive Hybrid Algorithm.

A production-ready Python implementation of the SHADA algorithm: a four-stage
hierarchical encoder shared across the image and text modalities, four task
heads (classification, detection, segmentation, language modeling), two
self-supervised objectives and a four-phase training pipeline, all behind a
familiar sklearn-style estimator.

Public API:
    SHADA: High-level sklearn-style estimator (fit/predict/score).
    SHADAConfig: Configuration dataclass (the model-shape contract).
    create_config: Tier-aware configuration factory.
    SHADANet: The unified encoder + head + SSL ``nn.Module``.
    HierarchicalEncoder: The shared multi-modal backbone.
    ModelTier / TaskType / TrainingPhase / Modality: Controlled vocabularies.

Example:
    >>> from shadax import SHADA
    >>> model = SHADA(tier="base", num_classes=10)
    >>> model.fit(X_train, y_train)            # doctest: +SKIP
    >>> predictions = model.predict(X_test)    # doctest: +SKIP
"""

__version__ = "0.2.0"
__author__ = "Omar"

from .config import (
    Modality,
    ModelTier,
    SHADAConfig,
    TaskType,
    TrainingPhase,
    create_config,
)
from .encoder import HierarchicalEncoder
from .model import SHADA
from .network import SHADANet

__all__ = [
    "SHADA",
    "SHADAConfig",
    "create_config",
    "ModelTier",
    "TaskType",
    "TrainingPhase",
    "Modality",
    "HierarchicalEncoder",
    "SHADANet",
]
