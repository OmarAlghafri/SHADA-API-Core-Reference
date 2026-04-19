"""
SHADAX - Self-supervised Hierarchical Adaptive Hybrid Algorithm

A production-ready Python implementation of the SHADA algorithm
for deep learning tasks.

API:
    SHADA: Main model class
    SHADAConfig: Configuration dataclass
    create_config: Factory function
    ModelTier: Model size enums
    TaskType: Task type enums
    TrainingPhase: Training phase enums

Example:
    >>> from shadax import SHADA
    >>> model = SHADA(tier="base", num_classes=10)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

__version__ = "0.1.0"
__author__ = "Omar"

from .core import (
    SHADA,
    SHADAConfig,
    create_config,
    ModelTier,
    TaskType,
    TrainingPhase,
)

__all__ = [
    "SHADA",
    "SHADAConfig",
    "create_config",
    "ModelTier",
    "TaskType",
    "TrainingPhase",
]