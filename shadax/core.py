"""
Backward-compatibility shim for :mod:`shadax.core`.

The original monolithic ``core.py`` has been split into focused modules
(:mod:`shadax.config`, :mod:`shadax.network`, :mod:`shadax.model`, ...). This
module preserves the old import surface so that existing code such as::

    from shadax.core import SHADA, SHADAConfig, create_config
    from shadax.core import ModelTier, TaskType, TrainingPhase

keeps working unchanged. Everything is re-exported from its new home.
"""

from __future__ import annotations

from shadax.config import (
    Modality,
    ModelTier,
    SHADAConfig,
    TaskType,
    TrainingPhase,
    create_config,
)
from shadax.model import SHADA

__all__ = [
    "SHADA",
    "SHADAConfig",
    "create_config",
    "ModelTier",
    "TaskType",
    "TrainingPhase",
    "Modality",
]
