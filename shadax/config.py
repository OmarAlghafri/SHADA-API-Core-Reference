"""
SHADA configuration layer.

This module is the *contract* that every other module in :mod:`shadax`
depends on. It is intentionally free of any heavy dependency (no PyTorch),
so it can be imported cheaply and reasoned about in isolation.

It defines:

* :class:`ModelTier`, :class:`TaskType`, :class:`TrainingPhase`,
  :class:`Modality` -- the controlled vocabularies used across the library.
* :class:`SHADAConfig` -- the single source of truth for the model shape.
* :data:`_TIER_CONFIGS` -- the per-tier architectural presets.
* :func:`create_config` -- the factory used by the high level API.

Spatial contract (images)
--------------------------
The hierarchical encoder reduces the spatial resolution by a fixed total
factor of ``32`` (a ``/4`` convolutional stem followed by three ``/2`` stage
downsamples). Input height and width must therefore be divisible by ``32``;
:func:`SHADAConfig.validate` and the encoder both enforce this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

__all__ = [
    "ModelTier",
    "TaskType",
    "TrainingPhase",
    "Modality",
    "SHADAConfig",
    "create_config",
    "STEM_REDUCTION",
    "STAGE_REDUCTION",
    "TOTAL_REDUCTION",
    "NUM_STAGES",
]

# --------------------------------------------------------------------------- #
# Architectural constants (the spatial contract).
# --------------------------------------------------------------------------- #
NUM_STAGES: int = 4
STEM_REDUCTION: int = 4            # convolutional stem downsamples H, W by 4.
STAGE_REDUCTION: int = 2          # each inter-stage downsample halves H, W.
# stem (/4) + 3 downsamples (/2 each) -> /32 total.
TOTAL_REDUCTION: int = STEM_REDUCTION * (STAGE_REDUCTION ** (NUM_STAGES - 1))


class ModelTier(str, Enum):
    """Model size tiers, ordered from smallest to largest."""

    NANO = "nano"
    BASE = "base"
    LARGE = "large"
    XL = "xl"


class TaskType(str, Enum):
    """Supported downstream tasks.

    Each value maps to a concrete task head in :mod:`shadax.heads` and to a
    loss in :mod:`shadax.training`.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    LANGUAGE_MODEL = "lm"


class TrainingPhase(str, Enum):
    """The four phases of the SHADA training pipeline.

    * ``PRETRAIN``  -- self-supervised pretraining (no labels required).
    * ``MULTITASK`` -- joint optimisation of the supervised objective and the
      self-supervised objective.
    * ``FINETUNE``  -- purely supervised optimisation of the target task.
    * ``DEPLOY``    -- inference / frozen-weights mode (no optimisation).
    """

    PRETRAIN = "pretrain"
    MULTITASK = "multitask"
    FINETUNE = "finetune"
    DEPLOY = "deploy"


class Modality(str, Enum):
    """Input modalities understood by the encoder."""

    IMAGE = "image"
    TEXT = "text"


@dataclass
class SHADAConfig:
    """Single source of truth for the SHADA architecture.

    Attributes:
        tier: Model size tier name (``nano``/``base``/``large``/``xl``).
        encoder_dims: Channel width of each of the four encoder stages.
        encoder_depths: Number of transformer blocks in each stage.
        num_heads: Number of attention heads in each stage.
        mlp_ratio: Hidden expansion ratio of the feed-forward sub-layer.
        dropout: Dropout probability used throughout the network.
        in_channels: Number of input image channels (e.g. 3 for RGB).
        image_size: Reference image size used for defaults; the encoder is
            resolution-adaptive, so this is only a hint (any H, W divisible by
            :data:`TOTAL_REDUCTION` works).
        max_seq_len: Maximum text sequence length supported.
        vocab_size: Vocabulary size for the text modality / language model.
        task: Primary :class:`TaskType` value (stored as ``str``).
        num_classes: Number of output classes. Interpreted per task:
            classification -> number of labels, detection -> object
            categories, segmentation -> per-pixel classes. Ignored for ``lm``
            (which predicts over ``vocab_size``).
        mask_ratio: Fraction of image patches masked during self-supervised
            pretraining (masked image modeling).
        text_mask_ratio: Fraction of text tokens masked during masked language
            modeling.
        decoder_dim: Width of the lightweight self-supervised decoder.
        decoder_depth: Number of blocks in the self-supervised decoder.
        pad_token_id: Token id treated as padding / ignored by the LM loss.
    """

    tier: str = "base"
    encoder_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    encoder_depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    in_channels: int = 3
    image_size: int = 224
    max_seq_len: int = 1024
    vocab_size: int = 50257
    task: str = "classification"
    num_classes: int = 1000
    mask_ratio: float = 0.75
    text_mask_ratio: float = 0.15
    decoder_dim: int = 256
    decoder_depth: int = 2
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        self.validate()

    # ----------------------------------------------------------------- #
    # Convenience accessors used by the rest of the library.
    # ----------------------------------------------------------------- #
    @property
    def embed_dim(self) -> int:
        """Width of the first (stem) stage."""
        return self.encoder_dims[0]

    @property
    def final_dim(self) -> int:
        """Width of the last encoder stage (the global feature dimension)."""
        return self.encoder_dims[-1]

    @property
    def num_stages(self) -> int:
        return len(self.encoder_dims)

    @property
    def task_type(self) -> TaskType:
        """The :class:`TaskType` enum corresponding to ``self.task``."""
        return TaskType(self.task)

    def stage_reduction(self, stage: int) -> int:
        """Total spatial reduction factor at the output of ``stage`` (0-based)."""
        return STEM_REDUCTION * (STAGE_REDUCTION ** stage)

    def validate(self) -> "SHADAConfig":
        """Validate internal consistency. Returns ``self`` for chaining.

        Raises:
            ValueError: if the per-stage lists are inconsistent or any value
                is out of range.
        """
        n = NUM_STAGES
        for name, seq in (
            ("encoder_dims", self.encoder_dims),
            ("encoder_depths", self.encoder_depths),
            ("num_heads", self.num_heads),
        ):
            if len(seq) != n:
                raise ValueError(
                    f"{name} must have exactly {n} entries, got {len(seq)}: {seq!r}"
                )
        for dim, heads in zip(self.encoder_dims, self.num_heads):
            if dim % heads != 0:
                raise ValueError(
                    f"each encoder dim must be divisible by its head count; "
                    f"dim={dim} is not divisible by num_heads={heads}"
                )
        if self.task not in {t.value for t in TaskType}:
            raise ValueError(
                f"task must be one of {[t.value for t in TaskType]}, got {self.task!r}"
            )
        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}")
        if not (0.0 <= self.text_mask_ratio < 1.0):
            raise ValueError(
                f"text_mask_ratio must be in [0, 1), got {self.text_mask_ratio}"
            )
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        return self


# --------------------------------------------------------------------------- #
# Per-tier presets. Only architectural fields are stored here; task-specific
# fields (task, num_classes, ...) are supplied by create_config / the user.
# --------------------------------------------------------------------------- #
_TIER_CONFIGS: Dict[str, Dict] = {
    "nano": {
        "encoder_dims": [64, 128, 256, 512],
        "encoder_depths": [2, 2, 4, 2],
        "num_heads": [2, 4, 8, 16],
        "max_seq_len": 512,
        "decoder_dim": 128,
        "decoder_depth": 1,
    },
    "base": {
        "encoder_dims": [128, 256, 512, 1024],
        "encoder_depths": [3, 4, 6, 3],
        "num_heads": [4, 8, 16, 32],
        "max_seq_len": 1024,
        "decoder_dim": 256,
        "decoder_depth": 2,
    },
    "large": {
        "encoder_dims": [192, 384, 768, 1536],
        "encoder_depths": [3, 4, 18, 3],
        "num_heads": [6, 12, 24, 48],
        "max_seq_len": 2048,
        "decoder_dim": 384,
        "decoder_depth": 2,
    },
    "xl": {
        "encoder_dims": [256, 512, 1024, 2048],
        "encoder_depths": [3, 4, 24, 3],
        "num_heads": [8, 16, 32, 64],
        "max_seq_len": 4096,
        "decoder_dim": 512,
        "decoder_depth": 4,
    },
}


def create_config(
    tier: str = "base",
    task: str = "classification",
    num_classes: int = 1000,
    **overrides,
) -> SHADAConfig:
    """Create a :class:`SHADAConfig` with tier-specific defaults.

    Args:
        tier: Model tier (``nano``/``base``/``large``/``xl``).
        task: Task type (``classification``/``detection``/``segmentation``/``lm``).
        num_classes: Number of output classes (see :class:`SHADAConfig`).
        **overrides: Any :class:`SHADAConfig` field to override.

    Returns:
        A validated :class:`SHADAConfig` instance.

    Raises:
        ValueError: if ``tier`` is unknown or the resulting config is invalid.

    Example:
        >>> config = create_config("base", task="classification", num_classes=10)
        >>> config.final_dim
        1024
    """
    if isinstance(tier, ModelTier):
        tier = tier.value
    if tier not in _TIER_CONFIGS:
        raise ValueError(
            f"tier must be one of {list(_TIER_CONFIGS)}, got {tier!r}"
        )
    if isinstance(task, TaskType):
        task = task.value
    kwargs = {
        **_TIER_CONFIGS[tier],
        "tier": tier,
        "task": task,
        "num_classes": num_classes,
        **overrides,
    }
    return SHADAConfig(**kwargs)
