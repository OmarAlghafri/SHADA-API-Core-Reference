"""
SHADA unified network.

This module defines :class:`SHADANet`, the single :class:`torch.nn.Module` that
ties together the three decoupled pieces of the library:

* the :class:`~shadax.encoder.HierarchicalEncoder` backbone (shared across the
  image and text modalities);
* the task head selected by :func:`~shadax.heads.build_head`
  (classification / language-model / segmentation / detection); and
* the matching self-supervised objective from :mod:`shadax.ssl`
  (:class:`~shadax.ssl.MaskedImageModeling` for image tasks,
  :class:`~shadax.ssl.MaskedLanguageModeling` for the language-model task).

Holding the self-supervised objective *as a submodule* (rather than rebuilding
it per step via :func:`~shadax.ssl.compute_ssl_loss`) is what makes the image
decoder's weights persistent and therefore trainable across the
:class:`~shadax.config.TrainingPhase.PRETRAIN` / ``MULTITASK`` phases.

For the language-model task the very same :class:`~shadax.heads.LanguageModelHead`
is used both as the supervised next-token predictor (in :meth:`SHADANet.forward`)
and as the masked-LM predictor injected into
:class:`~shadax.ssl.MaskedLanguageModeling` (in :meth:`SHADANet.ssl_loss`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from shadax.config import Modality, SHADAConfig, TaskType
from shadax.encoder import HierarchicalEncoder
from shadax.heads import build_head
from shadax.ssl import MaskedImageModeling, MaskedLanguageModeling


__all__ = ["SHADANet"]


class SHADANet(nn.Module):
    """Unified SHADA model: shared encoder + task head + SSL objective.

    The network owns one :class:`~shadax.encoder.HierarchicalEncoder`, one task
    head (chosen by ``config.task_type``) and the self-supervised objective that
    matches the task's primary modality. The primary modality is text for the
    language-model task and image for every other task.

    Args:
        config: The validated SHADA configuration describing the model shape.

    Attributes:
        encoder: The shared hierarchical backbone.
        head: The task head for ``config.task_type``.
        mim: The :class:`~shadax.ssl.MaskedImageModeling` objective (image
            tasks only; absent for the language-model task).
        mlm: The :class:`~shadax.ssl.MaskedLanguageModeling` objective
            (language-model task only; absent otherwise).
        config: The configuration the network was built from.
    """

    def __init__(self, config: SHADAConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = HierarchicalEncoder(config)
        self.head = build_head(config)

        # Primary modality: text for the language model, image otherwise. The
        # matching self-supervised objective is held as a submodule so its
        # parameters are persistent (and therefore trainable) across phases.
        if config.task_type is TaskType.LANGUAGE_MODEL:
            self.primary_modality = Modality.TEXT.value
            # The LM head doubles as the masked-LM predictor.
            self.mlm = MaskedLanguageModeling(config)
        else:
            self.primary_modality = Modality.IMAGE.value
            self.mim = MaskedImageModeling(config)

    # ------------------------------------------------------------------ #
    # Modality resolution.
    # ------------------------------------------------------------------ #
    def _resolve_modality(self, x: Tensor, modality: Optional[str]) -> str:
        """Resolve the input modality.

        Resolution order: an explicit ``modality`` argument always wins; else
        the modality is inferred from ``x`` (a floating tensor with ``ndim == 4``
        is an image; an integer tensor with ``ndim == 2`` is text); else the
        network's primary modality is used as a fallback.

        Args:
            x: The input tensor (image or token ids).
            modality: Optional explicit modality string/enum.

        Returns:
            The resolved modality string (``"image"`` or ``"text"``).
        """
        if modality is not None:
            return Modality(modality).value
        if torch.is_floating_point(x) and x.dim() == 4:
            return Modality.IMAGE.value
        if not torch.is_floating_point(x) and x.dim() == 2:
            return Modality.TEXT.value
        return self.primary_modality

    # ------------------------------------------------------------------ #
    # Forward / encode / SSL.
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor, modality: Optional[str] = None) -> Dict[str, Tensor]:
        """Run the encoder and route its output through the task head.

        Args:
            x: Either an image ``(B, in_channels, H, W)`` or a long tensor of
                token ids ``(B, L)``.
            modality: Optional explicit modality (``"image"``/``"text"``). When
                omitted it is inferred from ``x`` (see :meth:`_resolve_modality`).

        Returns:
            A dict whose task-specific key holds the prediction and which always
            also carries ``"global_features"`` ``(B, final_dim)``:

            * classification -> ``{"logits": (B, num_classes), ...}``
            * language model -> ``{"logits": (B, L, vocab_size), ...}``
            * segmentation -> ``{"segmentation": (B, num_classes, H, W), ...}``
            * detection -> ``{"heatmap": ..., "wh": ..., "offset": ..., ...}``
        """
        modality = self._resolve_modality(x, modality)
        causal = (
            self.config.task_type is TaskType.LANGUAGE_MODEL
            and modality == Modality.TEXT.value
        )
        enc = self.encoder(x, modality=modality, causal=causal)

        task = self.config.task_type
        out: Dict[str, Tensor]
        if task is TaskType.CLASSIFICATION:
            out = {"logits": self.head(enc["global_features"])}
        elif task is TaskType.LANGUAGE_MODEL:
            out = {"logits": self.head(enc["tokens"])}
        elif task is TaskType.SEGMENTATION:
            out = {
                "segmentation": self.head(
                    enc["feature_maps"], output_size=(x.shape[-2], x.shape[-1])
                )
            }
        elif task is TaskType.DETECTION:
            out = dict(self.head(enc["feature_maps"]))
        else:  # pragma: no cover - guarded by config.validate().
            raise ValueError(f"unsupported task type: {task!r}")

        out["global_features"] = enc["global_features"]
        return out

    def ssl_loss(self, x: Tensor, modality: Optional[str] = None) -> Dict[str, Tensor]:
        """Compute the self-supervised loss matching the input modality.

        Args:
            x: Either an image ``(B, in_channels, H, W)`` or a long tensor of
                token ids ``(B, L)``.
            modality: Optional explicit modality (see :meth:`_resolve_modality`).

        Returns:
            The dict returned by the underlying objective's ``forward`` (see
            :class:`~shadax.ssl.MaskedImageModeling` /
            :class:`~shadax.ssl.MaskedLanguageModeling`), always with a scalar
            ``"loss"`` entry.

        Raises:
            RuntimeError: if the resolved modality has no SSL objective on this
                network (e.g. text input to an image task).
        """
        modality = self._resolve_modality(x, modality)
        if modality == Modality.IMAGE.value:
            if not hasattr(self, "mim"):
                raise RuntimeError(
                    "this network has no image SSL objective "
                    f"(task={self.config.task!r})"
                )
            return self.mim(self.encoder, x)
        if not hasattr(self, "mlm"):
            raise RuntimeError(
                "this network has no text SSL objective "
                f"(task={self.config.task!r})"
            )
        return self.mlm(self.encoder, self.head, x)

    def encode(self, x: Tensor, modality: Optional[str] = None) -> Dict[str, Any]:
        """Return the raw encoder output dict (used for feature extraction).

        Args:
            x: Either an image ``(B, in_channels, H, W)`` or token ids ``(B, L)``.
            modality: Optional explicit modality (see :meth:`_resolve_modality`).

        Returns:
            The encoder output dict (see
            :meth:`shadax.encoder.HierarchicalEncoder.forward`).
        """
        modality = self._resolve_modality(x, modality)
        causal = (
            self.config.task_type is TaskType.LANGUAGE_MODEL
            and modality == Modality.TEXT.value
        )
        return self.encoder(x, modality=modality, causal=causal)
