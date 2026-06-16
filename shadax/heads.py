"""
SHADA task heads.

This module defines the four supervised task heads that turn the outputs of
:class:`shadax.encoder.HierarchicalEncoder` into task-specific predictions:

* :class:`ClassificationHead` -- image / sequence classification from the
  pooled ``global_features``.
* :class:`LanguageModelHead` -- per-token vocabulary logits from ``tokens``.
* :class:`SegmentationHead` -- an FPN-style dense per-pixel logit map built
  from the four image ``feature_maps``.
* :class:`DetectionHead` -- an anchor-free, CenterNet-style detector operating
  on the last (coarsest) image feature map.

The heads are deliberately *decoupled* from the encoder: each consumes only the
specific entry (or entries) of the encoder output dict that it needs, expressed
as plain tensors, and this module does not import the encoder. The convenience
factory :func:`build_head` selects the head matching a
:class:`~shadax.config.SHADAConfig`'s :class:`~shadax.config.TaskType`.

The shapes referenced below follow the encoder output contract documented in
:mod:`shadax.encoder`: ``feature_maps`` is a list of four spatial maps of
strides ``4, 8, 16, 32`` (image modality) and increasing channel width
``config.encoder_dims``; ``tokens`` is ``(B, N, config.final_dim)``; and
``global_features`` is ``(B, config.final_dim)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shadax.config import SHADAConfig, TaskType


__all__ = [
    "ClassificationHead",
    "LanguageModelHead",
    "SegmentationHead",
    "DetectionHead",
    "build_head",
]


class ClassificationHead(nn.Module):
    """Linear classification head over pooled global features.

    Normalises the pooled global feature vector, applies dropout, and projects
    it to per-class logits. It consumes only the ``global_features`` entry of
    the encoder output.

    Args:
        in_dim: Width of the input feature vector (``config.final_dim``).
        num_classes: Number of output classes.
        dropout: Dropout probability applied before the final projection.

    Shape:
        - Input: ``(B, in_dim)``
        - Output: ``(B, num_classes)``
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, num_classes)

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """Project pooled global features to class logits.

        Args:
            global_features: Pooled feature vector of shape ``(B, in_dim)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        x = self.norm(global_features)
        x = self.drop(x)
        return self.proj(x)


class LanguageModelHead(nn.Module):
    """Per-token language-modeling head over the vocabulary.

    Normalises each token representation and projects it to vocabulary logits,
    producing one distribution per sequence position. It consumes only the
    ``tokens`` entry of the encoder output.

    Args:
        in_dim: Width of each token representation (``config.final_dim``).
        vocab_size: Size of the output vocabulary.

    Shape:
        - Input: ``(B, L, in_dim)``
        - Output: ``(B, L, vocab_size)``
    """

    def __init__(self, in_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project per-token representations to vocabulary logits.

        Args:
            tokens: Token sequence of shape ``(B, L, in_dim)``.

        Returns:
            Per-token logits of shape ``(B, L, vocab_size)``.
        """
        return self.proj(self.norm(tokens))


class SegmentationHead(nn.Module):
    """FPN-style dense segmentation head over the four image feature maps.

    Each of the four hierarchical feature maps is projected to a common
    ``fpn_dim`` with a ``1x1`` convolution, bilinearly upsampled to the
    resolution of the first (finest, stride-4) feature map, and summed. The
    fused map is refined by a ``3x3`` convolution and projected to per-class
    logits by a final ``1x1`` convolution, then bilinearly upsampled to the
    requested output size (defaulting to the original input resolution, i.e.
    the stride-4 map upsampled by :data:`~shadax.config.STEM_REDUCTION`).

    Args:
        encoder_dims: Channel widths of the four input feature maps
            (``config.encoder_dims``).
        num_classes: Number of per-pixel output classes.
        fpn_dim: Common channel width the feature maps are projected to before
            fusion.

    Shape:
        - Input: list of four maps ``[(B, encoder_dims[i], H_i, W_i)]`` with
          strides ``4, 8, 16, 32``.
        - Output: ``(B, num_classes, H, W)`` where ``(H, W)`` is ``output_size``
          if given, else the stride-4 map size scaled by 4.
    """

    def __init__(
        self,
        encoder_dims: List[int],
        num_classes: int,
        fpn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.upsample_factor = 4  # finest feature map (stride 4) -> input scale.
        self.lateral = nn.ModuleList(
            [nn.Conv2d(dim, fpn_dim, kernel_size=1) for dim in encoder_dims]
        )
        self.fuse = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Fuse the hierarchical feature maps into a dense logit map.

        Args:
            feature_maps: The four image feature maps with strides ``4, 8, 16,
                32`` (channel widths ``encoder_dims``).
            output_size: Optional target spatial size ``(H, W)``. When omitted,
                the output is upsampled to the stride-4 map size scaled by 4
                (the original input resolution).

        Returns:
            Per-pixel logits of shape ``(B, num_classes, H, W)``.
        """
        target_hw = feature_maps[0].shape[-2:]
        fused: Optional[torch.Tensor] = None
        for lateral, fmap in zip(self.lateral, feature_maps):
            proj = lateral(fmap)
            if proj.shape[-2:] != target_hw:
                proj = F.interpolate(
                    proj, size=target_hw, mode="bilinear", align_corners=False
                )
            fused = proj if fused is None else fused + proj

        x = self.fuse(fused)
        x = self.classifier(x)

        if output_size is None:
            output_size = (
                target_hw[0] * self.upsample_factor,
                target_hw[1] * self.upsample_factor,
            )
        return F.interpolate(
            x, size=output_size, mode="bilinear", align_corners=False
        )


class DetectionHead(nn.Module):
    """Anchor-free, CenterNet-style detection head.

    Operates on the last (coarsest, stride-32) image feature map and predicts
    three dense maps from three independent convolutional branches: a per-class
    center heatmap (raw logits), a box width/height map, and a sub-pixel center
    offset map. Each branch is a ``3x3`` convolution with a ReLU followed by a
    ``1x1`` convolution to the branch-specific output channels.

    Args:
        in_dim: Channel width of the last feature map (``config.final_dim``).
        num_classes: Number of object categories (heatmap channels).
        head_dim: Hidden channel width of each branch's first convolution.

    Shape:
        - Input: list of feature maps; only the last ``(B, in_dim, Hs, Ws)`` is
          used.
        - Output: dict with ``"heatmap"`` ``(B, num_classes, Hs, Ws)``, ``"wh"``
          ``(B, 2, Hs, Ws)`` and ``"offset"`` ``(B, 2, Hs, Ws)``.
    """

    def __init__(self, in_dim: int, num_classes: int, head_dim: int = 128) -> None:
        super().__init__()
        self.heatmap = self._branch(in_dim, head_dim, num_classes)
        self.wh = self._branch(in_dim, head_dim, 2)
        self.offset = self._branch(in_dim, head_dim, 2)

    @staticmethod
    def _branch(in_dim: int, head_dim: int, out_ch: int) -> nn.Sequential:
        """Build a single prediction branch.

        Args:
            in_dim: Input channel width.
            head_dim: Hidden channel width of the ``3x3`` convolution.
            out_ch: Number of output channels of the branch.

        Returns:
            A ``Conv2d(3x3) -> ReLU -> Conv2d(1x1)`` sequential module.
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, out_ch, kernel_size=1),
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict center heatmap, box size and center offset maps.

        Args:
            feature_maps: The image feature maps; only the last one,
                ``(B, in_dim, Hs, Ws)``, is consumed.

        Returns:
            A dict with keys ``"heatmap"`` ``(B, num_classes, Hs, Ws)``,
            ``"wh"`` ``(B, 2, Hs, Ws)`` and ``"offset"`` ``(B, 2, Hs, Ws)``.
        """
        x = feature_maps[-1]
        return {
            "heatmap": self.heatmap(x),
            "wh": self.wh(x),
            "offset": self.offset(x),
        }


def build_head(config: SHADAConfig) -> nn.Module:
    """Build the task head matching a configuration's task type.

    Args:
        config: The validated SHADA configuration; its
            :attr:`~shadax.config.SHADAConfig.task_type` selects the head.

    Returns:
        The :class:`torch.nn.Module` head for ``config.task_type``:
        :class:`ClassificationHead`, :class:`LanguageModelHead`,
        :class:`SegmentationHead` or :class:`DetectionHead`.

    Raises:
        ValueError: if ``config.task_type`` is not a supported task.
    """
    task = config.task_type
    if task is TaskType.CLASSIFICATION:
        return ClassificationHead(config.final_dim, config.num_classes, config.dropout)
    if task is TaskType.LANGUAGE_MODEL:
        return LanguageModelHead(config.final_dim, config.vocab_size)
    if task is TaskType.SEGMENTATION:
        return SegmentationHead(config.encoder_dims, config.num_classes)
    if task is TaskType.DETECTION:
        return DetectionHead(config.final_dim, config.num_classes)
    raise ValueError(f"unsupported task type: {task!r}")
