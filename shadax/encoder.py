"""
SHADA hierarchical multi-modal encoder.

This module defines :class:`HierarchicalEncoder`, the four-stage backbone that
turns either an image ``(B, C, H, W)`` or a text token sequence ``(B, L)`` into
a hierarchy of feature maps plus pooled global features.

The encoder is built entirely from the primitives in :mod:`shadax.modules` and
driven by a :class:`shadax.config.SHADAConfig`. It has two input paths that
share their per-stage Transformer blocks:

* **Image path** -- a :class:`~shadax.modules.ConvStem` (``/4``), then four
  stages each consisting of a :class:`~shadax.modules.ConditionalPositionalEncoding`
  and a list of :class:`~shadax.modules.TransformerBlock` s operating on the
  flattened tokens, with a :class:`~shadax.modules.StageDownsample` (``/2``)
  between consecutive stages. Total spatial reduction is
  :data:`~shadax.config.TOTAL_REDUCTION` (= 32).
* **Text path** -- a :class:`~shadax.modules.TextEmbedding`, then the same four
  stages of Transformer blocks (text tokens are already a sequence), with a
  length-preserving :class:`~shadax.modules.TextStageProject` between stages.

The Transformer blocks are shared across modalities: both paths ultimately call
them on token sequences ``(B, N, dim)``.

The dictionary returned by :meth:`HierarchicalEncoder.forward` is a hard
contract consumed by :mod:`shadax.heads` and :mod:`shadax.ssl`; its keys and
shapes must not change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from shadax.config import (
    NUM_STAGES,
    STEM_REDUCTION,
    TOTAL_REDUCTION,
    Modality,
    SHADAConfig,
)
from shadax.modules import (
    ConditionalPositionalEncoding,
    ConvStem,
    StageDownsample,
    TextEmbedding,
    TextStageProject,
    TransformerBlock,
)


__all__ = ["HierarchicalEncoder"]


class _Stage(nn.Module):
    """A single encoder stage: positional encoding plus Transformer blocks.

    The conditional positional encoding is only used by the image path (it
    operates on spatial maps); the Transformer block list is shared between the
    image and text paths, both of which feed it token sequences.

    Args:
        dim: Channel / token width of this stage.
        depth: Number of Transformer blocks in this stage.
        num_heads: Number of attention heads per block.
        mlp_ratio: Hidden expansion ratio of each block's feed-forward layer.
        dropout: Dropout probability used throughout the stage.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cpe = ConditionalPositionalEncoding(dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )

    def run_blocks(
        self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run every Transformer block in this stage over a token sequence.

        Args:
            tokens: Token sequence of shape ``(B, N, dim)``.
            attn_mask: Optional additive float attention mask ``(N, N)``.

        Returns:
            Token sequence of shape ``(B, N, dim)``.
        """
        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask)
        return tokens


class HierarchicalEncoder(nn.Module):
    """Four-stage hierarchical encoder shared across image and text modalities.

    Built from a :class:`~shadax.config.SHADAConfig`, the encoder exposes a
    single :meth:`forward` that branches on ``modality``. Both branches produce
    four feature maps of increasing channel width (``config.encoder_dims``),
    plus a final token sequence and pooled global features.

    For the image modality, ``H`` and ``W`` must each be divisible by
    :data:`~shadax.config.TOTAL_REDUCTION` (= 32).

    Args:
        config: The validated SHADA configuration describing the model shape.
    """

    def __init__(self, config: SHADAConfig) -> None:
        super().__init__()
        self.config = config
        dims: List[int] = config.encoder_dims
        depths: List[int] = config.encoder_depths
        heads: List[int] = config.num_heads
        mlp_ratio: float = config.mlp_ratio
        dropout: float = config.dropout

        # Image entry point.
        self.stem = ConvStem(config.in_channels, dims[0], dropout=dropout)

        # Text entry point.
        self.text_embed = TextEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=dims[0],
            max_seq_len=config.max_seq_len,
            dropout=dropout,
            pad_token_id=config.pad_token_id,
        )

        # Per-stage modules (Transformer blocks shared across modalities).
        self.stages = nn.ModuleList(
            [
                _Stage(dims[i], depths[i], heads[i], mlp_ratio, dropout)
                for i in range(NUM_STAGES)
            ]
        )

        # Inter-stage reductions: image spatial downsamples and text channel
        # projections (3 each, one between every pair of consecutive stages).
        self.downsamples = nn.ModuleList(
            [StageDownsample(dims[i], dims[i + 1]) for i in range(NUM_STAGES - 1)]
        )
        self.text_projects = nn.ModuleList(
            [TextStageProject(dims[i], dims[i + 1]) for i in range(NUM_STAGES - 1)]
        )

    # ------------------------------------------------------------------ #
    # Forward dispatch.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        modality: Union[str, Modality] = "image",
        causal: bool = False,
    ) -> Dict[str, Any]:
        """Encode an input into a hierarchy of features.

        Args:
            x: Either an image ``(B, in_channels, H, W)`` (image modality) or a
                long tensor of token ids ``(B, L)`` (text modality).
            modality: ``"image"``/``"text"`` (or the corresponding
                :class:`~shadax.config.Modality` enum value).
            causal: Text only. If ``True``, a causal attention mask is applied
                so each position may attend only to itself and earlier
                positions.

        Returns:
            A dict with exactly the keys ``"feature_maps"`` (list of 4
            tensors), ``"tokens"`` ``(B, N_last, dims[-1])``,
            ``"global_features"`` ``(B, dims[-1])``, ``"hw"`` (an ``(Hs, Ws)``
            int tuple for images or ``None`` for text) and ``"modality"`` (the
            resolved modality string).

        Raises:
            ValueError: for the image modality if ``H`` or ``W`` is not
                divisible by :data:`~shadax.config.TOTAL_REDUCTION`, or if
                ``modality`` is not a recognised value.
        """
        modality = Modality(modality)
        if modality is Modality.IMAGE:
            return self._forward_image(x)
        if modality is Modality.TEXT:
            return self._forward_text(x, causal=causal)
        raise ValueError(f"unsupported modality: {modality!r}")

    def forward_features(
        self,
        x: torch.Tensor,
        modality: Union[str, Modality] = "image",
        causal: bool = False,
    ) -> List[torch.Tensor]:
        """Return only the list of four hierarchical feature maps.

        Convenience wrapper around :meth:`forward` that discards everything
        except the ``"feature_maps"`` entry.

        Args:
            x: Input image or token ids (see :meth:`forward`).
            modality: Input modality (see :meth:`forward`).
            causal: Whether to apply a causal mask (text only).

        Returns:
            The list of four feature-map tensors.
        """
        return self.forward(x, modality=modality, causal=causal)["feature_maps"]

    # ------------------------------------------------------------------ #
    # Modality-specific paths.
    # ------------------------------------------------------------------ #
    def _forward_image(self, x: torch.Tensor) -> Dict[str, Any]:
        """Image path of :meth:`forward` (see its docstring)."""
        _, _, height, width = x.shape
        if height % TOTAL_REDUCTION != 0 or width % TOTAL_REDUCTION != 0:
            raise ValueError(
                f"image height and width must each be divisible by "
                f"{TOTAL_REDUCTION}, got H={height}, W={width}"
            )

        m = self.stem(x)  # (B, dims[0], H/4, W/4)
        feature_maps: List[torch.Tensor] = []

        for i, stage in enumerate(self.stages):
            m = stage.cpe(m)
            b, c, h, w = m.shape
            tokens = m.flatten(2).transpose(1, 2)  # (B, H*W, C)
            tokens = stage.run_blocks(tokens)
            m = tokens.transpose(1, 2).reshape(b, c, h, w)  # (B, C, H, W)
            feature_maps.append(m)
            if i < NUM_STAGES - 1:
                m = self.downsamples[i](m)

        last = feature_maps[-1]
        b, c, h, w = last.shape
        tokens = last.flatten(2).transpose(1, 2)  # (B, N_last, dims[-1])
        global_features = tokens.mean(dim=1)  # (B, dims[-1])

        return {
            "feature_maps": feature_maps,
            "tokens": tokens,
            "global_features": global_features,
            "hw": (h, w),
            "modality": Modality.IMAGE.value,
        }

    def _forward_text(self, x: torch.Tensor, causal: bool = False) -> Dict[str, Any]:
        """Text path of :meth:`forward` (see its docstring)."""
        t = self.text_embed(x)  # (B, L, dims[0])
        seq_len = t.shape[1]

        attn_mask: Optional[torch.Tensor] = None
        if causal:
            attn_mask = self._causal_mask(seq_len, device=t.device, dtype=t.dtype)

        feature_maps: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            t = stage.run_blocks(t, attn_mask=attn_mask)  # (B, L, dims[i])
            feature_maps.append(t)
            if i < NUM_STAGES - 1:
                t = self.text_projects[i](t)  # (B, L, dims[i+1])

        tokens = t  # (B, L, dims[-1])
        global_features = tokens.mean(dim=1)  # (B, dims[-1])

        return {
            "feature_maps": feature_maps,
            "tokens": tokens,
            "global_features": global_features,
            "hw": None,
            "modality": Modality.TEXT.value,
        }

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build an additive causal attention mask.

        Args:
            seq_len: Sequence length ``L``.
            device: Device of the produced mask.
            dtype: Floating dtype of the produced mask.

        Returns:
            A ``(L, L)`` float tensor that is ``0`` on and below the diagonal
            and ``-inf`` above it, suitable for passing as ``attn_mask`` to
            :class:`~shadax.modules.TransformerBlock`.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)
