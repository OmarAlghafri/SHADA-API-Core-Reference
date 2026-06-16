"""
Shared neural-network primitives for SHADA.

This module collects the reusable :class:`torch.nn.Module` building blocks that
the SHADA hierarchical encoder (and, downstream, the heads and self-supervised
decoders) are assembled from. Every primitive here is modality-aware in the
sense that it operates either on image spatial maps ``(B, C, H, W)`` or on
token sequences ``(B, N, C)``; the encoder is responsible for converting
between the two layouts.

The two spatial primitives -- :class:`ConvStem` and :class:`StageDownsample` --
realise the spatial contract documented in :mod:`shadax.config`: the stem
reduces ``H`` and ``W`` by :data:`shadax.config.STEM_REDUCTION` (= 4) and each
stage downsample halves them (:data:`shadax.config.STAGE_REDUCTION` = 2).

The positional information for the image path is supplied by
:class:`ConditionalPositionalEncoding`, a depthwise convolution applied
residually to the spatial map. Because it is convolutional it adapts to any
input resolution -- there is no fixed-length positional table for images, which
is what makes the encoder resolution-adaptive. The text path instead uses the
learned positional table in :class:`TextEmbedding`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


__all__ = [
    "ConvStem",
    "ConditionalPositionalEncoding",
    "TransformerBlock",
    "StageDownsample",
    "TextEmbedding",
    "TextStageProject",
]


class ConvStem(nn.Module):
    """Convolutional stem that downsamples an image by a factor of 4.

    Two stride-2 ``3x3`` convolutions (with a GELU non-linearity and a
    normalization in between) map an input image to the first-stage feature
    map. This realises the :data:`shadax.config.STEM_REDUCTION` (= 4) part of
    the spatial contract.

    Args:
        in_channels: Number of input image channels (e.g. 3 for RGB).
        embed_dim: Channel width of the produced feature map (first encoder
            stage width).
        dropout: Dropout probability applied to the output feature map.

    Shape:
        - Input: ``(B, in_channels, H, W)``
        - Output: ``(B, embed_dim, H / 4, W / 4)``
    """

    def __init__(self, in_channels: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = embed_dim // 2
        self.proj1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image to a ``/4`` spatial feature map.

        Args:
            x: Input image of shape ``(B, in_channels, H, W)``.

        Returns:
            Feature map of shape ``(B, embed_dim, H / 4, W / 4)``.
        """
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class ConditionalPositionalEncoding(nn.Module):
    """Resolution-agnostic positional encoding for image spatial maps.

    A single depthwise ``3x3`` convolution is applied to the spatial map and
    added back residually. The convolution encodes the relative position of
    each location from its neighbours, so -- unlike a fixed-length positional
    table -- it works for arbitrary input resolutions. This is what keeps the
    image path of the encoder adaptive to any ``H, W``.

    Args:
        dim: Number of channels of the spatial map (also the number of
            depthwise groups).

    Shape:
        - Input: ``(B, dim, H, W)``
        - Output: ``(B, dim, H, W)``
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add depthwise-convolutional positional information residually.

        Args:
            x: Spatial map of shape ``(B, dim, H, W)``.

        Returns:
            Spatial map of shape ``(B, dim, H, W)``.
        """
        return x + self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block operating on token sequences.

    The block follows the standard pre-norm residual structure::

        x = x + Attn(LN(x))
        x = x + MLP(LN(x))

    Self-attention is implemented with :class:`torch.nn.MultiheadAttention` in
    ``batch_first`` layout. The first LayerNorm is computed once and reused for
    the query, key and value inputs of the attention.

    Args:
        dim: Token / embedding dimension.
        num_heads: Number of attention heads (``dim`` must be divisible by it).
        mlp_ratio: Hidden expansion ratio of the feed-forward sub-layer.
        dropout: Dropout probability used in the attention and the MLP.

    Shape:
        - Input: ``(B, N, dim)``
        - Output: ``(B, N, dim)``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply self-attention and the feed-forward sub-layer residually.

        Args:
            x: Token sequence of shape ``(B, N, dim)``.
            attn_mask: Optional additive float attention mask of shape
                ``(N, N)``. Entries set to ``-inf`` are forbidden attention
                links (e.g. a causal mask); ``0`` entries are allowed.

        Returns:
            Token sequence of shape ``(B, N, dim)``.
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, attn_mask=attn_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class StageDownsample(nn.Module):
    """Patch-merging downsample for image spatial maps.

    A normalization followed by a stride-2 ``2x2`` convolution halves the
    spatial resolution while changing the channel width from ``in_dim`` to
    ``out_dim``. This realises the :data:`shadax.config.STAGE_REDUCTION` (= 2)
    inter-stage reduction.

    Args:
        in_dim: Input channel width.
        out_dim: Output channel width.

    Shape:
        - Input: ``(B, in_dim, H, W)``
        - Output: ``(B, out_dim, H / 2, W / 2)``
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the spatial resolution and project the channels.

        Args:
            x: Spatial map of shape ``(B, in_dim, H, W)``.

        Returns:
            Spatial map of shape ``(B, out_dim, H / 2, W / 2)``.
        """
        x = self.norm(x)
        x = self.proj(x)
        return x


class TextEmbedding(nn.Module):
    """Token + learned positional embedding for the text modality.

    Maps a batch of integer token ids to dense embeddings, adds a learned
    positional embedding (one entry per position up to ``max_seq_len``) and
    applies dropout.

    Args:
        vocab_size: Size of the token vocabulary.
        embed_dim: Embedding dimension (first encoder stage width).
        max_seq_len: Maximum supported sequence length; the positional table
            has this many entries.
        dropout: Dropout probability applied to the summed embeddings.
        pad_token_id: Token id used as ``padding_idx`` of the token embedding
            (its embedding is fixed to zeros and not updated by gradients).

    Shape:
        - Input: ``(B, L)`` long token ids with ``L <= max_seq_len``
        - Output: ``(B, L, embed_dim)``
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token ids and add learned positional information.

        Args:
            x: Long tensor of token ids, shape ``(B, L)``.

        Returns:
            Embedding tensor of shape ``(B, L, embed_dim)``.

        Raises:
            ValueError: if the sequence length ``L`` exceeds ``max_seq_len``.
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=x.device)
        tokens = self.token_embed(x)
        pos = self.pos_embed(positions).unsqueeze(0)
        return self.drop(tokens + pos)


class TextStageProject(nn.Module):
    """Length-preserving channel projection for the text path.

    Applies a LayerNorm followed by a linear projection that changes the
    channel dimension only; the sequence length is unchanged. This is the text
    analogue of :class:`StageDownsample` (which, for images, also reduces the
    spatial resolution).

    Args:
        in_dim: Input channel width.
        out_dim: Output channel width.

    Shape:
        - Input: ``(B, L, in_dim)``
        - Output: ``(B, L, out_dim)``
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the channel dimension while preserving sequence length.

        Args:
            x: Token sequence of shape ``(B, L, in_dim)``.

        Returns:
            Token sequence of shape ``(B, L, out_dim)``.
        """
        return self.proj(self.norm(x))
