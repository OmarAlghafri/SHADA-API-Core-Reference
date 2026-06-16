"""
SHADA self-supervised learning (SSL) objectives.

This module defines the two self-supervised pretraining objectives used by the
SHADA training pipeline (:class:`shadax.config.TrainingPhase.PRETRAIN` and
``MULTITASK``):

* :class:`MaskedImageModeling` -- a MAE-style masked image modeling objective.
  A random fraction of the input image is masked, the (masked) image is passed
  through the encoder, and a lightweight convolutional decoder reconstructs the
  original pixels. The loss is the reconstruction error over the masked region.
* :class:`MaskedLanguageModeling` -- a BERT-style masked language modeling
  objective. A random fraction of the (non-padding) tokens are replaced by a
  reserved ``[MASK]`` id, the masked sequence is encoded, and a language-model
  head predicts the original token ids. The loss is the cross-entropy over the
  masked positions only.

Dependency injection
---------------------
Neither objective imports :class:`shadax.encoder.HierarchicalEncoder` or any
task head. Instead the encoder (and, for MLM, the LM head) are passed *in* at
:meth:`forward` time. This keeps :mod:`shadax.ssl` decoupled from
:mod:`shadax.encoder` / :mod:`shadax.heads`: the only contract is the
dictionary returned by ``encoder(x, modality=...)`` (see
:meth:`shadax.encoder.HierarchicalEncoder.forward`).

Persistent vs one-off use
--------------------------
Each objective is an :class:`torch.nn.Module` that owns learnable parameters
(the image decoder and mask token; the MLM objective itself is parameter-free
but is still a module for uniformity). The persistent owner of these modules is
the high-level network (built later), which should hold a single instance and
reuse it across steps so the decoder weights are trained. The
:func:`compute_ssl_loss` dispatcher is a thin convenience helper that
*constructs a fresh objective on every call*; it is therefore only suitable for
one-off / stateless calls (e.g. quick experiments or tests) and must NOT be
used in a training loop, since it would discard the decoder weights each step.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shadax.config import TOTAL_REDUCTION, SHADAConfig


__all__ = ["MaskedImageModeling", "MaskedLanguageModeling", "compute_ssl_loss"]


class MaskedImageModeling(nn.Module):
    """MAE-style masked image modeling objective.

    A random set of image patches (at granularity :data:`~shadax.config.TOTAL_REDUCTION`
    = 32) is masked: the masked pixels are replaced by a learnable mask token,
    the resulting image is passed through the injected encoder, and a
    lightweight convolutional decoder reconstructs the full-resolution image
    from the encoder's coarsest feature map. The loss is the mean-squared error
    between the reconstruction and the original image, computed over the masked
    pixels only.

    The decoder is sized from ``config.decoder_dim`` / ``config.decoder_depth``
    and consists of a ``1x1`` projection from ``config.final_dim`` to
    ``decoder_dim``, ``decoder_depth`` ``Conv2d 3x3`` + ``GELU`` blocks, a
    bilinear upsample back to the input resolution and a final ``1x1`` Conv2d
    that maps to ``config.in_channels``. Upsampling via
    :func:`torch.nn.functional.interpolate` keeps the decoder robust to any
    input resolution.

    Args:
        config: The validated SHADA configuration describing the model shape.
    """

    def __init__(self, config: SHADAConfig) -> None:
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.mask_ratio = config.mask_ratio
        self.patch_size = TOTAL_REDUCTION
        final_dim = config.final_dim
        decoder_dim = config.decoder_dim
        decoder_depth = config.decoder_depth

        # Learnable token broadcast over masked pixels of the input image.
        self.mask_token = nn.Parameter(torch.zeros(1, config.in_channels, 1, 1))

        # 1x1 projection from the encoder's final width to the decoder width.
        self.decoder_proj = nn.Conv2d(final_dim, decoder_dim, kernel_size=1)
        # decoder_depth refinement blocks at the coarse resolution.
        self.decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                for _ in range(decoder_depth)
            ]
        )
        # Final 1x1 projection to image channels (applied after upsampling).
        self.decoder_pred = nn.Conv2d(decoder_dim, config.in_channels, kernel_size=1)

    def _random_pixel_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Build a per-pixel boolean mask from a random patch mask.

        A random ``mask_ratio`` fraction (at least one patch) of the
        ``(H / g, W / g)`` patch grid is masked per sample, where
        ``g = TOTAL_REDUCTION``. The patch grid is then upsampled (nearest) to a
        full-resolution per-pixel mask.

        Args:
            images: Input images of shape ``(B, in_channels, H, W)``.

        Returns:
            A float mask of shape ``(B, 1, H, W)`` with ``1.0`` at masked pixels
            and ``0.0`` elsewhere (same device and dtype as ``images``).
        """
        b, _, height, width = images.shape
        g = self.patch_size
        grid_h = height // g
        grid_w = width // g
        num_patches = grid_h * grid_w
        num_mask = max(1, int(round(self.mask_ratio * num_patches)))

        # Per-sample random scores; the lowest `num_mask` scores are masked.
        scores = torch.rand(b, num_patches, device=images.device)
        ids = torch.argsort(scores, dim=1)
        patch_mask = torch.zeros(b, num_patches, device=images.device, dtype=images.dtype)
        patch_mask.scatter_(1, ids[:, :num_mask], 1.0)

        patch_mask = patch_mask.reshape(b, 1, grid_h, grid_w)
        mask = F.interpolate(patch_mask, size=(height, width), mode="nearest")
        return mask

    def forward(self, encoder: nn.Module, images: Tensor) -> Dict[str, Tensor]:
        """Run the masked image modeling objective.

        Args:
            encoder: The hierarchical encoder (injected). Called as
                ``encoder(masked, modality="image")``; only its
                ``"feature_maps"[-1]`` entry of shape
                ``(B, final_dim, H / 32, W / 32)`` is consumed.
            images: Input images of shape ``(B, in_channels, H, W)``. ``H`` and
                ``W`` must each be divisible by
                :data:`~shadax.config.TOTAL_REDUCTION` (the encoder enforces
                this).

        Returns:
            A dict with keys ``"loss"`` (scalar MSE over masked pixels),
            ``"recon"`` (the reconstruction ``(B, in_channels, H, W)``) and
            ``"mask"`` (the per-pixel float mask ``(B, 1, H, W)``).
        """
        _, _, height, width = images.shape

        # 1. Random per-pixel mask at patch granularity g = TOTAL_REDUCTION.
        mask = self._random_pixel_mask(images)  # (B, 1, H, W)

        # 2. Replace masked pixels with the (broadcast) learnable mask token.
        masked = images * (1.0 - mask) + self.mask_token * mask

        # 3. Encode the masked image and take the coarsest feature map.
        enc = encoder(masked, modality="image")
        feat = enc["feature_maps"][-1]  # (B, final_dim, H/32, W/32)

        # 4. Decode back to image resolution.
        x = self.decoder_proj(feat)
        for block in self.decoder_blocks:
            x = block(x)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        recon = self.decoder_pred(x)  # (B, in_channels, H, W)

        # 5. MSE over masked pixels only.
        loss = ((recon - images) ** 2 * mask).sum() / (
            mask.sum() * self.in_channels + 1e-6
        )

        return {"loss": loss, "recon": recon, "mask": mask}


class MaskedLanguageModeling(nn.Module):
    """BERT-style masked language modeling objective.

    A random ``config.text_mask_ratio`` fraction of the non-padding tokens are
    replaced by a reserved ``[MASK]`` id, the masked sequence is encoded, and an
    injected language-model head predicts the original token ids. The loss is
    the cross-entropy over the masked positions only.

    ``mask_token_id`` defaults to ``config.vocab_size - 1``. Users MUST reserve
    this id as the ``[MASK]`` token in their vocabulary (it must not be a real
    token), so that masking does not collide with genuine vocabulary entries.

    This module owns no learnable parameters; it is an :class:`torch.nn.Module`
    only for uniformity with :class:`MaskedImageModeling` and so the network can
    hold it as a submodule.

    Args:
        config: The validated SHADA configuration describing the model shape.
        mask_token_id: Vocabulary id used as the ``[MASK]`` token. Defaults to
            ``config.vocab_size - 1`` (a reserved id the user must set aside).
    """

    def __init__(self, config: SHADAConfig, mask_token_id: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.text_mask_ratio = config.text_mask_ratio
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = (
            config.vocab_size - 1 if mask_token_id is None else mask_token_id
        )

    def forward(
        self, encoder: nn.Module, lm_head: nn.Module, token_ids: Tensor
    ) -> Dict[str, Tensor]:
        """Run the masked language modeling objective.

        Args:
            encoder: The hierarchical encoder (injected). Called as
                ``encoder(masked_ids, modality="text")``; its ``"tokens"`` entry
                of shape ``(B, L, final_dim)`` is consumed.
            lm_head: A module mapping ``(B, L, final_dim)`` token features to
                ``(B, L, vocab_size)`` logits (injected).
            token_ids: Long tensor of token ids, shape ``(B, L)``.

        Returns:
            A dict with keys ``"loss"`` (scalar cross-entropy over masked
            positions), ``"logits"`` ``(B, L, vocab_size)`` and ``"mask"`` (the
            boolean mask ``(B, L)`` of masked positions).
        """
        # 1. Choose mask positions: Bernoulli(text_mask_ratio), never on padding.
        non_pad = token_ids != self.pad_token_id
        probs = torch.full_like(token_ids, self.text_mask_ratio, dtype=torch.float)
        mask = (torch.bernoulli(probs).bool()) & non_pad

        # Ensure at least one masked position so the loss is well-defined.
        if not bool(mask.any()):
            flat_non_pad = non_pad.flatten()
            candidates = torch.nonzero(flat_non_pad, as_tuple=False).flatten()
            if candidates.numel() > 0:
                choice = candidates[
                    torch.randint(candidates.numel(), (1,), device=token_ids.device)
                ]
                flat_mask = mask.flatten()
                flat_mask[choice] = True
                mask = flat_mask.reshape_as(token_ids)

        # 2. Replace masked positions with the [MASK] id.
        masked_ids = token_ids.clone()
        masked_ids[mask] = self.mask_token_id

        # 3. Encode and project to vocabulary logits.
        enc = encoder(masked_ids, modality="text")
        logits = lm_head(enc["tokens"])  # (B, L, vocab_size)

        # 4. Cross-entropy over masked positions only (predict original ids).
        targets = token_ids.clone()
        targets[~mask] = -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": logits, "mask": mask}


def compute_ssl_loss(
    config: SHADAConfig,
    encoder: nn.Module,
    x: Tensor,
    *,
    lm_head: Optional[nn.Module] = None,
) -> Dict[str, Tensor]:
    """One-off self-supervised loss dispatcher.

    Infers the modality from ``x`` and runs the matching objective:

    * a floating-point tensor with 4 dims ``(B, C, H, W)`` -> image
      (:class:`MaskedImageModeling`);
    * a long/int tensor with 2 dims ``(B, L)`` -> text
      (:class:`MaskedLanguageModeling`, which requires ``lm_head``).

    .. warning::
        This helper *constructs a fresh objective on every call*, so the image
        decoder's weights are discarded after each call. It is therefore only
        suitable for one-off / stateless use (quick experiments, tests). For
        training, the high-level network should hold persistent instances of
        :class:`MaskedImageModeling` / :class:`MaskedLanguageModeling` (both are
        exported for exactly this purpose) so the decoder is trained.

    Args:
        config: The validated SHADA configuration describing the model shape.
        encoder: The hierarchical encoder (injected).
        x: Either an image batch ``(B, C, H, W)`` (floating) or a token-id batch
            ``(B, L)`` (long/int).
        lm_head: Required for the text modality; maps ``(B, L, final_dim)`` to
            ``(B, L, vocab_size)``. Ignored for images.

    Returns:
        The dict returned by the selected objective's ``forward`` (see
        :class:`MaskedImageModeling` / :class:`MaskedLanguageModeling`).

    Raises:
        ValueError: if ``x`` is neither a 4-d floating image tensor nor a 2-d
            integer token tensor, or if the text modality is selected without an
            ``lm_head``.
    """
    if x.dim() == 4 and torch.is_floating_point(x):
        objective = MaskedImageModeling(config).to(x.device)
        return objective(encoder, x)
    if x.dim() == 2 and not torch.is_floating_point(x):
        if lm_head is None:
            raise ValueError(
                "compute_ssl_loss requires `lm_head` for the text modality "
                "(2-d integer token-id input)"
            )
        objective = MaskedLanguageModeling(config).to(x.device)
        return objective(encoder, lm_head, x)
    raise ValueError(
        "could not infer modality from x: expected a 4-d floating image tensor "
        f"(B, C, H, W) or a 2-d integer token tensor (B, L), got dim={x.dim()} "
        f"dtype={x.dtype}"
    )
