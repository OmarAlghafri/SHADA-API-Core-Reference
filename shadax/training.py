"""
SHADA losses and the four-phase training pipeline.

This module supplies the supervised task losses (:func:`compute_task_loss`) and
the optimisation driver (:func:`run_pipeline`) that walks a :class:`SHADANet`
through any sequence of :class:`~shadax.config.TrainingPhase` s:

* :class:`~shadax.config.TrainingPhase.PRETRAIN`  -- self-supervised only
  (labels are not required).
* :class:`~shadax.config.TrainingPhase.MULTITASK` -- supervised loss plus a
  weighted self-supervised loss (labels required).
* :class:`~shadax.config.TrainingPhase.FINETUNE`  -- supervised loss only
  (labels required).
* :class:`~shadax.config.TrainingPhase.DEPLOY`    -- evaluation mode, no
  optimisation is performed.

Detection target format
------------------------
The detection task uses the dense CenterNet-style target layout described on
:func:`compute_task_loss`: a dict of tensors on the stride-32 grid
``(Hs, Ws) = (H / 32, W / 32)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shadax.config import SHADAConfig, TaskType, TrainingPhase

__all__ = ["compute_task_loss", "TrainingHistory", "run_pipeline"]


# --------------------------------------------------------------------------- #
# Supervised task losses.
# --------------------------------------------------------------------------- #
def compute_task_loss(
    config: SHADAConfig,
    outputs: Dict[str, Tensor],
    x: Tensor,
    y: Any,
) -> Tensor:
    """Compute the supervised loss for ``config.task_type``.

    Args:
        config: The validated SHADA configuration (selects the loss and supplies
            ``pad_token_id`` for the language-model loss).
        outputs: The dict returned by :meth:`shadax.network.SHADANet.forward`.
        x: The model input. Only the language-model loss uses it (the input
            token ids ``(B, L)`` are the next-token targets).
        y: The supervised targets, interpreted per task (see below). May be
            ``None`` for the language-model task.

    Returns:
        A scalar loss tensor.

    Task-specific behaviour:
        * **CLASSIFICATION** -- ``y`` is ``(B,)`` integer class labels (one-hot
          ``(B, num_classes)`` is accepted and argmaxed first). Cross-entropy on
          ``outputs["logits"]`` ``(B, num_classes)``.
        * **LANGUAGE_MODEL** -- next-token cross-entropy: position ``t`` of
          ``outputs["logits"]`` ``(B, L, V)`` predicts input token ``t + 1`` of
          ``x`` ``(B, L)``. ``config.pad_token_id`` positions are ignored. ``y``
          is unused.
        * **SEGMENTATION** -- ``y`` is ``(B, H, W)`` integer class indices.
          Per-pixel cross-entropy on ``outputs["segmentation"]``
          ``(B, num_classes, H, W)``.
        * **DETECTION** -- ``y`` is a dict of dense targets on the stride-32 grid
          ``(Hs, Ws)``:

          ============  ===================  =====================================
          key           shape                meaning
          ============  ===================  =====================================
          ``heatmap``   ``(B, C, Hs, Ws)``   per-class center heatmap, float [0, 1]
          ``wh``        ``(B, 2, Hs, Ws)``   box width/height at each location
          ``offset``    ``(B, 2, Hs, Ws)``   sub-pixel center offset
          ``reg_mask``  ``(B, 1, Hs, Ws)``   1 where a center exists, else 0
          ============  ===================  =====================================

          Loss = focal-free BCE-with-logits on the heatmap plus L1 on ``wh`` and
          ``offset`` taken only at ``reg_mask`` locations and normalised by
          ``reg_mask.sum() + 1e-6``.

    Raises:
        ValueError: for an unsupported ``config.task_type``.
    """
    task = config.task_type

    if task is TaskType.CLASSIFICATION:
        logits = outputs["logits"]
        target = _as_tensor(y, device=logits.device)
        if target.dim() > 1 and target.shape[-1] == config.num_classes:
            target = target.argmax(dim=-1)
        return F.cross_entropy(logits, target.long())

    if task is TaskType.LANGUAGE_MODEL:
        logits = outputs["logits"]  # (B, L, V)
        vocab = logits.size(-1)
        ids = _as_tensor(x, device=logits.device).long()  # (B, L)
        # Shift: predict token t+1 from position t.
        pred = logits[:, :-1].reshape(-1, vocab)
        tgt = ids[:, 1:].reshape(-1)
        return F.cross_entropy(pred, tgt, ignore_index=config.pad_token_id)

    if task is TaskType.SEGMENTATION:
        seg = outputs["segmentation"]  # (B, C, H, W)
        target = _as_tensor(y, device=seg.device).long()  # (B, H, W)
        return F.cross_entropy(seg, target)

    if task is TaskType.DETECTION:
        if not isinstance(y, dict):
            raise ValueError(
                "detection targets must be a dict with keys 'heatmap', 'wh', "
                "'offset', 'reg_mask'"
            )
        device = outputs["heatmap"].device
        hm_t = _as_tensor(y["heatmap"], device=device).float()
        wh_t = _as_tensor(y["wh"], device=device).float()
        off_t = _as_tensor(y["offset"], device=device).float()
        reg_mask = _as_tensor(y["reg_mask"], device=device).float()

        hm_loss = F.binary_cross_entropy_with_logits(outputs["heatmap"], hm_t)
        denom = reg_mask.sum() + 1e-6
        wh_loss = F.l1_loss(
            outputs["wh"] * reg_mask, wh_t * reg_mask, reduction="sum"
        ) / denom
        off_loss = F.l1_loss(
            outputs["offset"] * reg_mask, off_t * reg_mask, reduction="sum"
        ) / denom
        return hm_loss + wh_loss + off_loss

    raise ValueError(f"unsupported task type: {task!r}")


def _as_tensor(value: Any, device: torch.device) -> Tensor:
    """Coerce ``value`` (tensor / ndarray / scalar) to a tensor on ``device``."""
    if isinstance(value, Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


# --------------------------------------------------------------------------- #
# Training history.
# --------------------------------------------------------------------------- #
@dataclass
class TrainingHistory:
    """Per-phase record of the average epoch losses seen during training.

    Attributes:
        phases: The phase names (in run order) that produced losses.
        losses: For each entry of ``phases``, the list of per-epoch average
            losses recorded during that phase.
    """

    phases: List[str] = field(default_factory=list)
    losses: List[List[float]] = field(default_factory=list)

    def add_phase(self, phase: str) -> None:
        """Start recording a new phase."""
        self.phases.append(phase)
        self.losses.append([])

    def record(self, loss: float) -> None:
        """Append an average epoch loss to the current (most recent) phase."""
        self.losses[-1].append(loss)

    def to_dict(self) -> Dict[str, List]:
        """Return the history as a plain dict (JSON-friendly)."""
        return {"phases": list(self.phases), "losses": [list(p) for p in self.losses]}


# --------------------------------------------------------------------------- #
# Batch indexing helpers.
# --------------------------------------------------------------------------- #
def _index(data: Any, idx: Tensor, device: torch.device) -> Any:
    """Index ``data`` by the batch indices ``idx`` and move it to ``device``.

    Handles the heterogeneous label/input containers used across tasks:

    * ``None`` -> ``None`` (unsupervised phases).
    * :class:`torch.Tensor` -> indexed and moved to ``device``.
    * :class:`numpy.ndarray` -> indexed, converted to a tensor and moved.
    * ``dict`` of the above (detection targets) -> each value indexed.

    Args:
        data: The container to index (``None`` / tensor / ndarray / dict).
        idx: A 1-D long tensor of batch indices.
        device: The device the sliced data should live on.

    Returns:
        The indexed container in the same kind as ``data``.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: _index(v, idx, device) for k, v in data.items()}
    if isinstance(data, Tensor):
        return data[idx.to(data.device)].to(device)
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data[idx.cpu().numpy()]).to(device)
    # Fallback: treat as a sequence.
    arr = torch.as_tensor(np.asarray(data))
    return arr[idx.cpu().numpy() if isinstance(idx, Tensor) else idx].to(device)


def _num_samples(X: Any) -> int:
    """Return the number of samples (first-dim length) of ``X``."""
    return int(X.shape[0])


# --------------------------------------------------------------------------- #
# The pipeline.
# --------------------------------------------------------------------------- #
def run_pipeline(
    net: nn.Module,
    X: Union[np.ndarray, Tensor],
    y: Any = None,
    *,
    phases: Sequence[TrainingPhase],
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: Union[str, torch.device],
    ssl_weight: float = 1.0,
    verbose: bool = True,
) -> Dict[str, List]:
    """Run a :class:`SHADANet` through a sequence of training phases.

    Each phase is optimised independently with a fresh
    :class:`torch.optim.AdamW` optimiser and a
    :class:`torch.optim.lr_scheduler.CosineAnnealingLR` schedule
    (``T_max = epochs``). Within a phase, every epoch iterates over shuffled
    minibatches of ``X`` (and the matching slice of ``y``), accumulating:

    * ``PRETRAIN``  -> ``net.ssl_loss(xb)["loss"]`` (labels not required);
    * ``MULTITASK`` -> ``task_loss + ssl_weight * net.ssl_loss(xb)["loss"]``;
    * ``FINETUNE``  -> ``task_loss``.

    Gradients are clipped to a max-norm of ``1.0``; the scheduler steps once per
    epoch. A :class:`~shadax.config.TrainingPhase.DEPLOY` phase only switches the
    network to ``eval`` mode and performs no optimisation.

    Args:
        net: The :class:`~shadax.network.SHADANet` to train (already built).
        X: Training inputs ``(N, ...)`` as a numpy array or a tensor.
        y: Supervised targets (``None`` / tensor / ndarray / dict of tensors for
            detection). Required for ``MULTITASK`` / ``FINETUNE``.
        phases: The ordered list of phases to run.
        epochs: Number of epochs per optimised phase.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Minibatch size.
        device: Device to train on.
        ssl_weight: Weight of the SSL term in the ``MULTITASK`` phase.
        verbose: Whether to print per-phase / per-epoch progress.

    Returns:
        A history dict ``{"phases": [...], "losses": [[...], ...]}`` recording
        the average loss of each epoch of each optimised phase.

    Raises:
        ValueError: if ``MULTITASK`` / ``FINETUNE`` is requested without ``y``.
    """
    device = torch.device(device)
    net.to(device)
    config: SHADAConfig = net.config
    history = TrainingHistory()
    n_samples = _num_samples(X)

    for phase in phases:
        phase = TrainingPhase(phase)

        if phase is TrainingPhase.DEPLOY:
            net.eval()
            if verbose:
                print(f"[{phase.value}] eval mode (no optimisation).")
            continue

        # The language-model task derives its supervised next-token targets from
        # the input ``x`` itself, so it never needs an explicit ``y``. Every
        # other task does for the optimised supervised phases.
        needs_labels = config.task_type is not TaskType.LANGUAGE_MODEL
        if (
            phase in (TrainingPhase.MULTITASK, TrainingPhase.FINETUNE)
            and needs_labels
            and y is None
        ):
            raise ValueError(
                f"phase {phase.value!r} requires labels `y`, but y is None"
            )

        net.train()
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        history.add_phase(phase.value)

        if verbose:
            print(f"[{phase.value}] {epochs} epoch(s), {n_samples} samples.")

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            running = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                xb = _index(X, idx, device)
                optimizer.zero_grad()

                if phase is TrainingPhase.PRETRAIN:
                    loss = net.ssl_loss(xb)["loss"]
                elif phase is TrainingPhase.FINETUNE:
                    yb = _index(y, idx, device)
                    outputs = net(xb)
                    loss = compute_task_loss(config, outputs, xb, yb)
                else:  # MULTITASK
                    yb = _index(y, idx, device)
                    outputs = net(xb)
                    task_loss = compute_task_loss(config, outputs, xb, yb)
                    ssl = net.ssl_loss(xb)["loss"]
                    loss = task_loss + ssl_weight * ssl

                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                running += float(loss.item())
                n_batches += 1

            scheduler.step()
            avg = running / max(1, n_batches)
            history.record(avg)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs} - loss {avg:.4f}")

    return history.to_dict()
