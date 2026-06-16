"""
SHADA high-level, sklearn-style estimator.

This module defines :class:`SHADA`, the user-facing wrapper that hides the
encoder / head / SSL machinery behind a familiar ``fit`` / ``predict`` /
``score`` interface. It supports all four tasks (classification, segmentation,
language modeling, detection), self-supervised pretraining, the full four-phase
pipeline, feature extraction and checkpoint round-tripping.

Example:
    >>> import numpy as np
    >>> from shadax import SHADA
    >>> X = np.random.randn(64, 3, 64, 64).astype("float32")
    >>> y = np.random.randint(0, 10, 64)
    >>> model = SHADA("nano", task="classification", num_classes=10)
    >>> _ = model.fit(X, y, epochs=2)
    >>> preds = model.predict(X)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from shadax.config import (
    SHADAConfig,
    TaskType,
    TrainingPhase,
    create_config,
)
from shadax.network import SHADANet
from shadax.training import run_pipeline

__all__ = ["SHADA"]


class SHADA:
    """SHADA - Self-supervised Hierarchical Adaptive Hybrid Algorithm.

    A unified architecture combining:

    * a hierarchical encoder (convolutional stem -> Transformer stages),
    * multi-modal support (image and text share the same backbone),
    * self-supervised learning objectives, and
    * a four-phase training pipeline (pretrain / multitask / finetune / deploy).

    The estimator follows the sklearn ``fit`` / ``predict`` convention.

    Args:
        tier: Model size tier (``nano``/``base``/``large``/``xl``), **or** a
            ready-made :class:`~shadax.config.SHADAConfig` instance (in which
            case ``task`` / ``num_classes`` / ``**kwargs`` are read from it).
        num_classes: Number of output classes (ignored when ``tier`` is a
            config).
        task: Task type (``classification``/``detection``/``segmentation``/``lm``).
        learning_rate: Optimiser learning rate.
        weight_decay: Optimiser weight decay.
        epochs: Default number of training epochs per phase.
        batch_size: Minibatch size.
        device: Device string (``"cpu"``, ``"cuda"``, ...). Auto-selected when
            ``None``.
        phases: Optional default list of :class:`~shadax.config.TrainingPhase`;
            resolved at :meth:`fit` time when ``None``.
        **kwargs: Extra :class:`~shadax.config.SHADAConfig` field overrides.

    Example:
        >>> from shadax import SHADA, create_config
        >>> model = SHADA(create_config("base", num_classes=10))
        >>> model.fit(X_train, y_train)            # doctest: +SKIP
        >>> predictions = model.predict(X_test)    # doctest: +SKIP
    """

    def __init__(
        self,
        tier: Union[str, SHADAConfig] = "base",
        num_classes: int = 1000,
        task: str = "classification",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        batch_size: int = 64,
        device: Optional[str] = None,
        phases: Optional[List[TrainingPhase]] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(tier, SHADAConfig):
            # Accept a ready-made config (the documented ``SHADA(config)`` form).
            self.config = tier
            num_classes = self.config.num_classes
        else:
            self.config = create_config(
                tier, task=task, num_classes=num_classes, **kwargs
            )

        self.num_classes = self.config.num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.phases = phases

        self._model: Optional[SHADANet] = None
        self._is_fitted = False
        self._classes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Introspection.
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    @property
    def _task(self) -> TaskType:
        return self.config.task_type

    # ------------------------------------------------------------------ #
    # Fitting.
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: Union[np.ndarray, Any],
        y: Optional[Union[np.ndarray, Any]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        epochs: Optional[int] = None,
        phases: Optional[List[TrainingPhase]] = None,
    ) -> "SHADA":
        """Fit the SHADA model to training data.

        Args:
            X: Training inputs.

                * Images: ``(N, C, H, W)`` or ``(N, H, W, C)`` (channels-last is
                  permuted automatically); ``(N, H, W)`` grayscale gets a channel
                  axis. ``H`` and ``W`` must be divisible by 32.
                * Text / language model: ``(N, L)`` integer token ids.
            y: Training targets, optional (``None`` triggers self-supervised
                pretraining):

                * Classification: ``(N,)`` integer labels (or one-hot
                  ``(N, num_classes)``).
                * Segmentation: ``(N, H, W)`` integer masks.
                * Detection: a dict of dense targets (see
                  :func:`shadax.training.compute_task_loss`).
                * Language model: ignored (``None``).
            eval_set: Reserved for a future held-out evaluation hook; unused.
            verbose: Whether to print training progress.
            epochs: Overrides ``self.epochs`` for this call when given. (Fixes
                the previously-broken ``model.fit(X, y, epochs=...)`` call.)
            phases: Explicit list of :class:`~shadax.config.TrainingPhase`.
                Resolution order: this argument > ``self.phases`` > a default
                (``[PRETRAIN]`` when ``y is None``, else ``[FINETUNE]``).

        Returns:
            ``self`` (the fitted estimator).
        """
        epochs = self.epochs if epochs is None else epochs

        # Resolve phases: explicit arg > instance default > task default.
        if phases is None:
            phases = self.phases
        if phases is None:
            phases = (
                [TrainingPhase.PRETRAIN] if y is None else [TrainingPhase.FINETUNE]
            )

        device = self._get_device()
        X_prepared = self._prepare_input(X)
        y_prepared = self._prepare_targets(y)

        if self._task is TaskType.CLASSIFICATION and y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim > 1:
                y_arr = np.argmax(y_arr, axis=1)
            self._classes = np.unique(y_arr)

        if verbose:
            shape = tuple(X_prepared.shape)
            print(f"SHADA fit (tier={self.config.tier}, task={self.config.task})")
            print(f"  Input shape: {shape}")
            print(f"  Phases: {[TrainingPhase(p).value for p in phases]}")
            print(f"  Epochs: {epochs}  Batch size: {self.batch_size}")
            print(f"  Device: {device}")

        self._model = SHADANet(self.config).to(device)

        run_pipeline(
            self._model,
            X_prepared,
            y_prepared,
            phases=phases,
            epochs=epochs,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            device=device,
            verbose=verbose,
        )

        self._is_fitted = True
        if verbose:
            print("Training complete.")
        return self

    def pretrain(
        self,
        X: Union[np.ndarray, Any],
        epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> "SHADA":
        """Self-supervised pretraining convenience wrapper.

        Equivalent to ``fit(X, y=None, phases=[TrainingPhase.PRETRAIN], ...)``.

        Args:
            X: Training inputs (images or token ids; see :meth:`fit`).
            epochs: Overrides ``self.epochs`` when given.
            verbose: Whether to print progress.

        Returns:
            ``self`` (the pretrained estimator).
        """
        return self.fit(
            X,
            y=None,
            phases=[TrainingPhase.PRETRAIN],
            epochs=epochs,
            verbose=verbose,
        )

    # ------------------------------------------------------------------ #
    # Prediction.
    # ------------------------------------------------------------------ #
    def predict(
        self,
        X: Union[np.ndarray, Any],
        return_probs: bool = False,
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        """Predict targets for ``X``.

        Behaviour per task:

        * **CLASSIFICATION** -> ``(N,)`` argmax labels. With ``return_probs`` a
          ``(labels, probs)`` tuple where ``probs`` is ``(N, num_classes)``.
        * **SEGMENTATION** -> ``(N, H, W)`` per-pixel argmax labels.
        * **LANGUAGE_MODEL** -> ``(N, L)`` per-position argmax token ids.
        * **DETECTION** -> a length-``N`` list of dicts (see
          :meth:`_decode_detections`); ``return_probs`` is ignored.

        Args:
            X: Input samples (see :meth:`fit`).
            return_probs: Classification only -- also return probabilities.

        Returns:
            The predictions in the task-specific layout described above.
        """
        outputs = self._forward_eval(X)
        task = self._task

        if task is TaskType.CLASSIFICATION:
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
            if return_probs:
                return preds, probs.cpu().numpy()
            return preds

        if task is TaskType.SEGMENTATION:
            return outputs["segmentation"].argmax(dim=1).cpu().numpy()

        if task is TaskType.LANGUAGE_MODEL:
            return outputs["logits"].argmax(dim=-1).cpu().numpy()

        if task is TaskType.DETECTION:
            return self._decode_detections(outputs)

        raise ValueError(f"unsupported task type: {task!r}")  # pragma: no cover

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Return per-sample / per-position probabilities.

        Args:
            X: Input samples (see :meth:`fit`).

        Returns:
            * classification -> ``(N, num_classes)``;
            * segmentation -> ``(N, num_classes, H, W)``;
            * language model -> ``(N, L, vocab_size)``.

        Raises:
            NotImplementedError: for the detection task (use :meth:`predict`).
        """
        outputs = self._forward_eval(X)
        task = self._task

        if task is TaskType.CLASSIFICATION:
            return torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
        if task is TaskType.SEGMENTATION:
            return torch.softmax(outputs["segmentation"], dim=1).cpu().numpy()
        if task is TaskType.LANGUAGE_MODEL:
            return torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
        raise NotImplementedError(
            "predict_proba is not defined for the detection task; use predict()."
        )

    def score(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> float:
        """Return a task-appropriate accuracy score in ``[0, 1]``.

        Args:
            X: Test samples.
            y: Ground-truth targets (matching the task; see :meth:`fit`).

        Returns:
            * classification -> top-1 accuracy;
            * segmentation -> mean per-pixel accuracy;
            * language model -> next-token accuracy (excluding ``pad_token_id``).

        Raises:
            NotImplementedError: for detection (use a dedicated mAP metric, e.g.
                ``torchmetrics.detection`` or ``pycocotools``).
        """
        task = self._task
        preds = self.predict(X)
        y_arr = np.asarray(y)

        if task is TaskType.CLASSIFICATION:
            if y_arr.ndim > 1:
                y_arr = np.argmax(y_arr, axis=1)
            return float(np.mean(np.asarray(preds) == y_arr))

        if task is TaskType.SEGMENTATION:
            return float(np.mean(np.asarray(preds) == y_arr))

        if task is TaskType.LANGUAGE_MODEL:
            # Next-token accuracy: predictions at t correspond to input t+1.
            preds = np.asarray(preds)
            pred_next = preds[:, :-1]
            target_next = y_arr[:, 1:]
            valid = target_next != self.config.pad_token_id
            if valid.sum() == 0:
                return 0.0
            return float(np.mean((pred_next == target_next)[valid]))

        raise NotImplementedError(
            "score() is not defined for the detection task; use a detection "
            "metrics library (e.g. torchmetrics.detection.MeanAveragePrecision "
            "or pycocotools)."
        )

    # ------------------------------------------------------------------ #
    # Feature extraction.
    # ------------------------------------------------------------------ #
    def extract_features(
        self,
        X: Union[np.ndarray, Any],
        layer: str = "global",
    ) -> np.ndarray:
        """Extract encoder feature representations.

        Args:
            X: Input samples (see :meth:`fit`).
            layer: Which representation to return:

                * ``"global"``  -> pooled global features ``(N, final_dim)``;
                * ``"tokens"``  -> final token sequence ``(N, N_tok, final_dim)``;
                * ``"spatial"`` -> last image feature map
                  ``(N, final_dim, Hs, Ws)`` (image modality only).

        Returns:
            The requested features as a numpy array.

        Raises:
            RuntimeError: if the model has not been fitted.
            ValueError: for an unknown ``layer``.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before extracting features.")
        if layer not in {"global", "tokens", "spatial"}:
            raise ValueError(
                f"layer must be one of 'global'/'tokens'/'spatial', got {layer!r}"
            )

        self._model.eval()
        device = next(self._model.parameters()).device
        x = self._prepare_input(X).to(device)
        with torch.no_grad():
            enc = self._model.encode(x)
            if layer == "global":
                feats = enc["global_features"]
            elif layer == "tokens":
                feats = enc["tokens"]
            else:  # spatial
                feats = enc["feature_maps"][-1]
        return feats.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Persistence.
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save the fitted model to ``path``.

        Stores the config, the network ``state_dict``, the observed classes and
        the scalar hyper-parameters.

        Args:
            path: Destination file path (``.pt`` / ``.pth``).

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before saving.")
        checkpoint = {
            "config": self.config,
            "state_dict": self._model.state_dict(),
            "classes": self._classes,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> "SHADA":
        """Load a model previously written by :meth:`save`.

        Rebuilds the :class:`~shadax.network.SHADANet` from the stored config and
        restores its weights and hyper-parameters.

        Args:
            path: Source file path.

        Returns:
            ``self`` (for chaining).
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self._model = SHADANet(self.config)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._classes = checkpoint.get("classes")
        self.num_classes = checkpoint.get("num_classes", self.config.num_classes)
        self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
        self.weight_decay = checkpoint.get("weight_decay", self.weight_decay)
        self.epochs = checkpoint.get("epochs", self.epochs)
        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Internal helpers.
    # ------------------------------------------------------------------ #
    def _forward_eval(self, X: Union[np.ndarray, Any]) -> Dict[str, torch.Tensor]:
        """Run a no-grad forward pass and return the raw output dict."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        self._model.eval()
        device = next(self._model.parameters()).device
        x = self._prepare_input(X).to(device)
        with torch.no_grad():
            return self._model(x)

    def _decode_detections(
        self,
        outputs: Dict[str, torch.Tensor],
        top_k: int = 20,
    ) -> List[Dict[str, np.ndarray]]:
        """Decode CenterNet-style dense outputs into per-image detections.

        The decode follows the standard CenterNet inference recipe:

        1. ``sigmoid`` the heatmap to get per-class confidences;
        2. keep only local maxima via a ``3x3`` max-pool equality test (cheap
           NMS);
        3. take the global top-``k`` peaks across all classes;
        4. read the box size (``wh``) and sub-pixel ``offset`` at each peak and
           assemble ``cx, cy, w, h`` boxes in stride-32 grid coordinates.

        Args:
            outputs: The detection head output dict (``heatmap``/``wh``/``offset``).
            top_k: Maximum number of detections kept per image.

        Returns:
            A length-``B`` list of dicts, each with arrays
            ``{"boxes": (k, 4), "scores": (k,), "labels": (k,)}`` where boxes are
            ``(cx, cy, w, h)``.
        """
        heatmap = torch.sigmoid(outputs["heatmap"])  # (B, C, Hs, Ws)
        wh = outputs["wh"]                            # (B, 2, Hs, Ws)
        offset = outputs["offset"]                    # (B, 2, Hs, Ws)
        b, c, hs, ws = heatmap.shape

        # Local-maxima keep mask (3x3 NMS).
        pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        keep = (pooled == heatmap).float()
        scores_map = heatmap * keep  # (B, C, Hs, Ws)

        k = min(top_k, c * hs * ws)
        flat = scores_map.reshape(b, -1)              # (B, C*Hs*Ws)
        top_scores, top_idx = torch.topk(flat, k, dim=1)

        # Decode flat indices -> (class, y, x).
        labels = (top_idx // (hs * ws))
        rem = top_idx % (hs * ws)
        ys = rem // ws
        xs = rem % ws

        wh_flat = wh.reshape(b, 2, -1)                # (B, 2, Hs*Ws)
        off_flat = offset.reshape(b, 2, -1)

        results: List[Dict[str, np.ndarray]] = []
        for i in range(b):
            sel = rem[i]                              # (k,) spatial indices
            w_i = wh_flat[i, 0, sel]
            h_i = wh_flat[i, 1, sel]
            off_x = off_flat[i, 0, sel]
            off_y = off_flat[i, 1, sel]
            cx = xs[i].float() + off_x
            cy = ys[i].float() + off_y
            boxes = torch.stack([cx, cy, w_i, h_i], dim=1)  # (k, 4)
            results.append(
                {
                    "boxes": boxes.cpu().numpy(),
                    "scores": top_scores[i].cpu().numpy(),
                    "labels": labels[i].cpu().numpy(),
                }
            )
        return results

    def _prepare_input(self, X: Union[np.ndarray, Any]) -> torch.Tensor:
        """Convert ``X`` to the tensor layout the network expects.

        Text / language-model inputs (2-D) are kept as long token ids. Image
        inputs are floated and reshaped to ``(N, C, H, W)``: a 3-D grayscale
        batch gets a channel axis, and a channels-last ``(N, H, W, C)`` batch
        (last dim <= 4) is permuted to channels-first.

        Args:
            X: Raw input (numpy array or tensor).

        Returns:
            A tensor ready for :meth:`SHADANet.forward`.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        elif not isinstance(X, torch.Tensor) and hasattr(X, "numpy"):
            X = torch.from_numpy(X.numpy())
        elif not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        # Text / language model: keep integer token ids (B, L) untouched.
        if self._task is TaskType.LANGUAGE_MODEL or (
            X.dim() == 2 and not torch.is_floating_point(X)
        ):
            return X.long()

        X = X.float()
        if X.dim() == 3:
            X = X.unsqueeze(1)
        elif X.dim() == 4 and X.shape[-1] <= 4 and X.shape[1] > 4:
            # Channels-last (N, H, W, C) -> channels-first.
            X = X.permute(0, 3, 1, 2).contiguous()
        return X

    def _prepare_targets(self, y: Any) -> Any:
        """Convert supervised targets ``y`` to tensors (dtype per task).

        Args:
            y: ``None`` / ndarray / tensor / dict-of-tensors (detection).

        Returns:
            ``None`` if ``y is None``; otherwise tensors with the per-task dtype
            (long for class/segmentation labels, float for detection maps).
        """
        if y is None:
            return None

        if self._task is TaskType.DETECTION:
            if not isinstance(y, dict):
                raise ValueError(
                    "detection targets must be a dict (see compute_task_loss)."
                )
            return {k: torch.as_tensor(np.asarray(v)).float() for k, v in y.items()}

        arr = np.asarray(y)
        if self._task is TaskType.CLASSIFICATION and arr.ndim > 1:
            arr = np.argmax(arr, axis=1)
        return torch.as_tensor(arr).long()

    def _get_device(self) -> torch.device:
        """Resolve the computation device (explicit or auto-selected)."""
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self) -> str:
        return (
            f"SHADA(tier={self.config.tier!r}, task={self.config.task!r}, "
            f"num_classes={self.num_classes}, fitted={self._is_fitted})"
        )
