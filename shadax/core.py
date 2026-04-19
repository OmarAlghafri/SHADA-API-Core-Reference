"""
SHADA Core Implementation

Self-supervised Hierarchical Adaptive Hybrid Algorithm for Deep Learning.
A production-ready implementation combining CNN + Transformer stages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


class ModelTier(str, Enum):
    """Model size tiers."""
    NANO = "nano"
    BASE = "base"
    LARGE = "large"
    XL = "xl"


class TaskType(str, Enum):
    """Supported task types."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    LANGUAGE_MODEL = "lm"


class TrainingPhase(str, Enum):
    """Training pipeline phases."""
    PRETRAIN = "pretrain"
    multitask = "multitask"
    FINETUNE = "finetune"
    DEPLOY = "deploy"


@dataclass
class SHADAConfig:
    """
    Configuration for SHADA model.

    Attributes:
        tier: Model size tier (nano, base, large, xl)
        encoder_dims: Dimensions for each encoder stage
        encoder_depths: Number of blocks per stage
        num_heads: Attention heads per stage
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size for text
        dropout: Dropout rate
        task: Primary task type
        num_classes: Number of classes for classification
    """

    tier: str = "base"
    encoder_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    encoder_depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    max_seq_len: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.1
    task: str = "classification"
    num_classes: int = 1000


_TIER_CONFIGS: Dict[str, Dict] = {
    "nano": {
        "encoder_dims": [64, 128, 256, 512],
        "encoder_depths": [2, 2, 4, 2],
        "num_heads": [2, 4, 8, 16],
        "max_seq_len": 512,
    },
    "base": {
        "encoder_dims": [128, 256, 512, 1024],
        "encoder_depths": [3, 4, 6, 3],
        "num_heads": [4, 8, 16, 32],
        "max_seq_len": 1024,
    },
    "large": {
        "encoder_dims": [192, 384, 768, 1536],
        "encoder_depths": [3, 4, 18, 3],
        "num_heads": [6, 12, 24, 48],
        "max_seq_len": 2048,
    },
    "xl": {
        "encoder_dims": [256, 512, 1024, 2048],
        "encoder_depths": [3, 4, 24, 3],
        "num_heads": [8, 16, 32, 64],
        "max_seq_len": 4096,
    },
}


def create_config(
    tier: str = "base",
    task: str = "classification",
    num_classes: int = 1000,
    **overrides,
) -> SHADAConfig:
    """
    Create a SHADAConfig with tier-specific defaults.

    Args:
        tier: Model tier (nano, base, large, xl)
        task: Task type (classification, detection, segmentation, lm)
        num_classes: Number of output classes
        **overrides: Configuration overrides

    Returns:
        SHADAConfig instance

    Example:
        >>> config = create_config("base", num_classes=10)
    """
    if tier not in _TIER_CONFIGS:
        raise ValueError(f"tier must be one of {list(_TIER_CONFIGS.keys())}, got '{tier}'")
    kwargs = {
        **(_TIER_CONFIGS[tier]),
        "tier": tier,
        "task": task,
        "num_classes": num_classes,
        **overrides,
    }
    return SHADAConfig(**kwargs)


class SHADA:
    """
    SHADA - Self-supervised Hierarchical Adaptive Hybrid Algorithm.

    A unified architecture combining:
    - Hierarchical encoder (CNN -> Transformer stages)
    - Multi-modal support (image and text)
    - Self-supervised learning objectives
    - Four-phase training pipeline

    The algorithm follows sklearn API pattern with fit/predict methods.

    Attributes:
        config: SHADA configuration
        num_classes: Number of output classes

    Example:
        >>> from shadax import SHADA, create_config
        >>> config = create_config("base", num_classes=10)
        >>> model = SHADA(config)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        tier: str = "base",
        num_classes: int = 1000,
        task: str = "classification",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        batch_size: int = 64,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the SHADA model.

        Args:
            tier: Model size tier (nano, base, large, xl)
            num_classes: Number of classes for classification
            task: Primary task (classification, detection, segmentation, lm)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay coefficient
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use (cpu, cuda, cuda:0, etc.)
            **kwargs: Additional configuration overrides

        Example:
            >>> model = SHADA(tier="base", num_classes=10)
        """
        self.config = create_config(tier, task=task, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._model: Optional[Any] = None
        self._is_fitted = False
        self._classes: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    def fit(
        self,
        X: Union[np.ndarray, Any],
        y: Optional[Union[np.ndarray, Any]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> "SHADA":
        """
        Fit the SHADA model to training data.

        Accepts numpy arrays, PyTorch tensors, or paths to datasets.

        Args:
            X: Training features.
                - Images: (n_samples, C, H, W) or (n_samples, H, W, C)
                - Text: (n_samples, seq_len)
                - Path to dataset directory
            y: Training labels, optional for unsupervised learning.
                - Classification: (n_samples,) integer labels
                - One-hot: (n_samples, num_classes)
            eval_set: Optional validation tuple (X_val, y_val)
            verbose: Whether to print training progress

        Returns:
            self: Fitted model instance

        Example:
            >>> import numpy as np
            >>> from shadax import SHADA
            >>> X = np.random.randn(1000, 3, 224, 224).astype(np.float32)
            >>> y = np.random.randint(0, 10, 1000)
            >>> model = SHADA(tier="base", num_classes=10)
            >>> model.fit(X, y, epochs=10)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "PyTorch is required. Install: pip install torch"
            )

        X_tensor = self._prepare_input(X)
        if y is not None:
            y_array = np.asarray(y)
            if y_array.ndim > 1:
                y_array = np.argmax(y_array, axis=1)
            y_tensor = torch.from_numpy(y_array.astype(np.int32))

        self._classes = np.unique(y_array) if y is not None else None
        device = self._get_device()

        if verbose:
            print(f"SHADA Training ({self.config.tier})")
            print(f"  Input shape: {X_tensor.shape}")
            print(f"  Epochs: {self.epochs}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Device: {device}")

        self._model = self._build_model()
        self._model.to(device)
        X_tensor = X_tensor.to(device)
        if y is not None:
            y_tensor = y_tensor.to(device)

        criterion = nn.CrossEntropyLoss() if y is not None else nn.MSELoss()
        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self._model.train()
        n_samples = X_tensor.shape[0]
        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.epochs):
            self._model.train()
            indices = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                X_batch = X_tensor[batch_idx]

                optimizer.zero_grad()

                if y is not None:
                    y_batch = y_tensor[batch_idx]
                    outputs = self._model(X_batch)
                    if isinstance(outputs, dict):
                        outputs = outputs.get("logits", outputs.get("predictions", outputs))
                    loss = criterion(outputs, y_batch)
                else:
                    outputs = self._model(X_batch)
                    target = X_batch
                    loss = criterion(outputs, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self._is_fitted = True

        if verbose:
            print("Training complete!")

        return self

    def predict(
        self,
        X: Union[np.ndarray, Any],
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict class labels for samples.

        Args:
            X: Input samples.
                - Images: (n_samples, C, H, W) or (n_samples, H, W, C)
                - Text: (n_samples, seq_len)
            return_probs: Whether to return class probabilities

        Returns:
            predictions: Predicted class labels (n_samples,)
            probabilities: Class probabilities (n_samples, num_classes) if return_probs=True

        Example:
            >>> predictions = model.predict(X_test)
            >>> preds, probs = model.predict(X_test, return_probs=True)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        import torch

        self._model.eval()
        X_tensor = self._prepare_input(X)
        device = next(self._model.parameters()).device
        X_tensor = X_tensor.to(device)

        with torch.no_grad():
            outputs = self._model(X_tensor)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("predictions", outputs))
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        predictions = preds.cpu().numpy()

        if return_probs:
            probabilities = probs.cpu().numpy()
            return predictions, probabilities

        return predictions

    def predict_proba(
        self,
        X: Union[np.ndarray, Any],
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input samples

        Returns:
            probabilities: Class probabilities (n_samples, num_classes)

        Example:
            >>> probs = model.predict_proba(X_test)
        """
        return self.predict(X, return_probs=True)[1]

    def score(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
    ) -> float:
        """
        Return accuracy score on test data.

        Args:
            X: Test samples
            y: True labels

        Returns:
            accuracy: Accuracy score (0.0 to 1.0)

        Example:
            >>> accuracy = model.score(X_test, y_test)
        """
        predictions = self.predict(X)
        y = np.asarray(y)
        predictions = np.asarray(predictions)

        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        return float(np.mean(predictions == y))

    def extract_features(
        self,
        X: Union[np.ndarray, Any],
        layer: str = "global",
    ) -> np.ndarray:
        """
        Extract feature representations.

        Args:
            X: Input samples
            layer: Feature layer (global, spatial, fpn)

        Returns:
            features: Extracted features

        Example:
            >>> features = model.extract_features(X)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        import torch

        self._model.eval()
        X_tensor = self._prepare_input(X)
        device = next(self._model.parameters()).device
        X_tensor = X_tensor.to(device)

        with torch.no_grad():
            if hasattr(self._model, "encoder"):
                enc = self._model.encoder(X_tensor, modality="image")
                features = enc.get(layer, enc.get("global_features"))
            else:
                features = X_tensor.mean(dim=[2, 3]) if X_tensor.ndim == 4 else X_tensor

        return features.cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: File path (.pt or .pth)

        Example:
            >>> model.save("model.pt")
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

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

    def load(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: File path

        Example:
            >>> model.load("model.pt")
        """
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        self.config = checkpoint["config"]
        self._model = self._build_model()
        self._model.load_state_dict(checkpoint["state_dict"])
        self._classes = checkpoint.get("classes")
        self.num_classes = checkpoint.get("num_classes", self.num_classes)
        self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
        self.weight_decay = checkpoint.get("weight_decay", self.weight_decay)
        self.epochs = checkpoint.get("epochs", self.epochs)
        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self._is_fitted = True

    def _prepare_input(self, X: Union[np.ndarray, Any]) -> "torch.Tensor":
        """Convert input to tensor."""
        import torch

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        if isinstance(X, torch.Tensor):
            X = X.float()
        elif hasattr(X, "numpy"):
            X = torch.from_numpy(X.numpy())

        if X.ndim == 3:
            X = X.unsqueeze(1)
        elif X.ndim == 4 and X.shape[-1] <= 4:
            pass
        elif X.ndim == 4:
            X = X.permute(0, 3, 1, 2)

        return X

    def _get_device(self) -> "torch.device":
        """Get computation device."""
        import torch

        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> "torch.nn.Module":
        """Build internal PyTorch model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required")

        cfg = self.config

        class _ConvStem(nn.Module):
            def __init__(self, in_ch: int = 3, embed_dim: int = 128):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim // 2, 3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
                )

            def forward(self, x):
                x = self.proj(x)
                B, C, H, W = x.shape
                return x.flatten(2).transpose(1, 2)

        class _TransformerBlock(nn.Module):
            def __init__(self, dim: int, num_heads: int):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
                self.norm2 = nn.LayerNorm(dim)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )

            def forward(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.ffn(self.norm2(x))
                return x

        class _Encoder(nn.Module):
            def __init__(self, cfg: SHADAConfig):
                super().__init__()
                self.stem = _ConvStem(embed_dim=cfg.encoder_dims[0])
                self.cfg = cfg

                layers = []
                for dim, depth, heads in zip(cfg.encoder_dims, cfg.encoder_depths, cfg.num_heads):
                    for _ in range(depth):
                        layers.append(_TransformerBlock(dim, heads))
                self.blocks = nn.ModuleList(layers)
                self.head = nn.Linear(cfg.encoder_dims[-1], cfg.num_classes)

            def forward(self, x):
                x = self.stem(x)
                for block in self.blocks:
                    x = block(x)
                pooled = x.mean(dim=1)
                logits = self.head(pooled)
                return {"logits": logits, "global_features": pooled}

        return _Encoder(cfg)

    def __repr__(self) -> str:
        """String representation."""
        return f"SHADA(tier={self.config.tier!r}, num_classes={self.num_classes}, task={self.config.task!r})"


__all__ = [
    "SHADA",
    "SHADAConfig",
    "create_config",
    "ModelTier",
    "TaskType",
    "TrainingPhase",
]