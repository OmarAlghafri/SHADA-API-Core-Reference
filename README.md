# shadax

SHADA - Self-supervised Hierarchical Adaptive Hybrid Algorithm for Deep Learning.

A production-ready Python implementation combining CNN + Transformer stages for multi-modal learning.

## Installation

```bash
pip install dist/*.whl
```

## Usage

```python
import numpy as np
from shadax import SHADA

# Create model
model = SHADA(tier="base", num_classes=10)

# Create sample data (batch_size, channels, height, width)
X_train = np.random.randn(1000, 3, 224, 224).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# Fit model
model.fit(X_train, y_train, epochs=10)

# Predict
predictions = model.predict(X_train[:10])
```

## API

- `SHADA`: Main model class with fit/predict interface
- `SHADAConfig`: Configuration dataclass
- `create_config`: Factory function for configurations