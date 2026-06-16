# shadax

**SHADA** — the **S**elf-supervised **H**ierarchical **A**daptive **H**ybrid **A**lgorithm — is a single
PyTorch architecture that pairs a convolutional stem with four hierarchical Transformer stages into one
**hybrid** backbone, then shares that backbone across **two modalities** (images and text) and **four
downstream tasks** (classification, segmentation, language modeling, detection). It is **adaptive** to
input resolution because the image path uses conditional (depthwise-convolutional) positional encoding
rather than a fixed positional table, and it is **self-supervised**: it ships with masked-image-modeling
(MAE-style) and masked-language-modeling (BERT-style) objectives and a four-phase training pipeline
(pretrain → multitask → finetune → deploy). Everything is exposed behind a familiar scikit-learn-style
estimator (`fit` / `predict` / `score`).

This README documents the library exactly as implemented. There are **no** bundled pretrained weights,
**no** bundled datasets, and **no** CUDA-only features — the model runs on CPU or GPU.

## Key features

- **Hybrid hierarchical backbone** — a `/4` convolutional stem followed by four Transformer stages, with
  `/2` patch-merging downsamples between consecutive stages (total spatial reduction **32**).
- **One shared multi-modal encoder** — the *same* `HierarchicalEncoder` processes images `(B, C, H, W)`
  and text token ids `(B, L)`; the Transformer blocks are shared between the two paths.
- **Resolution-adaptive** — images use conditional positional encoding (a residual depthwise conv), so
  there is no fixed positional table for images; any `H`, `W` divisible by 32 works.
- **Four model tiers** — `nano` / `base` / `large` / `xl`.
- **Four task heads** — classification, segmentation, language model (`lm`), detection (anchor-free
  CenterNet-style) — all really implemented.
- **Self-supervised objectives** — masked image modeling (`mask_ratio=0.75` default) and masked language
  modeling (`text_mask_ratio=0.15` default).
- **Four-phase training pipeline** — `pretrain`, `multitask`, `finetune`, `deploy`.
- **scikit-learn-style API** — `fit` / `predict` / `predict_proba` / `score` / `extract_features` /
  `save` / `load`.

## Installation

PyTorch is a **hard dependency** (`torch>=2.0`), along with `numpy`. PyTorch is not bundled — install the
build appropriate for your platform/accelerator (see <https://pytorch.org>) if `pip` does not resolve a
suitable wheel automatically.

### From PyPI (recommended)
```bash
pip install shadax
```

### From GitHub
```bash
pip install git+https://github.com/OmarAlghafri/SHADA-API-Core-Reference.git
```

### From a local wheel
```bash
pip install dist/*.whl
```

## Quickstart (image classification)

```python
import numpy as np
from shadax import SHADA

# Tiny synthetic dataset: 16 RGB images, 64x64 (H, W must be divisible by 32).
X = np.random.randn(16, 3, 64, 64).astype("float32")
y = np.random.randint(0, 10, size=16)

# Build a classification model (tier + number of classes).
model = SHADA(tier="nano", task="classification", num_classes=10)

# fit accepts epochs= directly. With y given, the default phase is [FINETUNE].
model.fit(X, y, epochs=2)

# predict -> (N,) integer labels; score -> top-1 accuracy in [0, 1].
preds = model.predict(X)
acc = model.score(X, y)
print(preds.shape, acc)
```

## Architecture overview

The backbone is a `HierarchicalEncoder` driven entirely by a `SHADAConfig`. It has two input paths that
**share their per-stage Transformer blocks** — both paths ultimately run the blocks over a token sequence
`(B, N, dim)`.

**Image path**

```
image (B, C, H, W)
   |  ConvStem                    -> (B, dims[0], H/4,  W/4)     [/4]
   v
 Stage 0  CPE + Transformer blocks (dims[0])
   |  StageDownsample (2x2, s2)   -> (B, dims[1], H/8,  W/8)     [/2]
   v
 Stage 1  CPE + Transformer blocks (dims[1])
   |  StageDownsample             -> (B, dims[2], H/16, W/16)    [/2]
   v
 Stage 2  CPE + Transformer blocks (dims[2])
   |  StageDownsample             -> (B, dims[3], H/32, W/32)    [/2]
   v
 Stage 3  CPE + Transformer blocks (dims[3])
   |
   v
 feature_maps: 4 maps at strides 4, 8, 16, 32
 tokens:          (B, (H/32)*(W/32), dims[-1])
 global_features: (B, dims[-1])          (mean-pooled over tokens)
```

- **Conv stem** (`/4`): two stride-2 `3x3` convolutions with normalization and GELU.
- **Conditional positional encoding (CPE)**: a residual depthwise `3x3` convolution applied at the start
  of each image stage. Because it is convolutional it adapts to arbitrary resolution — there is no fixed
  positional table for images. This is what makes the image path **resolution-adaptive**.
- **Patch-merging downsample** (`/2`): a normalization plus a stride-2 `2x2` convolution between stages,
  which also widens the channels (`dims[i] -> dims[i+1]`).
- **Spatial-32 constraint**: stem `/4` × three stage `/2` downsamples = **`/32`** total, so image `H` and
  `W` must each be divisible by 32. Both `SHADAConfig.validate` and the encoder enforce this.

**Text path**

The same four stages of Transformer blocks run over a token sequence produced by a learned token +
positional `TextEmbedding`. Between stages a length-preserving `TextStageProject` (LayerNorm + Linear)
changes the channel width (`dims[i] -> dims[i+1]`) without changing the sequence length. For the language
model task a causal attention mask is applied so each position only attends to itself and earlier
positions.

**Encoder output contract** — `encoder(x, modality=...)` returns a dict with keys:

| key               | image                              | text                          |
| ----------------- | ---------------------------------- | ----------------------------- |
| `feature_maps`    | list of 4 maps, strides 4/8/16/32  | list of 4 `(B, L, dims[i])`   |
| `tokens`          | `(B, (H/32)*(W/32), dims[-1])`     | `(B, L, dims[-1])`            |
| `global_features` | `(B, dims[-1])`                    | `(B, dims[-1])`               |
| `hw`              | `(Hs, Ws)` int tuple               | `None`                        |
| `modality`        | `"image"`                          | `"text"`                      |

## The four tasks

Select a task with `task=` (or via `create_config(...)`). Each task has a real head, a real loss, and a
task-specific `predict` / `score` contract.

### Classification

- **Input** `X`: `(N, C, H, W)` images (`H`, `W` divisible by 32).
- **Target** `y`: `(N,)` integer labels (one-hot `(N, num_classes)` is also accepted and argmaxed).
- **`predict(X)`** → `(N,)` integer labels. With `return_probs=True`, returns `(labels, probs)` where
  `probs` is `(N, num_classes)`.
- **`score(X, y)`** → top-1 accuracy.

```python
import numpy as np
from shadax import SHADA

X = np.random.randn(16, 3, 64, 64).astype("float32")
y = np.random.randint(0, 5, size=16)

model = SHADA(tier="nano", task="classification", num_classes=5)
model.fit(X, y, epochs=2)

labels = model.predict(X)                     # (N,)
labels, probs = model.predict(X, return_probs=True)  # (N,), (N, 5)
print(labels.shape, probs.shape, model.score(X, y))
```

### Segmentation

- **Input** `X`: `(N, C, H, W)` images.
- **Target** `y`: `(N, H, W)` integer per-pixel class masks.
- **`predict(X)`** → `(N, H, W)` per-pixel argmax labels.
- **`score(X, y)`** → mean per-pixel accuracy.

```python
import numpy as np
from shadax import SHADA

X = np.random.randn(8, 3, 64, 64).astype("float32")
y = np.random.randint(0, 4, size=(8, 64, 64))   # (N, H, W) masks

model = SHADA(tier="nano", task="segmentation", num_classes=4)
model.fit(X, y, epochs=2)

masks = model.predict(X)                          # (N, H, W)
print(masks.shape, model.score(X, y))
```

### Language model (`lm`)

- **Input** `X`: `(N, L)` integer token ids. Reserve id `vocab_size - 1` as the `[MASK]` token (used by
  masked-LM pretraining); do not assign it to a real token.
- **Target** `y`: the `lm` loss is next-token prediction over `X` itself, so **no `y` is needed** — for the
  language-model task even the supervised phases (`FINETUNE` / `MULTITASK`) accept `y=None`. Self-supervised
  pretraining likewise uses `fit(X)` / `pretrain(X)`.
- **`predict(X)`** → `(N, L)` per-position argmax (next-token) ids.
- **`score(X, y)`** → next-token accuracy (positions equal to `pad_token_id` are ignored). Pass the same
  token-id array as `y`.

```python
import numpy as np
from shadax import SHADA, TrainingPhase

vocab_size = 256
# Keep ids in [1, vocab_size-2]: id 0 is padding, id vocab_size-1 is [MASK].
X = np.random.randint(1, vocab_size - 1, size=(8, 32))

model = SHADA(tier="nano", task="lm", vocab_size=vocab_size, max_seq_len=64)
# Supervised next-token finetuning. lm derives its targets from X itself, so no y.
model.fit(X, phases=[TrainingPhase.FINETUNE], epochs=2)

next_ids = model.predict(X)        # (N, L)
print(next_ids.shape, model.score(X, X))
```

### Detection (anchor-free, CenterNet-style)

The detection head predicts dense maps on the **stride-32 grid** `(Hs, Ws) = (H/32, W/32)`.

- **Input** `X`: `(N, C, H, W)` images.
- **Target** `y`: a dict of dense CenterNet targets on `(Hs, Ws)`:

  | key        | shape              | meaning                                |
  | ---------- | ------------------ | -------------------------------------- |
  | `heatmap`  | `(N, C, Hs, Ws)`   | per-class center heatmap, float `[0,1]`|
  | `wh`       | `(N, 2, Hs, Ws)`   | box width/height at each location      |
  | `offset`   | `(N, 2, Hs, Ws)`   | sub-pixel center offset                |
  | `reg_mask` | `(N, 1, Hs, Ws)`   | `1` where a center exists, else `0`    |

  Here `C` is `num_classes` (object categories).
- **`predict(X)`** → a length-`N` list of dicts `{"boxes": (k, 4), "scores": (k,), "labels": (k,)}`, where
  each box is `(cx, cy, w, h)` in stride-32 grid coordinates. `k` is the top-`k` peaks kept
  (`min(20, num_classes * Hs * Ws)`), so on a tiny grid `k` may be smaller than 20.
- **`score(X, y)`** raises `NotImplementedError` — use a dedicated mAP metric (e.g.
  `torchmetrics.detection.MeanAveragePrecision` or `pycocotools`).

```python
import numpy as np
from shadax import SHADA

N, C_in, H, W = 4, 3, 64, 64
num_classes = 3
Hs, Ws = H // 32, W // 32        # stride-32 grid (here 2 x 2)

X = np.random.randn(N, C_in, H, W).astype("float32")

# Build the dense CenterNet target dict. Here we plant one object per image.
heatmap  = np.zeros((N, num_classes, Hs, Ws), dtype="float32")
wh       = np.zeros((N, 2, Hs, Ws), dtype="float32")
offset   = np.zeros((N, 2, Hs, Ws), dtype="float32")
reg_mask = np.zeros((N, 1, Hs, Ws), dtype="float32")
for i in range(N):
    cls, gy, gx = i % num_classes, i % Hs, i % Ws
    heatmap[i, cls, gy, gx] = 1.0      # a center peak for class `cls`
    wh[i, :, gy, gx] = [1.5, 1.0]      # box size at that center
    offset[i, :, gy, gx] = [0.2, 0.1]  # sub-pixel offset
    reg_mask[i, 0, gy, gx] = 1.0       # mark the center location

y = {"heatmap": heatmap, "wh": wh, "offset": offset, "reg_mask": reg_mask}

model = SHADA(tier="nano", task="detection", num_classes=num_classes)
model.fit(X, y, epochs=2)

dets = model.predict(X)                # list of N dicts
print(len(dets), dets[0]["boxes"].shape, dets[0]["scores"].shape, dets[0]["labels"].shape)
```

## Self-supervised pretraining

Two objectives ship with the library, selected automatically from the input modality:

- **Masked image modeling (MIM)** — MAE-style. A random fraction (`mask_ratio`, default `0.75`) of the
  image patches (at granularity 32) is replaced by a learnable mask token; the masked image is encoded and
  a lightweight convolutional decoder reconstructs the original pixels. The loss is the MSE over masked
  pixels only.
- **Masked language modeling (MLM)** — BERT-style. A random fraction (`text_mask_ratio`, default `0.15`)
  of the non-padding tokens is replaced by the reserved `[MASK]` id (`vocab_size - 1`); the masked
  sequence is encoded and the LM head predicts the originals. The loss is cross-entropy over the masked
  positions only.

Call `pretrain(X, ...)` (a convenience wrapper for `fit(X, y=None, phases=[PRETRAIN], ...)`):

```python
import numpy as np
from shadax import SHADA

# Image pretraining (MIM). No labels.
X_img = np.random.randn(16, 3, 64, 64).astype("float32")
img_model = SHADA(tier="nano", task="classification", num_classes=10)
img_model.pretrain(X_img, epochs=2)

# Text pretraining (MLM). Reserve id vocab_size-1 as [MASK].
vocab_size = 256
X_txt = np.random.randint(1, vocab_size - 1, size=(16, 32))
txt_model = SHADA(tier="nano", task="lm", vocab_size=vocab_size, max_seq_len=64)
txt_model.pretrain(X_txt, epochs=2)
```

## The four-phase training pipeline

`TrainingPhase` defines four phases; `fit(..., phases=[...])` runs any sequence of them in order. Each
optimised phase uses a fresh `AdamW` optimiser and a cosine-annealing schedule.

| phase       | what it optimises                                  | needs labels? |
| ----------- | -------------------------------------------------- | ------------- |
| `PRETRAIN`  | self-supervised loss only (MIM or MLM)             | no            |
| `MULTITASK` | task loss `+ ssl_weight *` self-supervised loss    | yes           |
| `FINETUNE`  | task loss only                                     | yes           |
| `DEPLOY`    | nothing — switches to `eval` mode, no optimisation | no            |

**Default phases** when `phases` is not given: `[PRETRAIN]` if `y is None`, otherwise `[FINETUNE]`.

```python
import numpy as np
from shadax import SHADA, TrainingPhase

X = np.random.randn(16, 3, 64, 64).astype("float32")
y = np.random.randint(0, 10, size=16)

model = SHADA(tier="nano", task="classification", num_classes=10)

# Full pipeline: self-supervised pretrain, then joint multitask, then finetune.
model.fit(
    X, y,
    phases=[TrainingPhase.PRETRAIN, TrainingPhase.MULTITASK, TrainingPhase.FINETUNE],
    epochs=2,
)
print(model.score(X, y))
```

## Multi-modal usage

The lower-level `HierarchicalEncoder` is public and processes both modalities with the *same* weights.
Build it from a config and call it with `modality="image"` or `modality="text"`.

```python
import torch
from shadax import HierarchicalEncoder, create_config

config = create_config("nano", task="classification", num_classes=10)
encoder = HierarchicalEncoder(config).eval()

# Image batch: (B, C, H, W), H and W divisible by 32.
images = torch.randn(2, 3, 64, 64)
img_out = encoder(images, modality="image")
print("image global:", img_out["global_features"].shape)   # (2, dims[-1])
print("image tokens:", img_out["tokens"].shape)            # (2, (H/32)*(W/32), dims[-1])

# Text batch: (B, L) integer token ids.
tokens = torch.randint(0, config.vocab_size, (2, 16))
txt_out = encoder(tokens, modality="text")
print("text global:", txt_out["global_features"].shape)    # (2, dims[-1])
print("text tokens:", txt_out["tokens"].shape)             # (2, 16, dims[-1])
```

## Model tiers

All tiers share `encoder_depths`/`num_heads` *lengths* of 4 (one per stage). `max_seq_len` is the text
positional-table size for that tier.

| tier   | `encoder_dims`           | `encoder_depths` | `num_heads`     | `max_seq_len` |
| ------ | ------------------------ | ---------------- | --------------- | ------------- |
| `nano` | `[64, 128, 256, 512]`    | `[2, 2, 4, 2]`   | `[2, 4, 8, 16]` | 512           |
| `base` | `[128, 256, 512, 1024]`  | `[3, 4, 6, 3]`   | `[4, 8, 16, 32]`| 1024          |
| `large`| `[192, 384, 768, 1536]`  | `[3, 4, 18, 3]`  | `[6, 12, 24, 48]`| 2048         |
| `xl`   | `[256, 512, 1024, 2048]` | `[3, 4, 24, 3]`  | `[8, 16, 32, 64]`| 4096         |

## Feature extraction

Once fitted, `extract_features(X, layer=...)` returns encoder representations as numpy arrays:

- `layer="global"` → pooled global features `(N, final_dim)`.
- `layer="tokens"` → final token sequence `(N, N_tok, final_dim)`.
- `layer="spatial"` → last image feature map `(N, final_dim, Hs, Ws)` (image modality).

```python
import numpy as np
from shadax import SHADA

X = np.random.randn(8, 3, 64, 64).astype("float32")
y = np.random.randint(0, 10, size=8)
model = SHADA(tier="nano", task="classification", num_classes=10).fit(X, y, epochs=1)

g = model.extract_features(X, layer="global")    # (8, final_dim)
t = model.extract_features(X, layer="tokens")    # (8, N_tok, final_dim)
s = model.extract_features(X, layer="spatial")   # (8, final_dim, 2, 2)
print(g.shape, t.shape, s.shape)
```

## Save / load

```python
import numpy as np
from shadax import SHADA

X = np.random.randn(8, 3, 64, 64).astype("float32")
y = np.random.randint(0, 10, size=8)
model = SHADA(tier="nano", task="classification", num_classes=10).fit(X, y, epochs=1)

model.save("shada_model.pt")

reloaded = SHADA(tier="nano", task="classification", num_classes=10)
reloaded.load("shada_model.pt")
print(reloaded.is_fitted, reloaded.predict(X).shape)
```

## API reference

### `SHADA` (high-level estimator)

```python
SHADA(
    tier="base",            # tier string ("nano"/"base"/"large"/"xl") OR a SHADAConfig
    num_classes=1000,
    task="classification",  # "classification" / "segmentation" / "lm" / "detection"
    learning_rate=1e-4,
    weight_decay=0.05,
    epochs=100,             # default epochs per phase
    batch_size=64,
    device=None,            # None -> "cuda" if available else "cpu"
    phases=None,            # default phase list; resolved at fit() time
    **kwargs,               # extra SHADAConfig overrides (e.g. vocab_size, max_seq_len)
)
```

Methods:

| method | signature | summary |
| ------ | --------- | ------- |
| `fit` | `fit(X, y=None, eval_set=None, verbose=True, epochs=None, phases=None) -> self` | Train through the resolved phases. `fit` **does** accept `epochs=`. |
| `pretrain` | `pretrain(X, epochs=None, verbose=True) -> self` | SSL-only shortcut: `fit(X, y=None, phases=[PRETRAIN], ...)`. |
| `predict` | `predict(X, return_probs=False)` | Task-specific predictions (see each task above). |
| `predict_proba` | `predict_proba(X) -> np.ndarray` | Probabilities (classification/segmentation/lm); raises `NotImplementedError` for detection. |
| `score` | `score(X, y) -> float` | Accuracy in `[0, 1]`; raises `NotImplementedError` for detection. |
| `extract_features` | `extract_features(X, layer="global")` | `"global"` / `"tokens"` / `"spatial"` features. |
| `save` | `save(path) -> None` | Save config + weights + hyper-parameters. |
| `load` | `load(path) -> self` | Restore a saved model. |
| `is_fitted` | property `-> bool` | Whether the model has been fitted. |

### `SHADAConfig` (dataclass — the model-shape contract)

Key fields (with defaults): `tier="base"`, `encoder_dims=[128,256,512,1024]`, `encoder_depths=[3,4,6,3]`,
`num_heads=[4,8,16,32]`, `mlp_ratio=4.0`, `dropout=0.1`, `in_channels=3`, `image_size=224`,
`max_seq_len=1024`, `vocab_size=50257`, `task="classification"`, `num_classes=1000`, `mask_ratio=0.75`,
`text_mask_ratio=0.15`, `decoder_dim=256`, `decoder_depth=2`, `pad_token_id=0`. Properties: `embed_dim`,
`final_dim`, `num_stages`, `task_type`. `validate()` enforces the per-stage list lengths (4 entries each),
divisibility of each `encoder_dims[i]` by `num_heads[i]`, and the mask-ratio / `num_classes` ranges.

### `create_config`

```python
create_config(tier="base", task="classification", num_classes=1000, **overrides) -> SHADAConfig
```

Builds a validated `SHADAConfig` from a tier preset, applying any field `**overrides`.

### `HierarchicalEncoder` (shared backbone, `nn.Module`)

```python
HierarchicalEncoder(config: SHADAConfig)
encoder(x, modality="image", causal=False) -> dict   # see the encoder output contract above
encoder.forward_features(x, modality="image", causal=False) -> list   # the 4 feature maps only
```

### `SHADANet` (unified encoder + head + SSL, `nn.Module`)

```python
SHADANet(config: SHADAConfig)
net(x, modality=None) -> dict          # routes the encoder output through the task head
net.ssl_loss(x, modality=None) -> dict # the modality-matched self-supervised loss
net.encode(x, modality=None) -> dict   # raw encoder output (feature extraction)
```

### Enums (controlled vocabularies)

- `ModelTier`: `NANO`, `BASE`, `LARGE`, `XL`.
- `TaskType`: `CLASSIFICATION`, `DETECTION`, `SEGMENTATION`, `LANGUAGE_MODEL` (value `"lm"`).
- `TrainingPhase`: `PRETRAIN`, `MULTITASK`, `FINETUNE`, `DEPLOY`.
- `Modality`: `IMAGE`, `TEXT`.

## Requirements

- **Python** `>=3.8`.
- **PyTorch** `>=2.0` (hard dependency).
- **NumPy**.

## License

Released under the **MIT License**. See [LICENSE](LICENSE).
