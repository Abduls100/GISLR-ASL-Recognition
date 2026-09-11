# Transformer Models

## Overview

This folder contains the **Enhanced Transformer** experiment reported in the article *“Multimodal Sign Language Recognition with Transformer-Based Architectures and Language-Aware Speech Generation.”* The model extends the Compact Transformer baseline with longer temporal input and richer landmark-derived kinematic features.

The naming and metrics in this README are aligned with the article and the root repository README.

---

## Notebook

- **`enhanced_transformer.ipynb`** — training, validation, and evaluation of the Enhanced Transformer.

The Compact Transformer baseline is maintained separately at:

```text
../baseline/compact_transformer.ipynb
```

---

## Model Configuration

| Item | Enhanced Transformer |
|---|---:|
| Input length | 128 frames |
| Descriptor dimension | 609 |
| Latent dimension | 256 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Feed-forward dimension | 512 |
| Temporal modeling | Transformer + depthwise temporal convolution |
| Precision | Mixed FP16/FP32 |
| Classes | 250 |
| Parameters | 2,366,970 |

The input representation includes normalized landmark coordinates together with selected geometric and temporal features, including wrist-to-fingertip distances and first- and second-order temporal derivatives.

---

## Training Configuration

| Hyperparameter | Value |
|---|---:|
| Batch size | 64 |
| Maximum epochs | 200 |
| Optimizer | AdamW |
| Maximum learning rate | 4e-4 |
| Weight decay | 0.05 |
| Label smoothing | 0.10 |
| Gradient clipping | 1.0 |
| Early-stopping patience | 25 |
| Precision | Mixed precision |

---

## Reported Results

The following values are the metrics reported in the article.

| Metric | Value |
|---|---:|
| Training Top-1 | 89.35% |
| Validation Top-1 | 74.12% |
| Validation Top-5 | 92.10% |
| Validation Top-10 | 96.41% |
| Validation loss | 1.8532 |
| Test Top-1 | **75.25%** |
| Test Top-5 | **90.65%** |
| Test Top-10 | 93.33% |
| Mean confidence | 0.6813 |
| Parameters | 2,366,970 |

These values should be used instead of earlier approximate claims such as `~88%` or `92%+`, which were not the article-reported held-out results.

---

## Evaluation Context

The Enhanced Transformer uses the **128-frame evaluation configuration**. It should therefore not be treated as a strict one-to-one ablation against the 64-frame Compact Transformer under an identical held-out partition.

Its role in the study is to examine whether a richer kinematic representation and longer temporal context improve recognition performance relative to the compact Transformer design.

---

## Relationship to the Compact Transformer

The model progression is:

```text
Compact Transformer
    ↓
longer temporal window
    ↓
richer geometric + temporal descriptors
    ↓
deeper Transformer encoder
    ↓
Enhanced Transformer
```

The Enhanced Transformer increases representation capacity and training accuracy, although its held-out Top-1 accuracy is slightly lower than that of the Compact Transformer under the corresponding evaluation configurations.

---

## Reproducibility Notes

When reproducing this experiment, record:

- notebook version;
- dataset split/evaluation configuration;
- checkpoint used;
- input length;
- model parameter count;
- validation metrics;
- held-out/test metrics.

Do not mix validation accuracy, checkpoint-selection accuracy, and held-out test accuracy.

---

## Related Files

```text
notebooks/
├── baseline/
│   └── compact_transformer.ipynb
└── transformer/
    ├── enhanced_transformer.ipynb
    └── README.md
```

---

## Dataset

The experiment uses the public **Google Isolated Sign Language Recognition (GISLR)** dataset:

https://www.kaggle.com/competitions/asl-signs

The task contains 250 isolated ASL sign classes represented using MediaPipe landmark sequences.

---

## Status

**Research implementation.**  
This notebook is provided for reproducibility and analysis of the experiment reported in the article.
