# Hybrid SqueezeFormer Models

## Overview

This folder contains the **Enhanced Hybrid SqueezeFormer** training and evaluation notebooks reported in the article *“Multimodal Sign Language Recognition with Transformer-Based Architectures and Language-Aware Speech Generation.”*

The Enhanced Hybrid SqueezeFormer combines local Conv1D temporal modeling, Transformer-based global attention, explicit 3D kinematic features, and Efficient Channel Attention (ECA). The naming and metrics in this README are aligned with the article and the root repository README.

---

## Notebooks

- **`enhanced_hybrid_squeezeformer.ipynb`** — training and validation of the Enhanced Hybrid SqueezeFormer.
- **`enhanced_hybrid_squeezeformer_evaluation.ipynb`** — held-out evaluation, Top-K metrics, confusion analysis, and per-class performance.

The Hybrid SqueezeFormer baseline is maintained separately at:

```text
../baseline/hybrid_squeezeformer.ipynb
```

---

## Model Configuration

| Item | Enhanced Hybrid SqueezeFormer |
|---|---:|
| Input length | 128 frames |
| Landmark coordinates | 3D (x, y, z) |
| Kinematic features | Position + velocity + acceleration |
| Embedding dimension | 192 |
| Conv1D blocks | 6 |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Channel recalibration | ECA |
| Precision | Mixed FP16/FP32 |
| Classes | 250 |
| Parameters | 1,915,288 |

The model uses an explicit validity mask and combines local temporal feature extraction with global attention-based sequence modeling.

---

## Training Configuration

| Hyperparameter | Value |
|---|---:|
| Input length | 128 frames |
| Batch size | 64 |
| Maximum epochs | 200 |
| Optimizer | AdamW |
| Maximum learning rate | 4e-4 |
| Weight decay | 0.05 |
| Gradient clipping | 1.0 |
| Early-stopping patience | 25 |
| Precision | Mixed precision |

Training also includes feature/time masking and low-frame-rate robustness augmentation.

---

## Reported Results

The following values are the metrics reported in the article.

| Metric | Value |
|---|---:|
| Training Top-1 | 83.14% |
| Validation Top-1 | 74.98% |
| Best validation-accuracy checkpoint | 75.05% |
| Validation Top-5 | 92.89% |
| Validation macro-F1 | 0.7451 |
| Median per-class F1 | 0.7665 |
| Test Top-1 | **77.47%** |
| Test Top-5 | **91.78%** |
| Test loss | 1.2410 |
| Parameters | **1,915,288** |

These values replace earlier approximate or development-stage claims such as `~90% Top-1`, `~98% Top-5`, and `922K parameters`, which are not the article-reported final metrics.

---

## Improvement Over Hybrid SqueezeFormer

The article reports the following progression:

| Metric | Hybrid SqueezeFormer | Enhanced Hybrid SqueezeFormer |
|---|---:|---:|
| Test Top-1 | 73.78% | **77.47%** |
| Test Top-5 | 92.51% | 91.78% |
| Parameters | 2,681,594 | **1,915,288** |

This corresponds to:

- **+3.69 percentage points** in test Top-1 accuracy;
- approximately **28.6% fewer parameters**.

Because the baseline and enhanced models use different evaluation configurations, these results should be interpreted as a design progression rather than as a strict identical-split ablation.

---

## Architecture Progression

```text
Hybrid SqueezeFormer
    ↓
longer 128-frame temporal window
    ↓
3D position + velocity + acceleration
    ↓
explicit validity masking
    ↓
Conv1D local modeling
    ↓
Transformer global attention
    ↓
ECA channel recalibration
    ↓
Enhanced Hybrid SqueezeFormer
```

---

## Evaluation

The evaluation notebook includes:

- Top-K accuracy;
- confusion analysis;
- per-class precision, recall, and F1;
- strongest and weakest classes;
- model confidence analysis;
- qualitative inspection of common confusion pairs.

The article reports examples of recurring confusions including:

- `mouth` vs. `lips`;
- `awake` vs. `wake`;
- `nap` vs. `sleep`.

---

## Deployment Context

The Enhanced Hybrid SqueezeFormer is used in the repository's downstream proof-of-concept recognition-to-speech pipeline.

The complete system should **not** be described as a real-time production interpreter. A recorded end-to-end run reported approximately **54.88 s** total latency, with the MediaPipe visual front end contributing most of the runtime.

Appropriate wording is:

- research prototype;
- proof-of-concept assistive pipeline;
- video-based sign recognition and sign-to-speech demonstration.

Avoid unsupported descriptions such as:

- production-ready;
- real-time conversational interpretation;
- 2× faster deployment;
- 48% memory reduction;

unless those claims are separately reproduced and documented.

---

## Reproducibility Notes

When reproducing the experiment, record:

- notebook version;
- checkpoint;
- input length;
- evaluation configuration;
- model parameter count;
- validation metrics;
- held-out/test metrics.

The value **75.05%** refers to the best validation-accuracy checkpoint, whereas **74.98%** is the validation Top-1 value reported in the article's restored evaluation. These should not be treated as contradictory test results.

---

## Related Files

```text
notebooks/
├── baseline/
│   └── hybrid_squeezeformer.ipynb
└── HSF/
    ├── enhanced_hybrid_squeezeformer.ipynb
    ├── enhanced_hybrid_squeezeformer_evaluation.ipynb
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
The notebooks are provided for reproducibility and analysis of the Hybrid SqueezeFormer experiments reported in the article.
