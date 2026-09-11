# GISLR: Multimodal Sign Language Recognition with Transformer-Based Architectures

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/Status-Research-blue)](https://github.com/Abduls100/GISLR-ASL-Recognition)

This repository contains the implementation and evaluation notebooks associated with the study **“Multimodal Sign Language Recognition with Transformer-Based Architectures and Language-Aware Speech Generation.”** The work investigates four landmark-based isolated American Sign Language (ASL) recognition models on the Google Isolated Sign Language Recognition (GISLR) dataset and demonstrates a downstream proof-of-concept pipeline for converting recognized glosses into refined text and speech.

> **Scope:** The recognition experiments address **isolated sign classification**. The language-refinement and text-to-speech components are downstream assistive modules and should not be interpreted as a full continuous sign-language translation system.

---

## Contents

- [Models](#models)
- [Reported Results](#reported-results)
- [Evaluation Note](#evaluation-note)
- [Repository Structure](#repository-structure)
- [Notebook Mapping](#notebook-mapping)
- [Dataset](#dataset)
- [Model Summaries](#model-summaries)
- [Inference and Deployment](#inference-and-deployment)
- [Installation](#installation)
- [Running the Notebooks](#running-the-notebooks)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

---

## Models

The article reports four model configurations:

1. **Compact Transformer**
2. **Enhanced Transformer**
3. **Hybrid SqueezeFormer**
4. **Enhanced Hybrid SqueezeFormer**

The Transformer family emphasizes global temporal modeling, while the Hybrid SqueezeFormer family combines local temporal convolution with Transformer-based attention.

---

## Reported Results

The following values are the metrics reported in the article and are the canonical values used by this repository documentation.

| Model | Frames | Train Top-1 | Validation Top-1 | Test Top-1 | Test Top-5 | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Compact Transformer | 64 | 65.43% | 73.64% | 76.03% | 91.52% | 2,044,282 |
| Enhanced Transformer | 128 | 89.35% | 74.12% | 75.25% | 90.65% | 2,366,970 |
| Hybrid SqueezeFormer | 64 | 78.15% | 67.46% | 73.78% | 92.51% | 2,681,594 |
| Enhanced Hybrid SqueezeFormer | 128 | 83.14% | 74.98% | **77.47%** | 91.78% | **1,915,288** |

### Additional reported metrics

- **Compact Transformer:** validation loss 2.7284, validation Top-5 92.43%, validation Top-10 95.60%, test loss 2.7102.
- **Enhanced Transformer:** validation loss 1.8532, validation Top-5 92.10%, validation Top-10 96.41%, test Top-10 93.33%, mean confidence 0.6813.
- **Hybrid SqueezeFormer:** validation loss 1.7312, validation Top-5 88.38%, validation macro-F1 approximately 0.66, test loss 1.4430.
- **Enhanced Hybrid SqueezeFormer:** validation Top-5 92.89%, validation macro-F1 0.7451, median per-class F1 0.7665, test loss 1.2410.

---

## Evaluation Note

The four experiments were developed under **two evaluation configurations** rather than one identical controlled split. The 64-frame and 128-frame experiments therefore represent an experimental design progression and should **not** be interpreted as a strict component-by-component ablation under exactly the same held-out partition.

For this reason, comparisons between models should consider both architecture and evaluation configuration.

The **Enhanced Hybrid SqueezeFormer** achieves the highest reported test Top-1 accuracy in the study at **77.47%**, while reducing the parameter count from **2.68M** in the Hybrid SqueezeFormer to **1.92M**, a reduction of approximately **28.6%**.

---

## Repository Structure

```text
GISLR-ASL-Recognition/
│
├── notebooks/
│   ├── baseline/
│   │   ├── compact_transformer.ipynb
│   │   ├── hybrid_squeezeformer.ipynb
│   │   └── README.md
│   │
│   ├── transformer/
│   │   ├── enhanced_transformer.ipynb
│   │   └── README.md
│   │
│   ├── HSF/
│   │   ├── enhanced_hybrid_squeezeformer.ipynb
│   │   ├── enhanced_hybrid_squeezeformer_evaluation.ipynb
│   │   ├── testing-hybrid-model.ipynb
│   │   └── README.md
│   │
│   ├── inference/
│   │   └── ...
│   │
│   └── requirements.txt
│
├── README.md
├── .gitignore
└── LICENSE
```

---

## Notebook Mapping

| Article model / task | Repository notebook |
|---|---|
| Compact Transformer | `notebooks/baseline/compact_transformer.ipynb` |
| Enhanced Transformer | `notebooks/transformer/enhanced_transformer.ipynb` |
| Hybrid SqueezeFormer | `notebooks/baseline/hybrid_squeezeformer.ipynb` |
| Enhanced Hybrid SqueezeFormer | `notebooks/HSF/enhanced_hybrid_squeezeformer.ipynb` |
| Enhanced Hybrid SqueezeFormer evaluation | `notebooks/HSF/enhanced_hybrid_squeezeformer_evaluation.ipynb` |
| Experimental/debugging analysis | `notebooks/HSF/testing-hybrid-model.ipynb` |

The notebook names above are intentionally aligned with the model names used in the article.

---

## Dataset

The experiments use the **Google Isolated Sign Language Recognition (GISLR)** dataset:

- **250 sign classes**
- **94,477 labeled training videos**
- MediaPipe landmark sequences
- Participant information provided in the original dataset metadata

Dataset page:

https://www.kaggle.com/competitions/asl-signs

### Landmark representation

The study uses selected hand, lip, and pose landmarks rather than raw RGB video.

Depending on the model configuration, the input representation includes spatial landmark coordinates, normalized hand/lip/upper-body pose information, temporal motion features, first- and second-order temporal derivatives, and selected geometric distance features. The enhanced models use richer 3D kinematic representations and longer temporal windows.

---

## Model Summaries

### Compact Transformer

A lightweight 64-frame Transformer baseline using selected lip, dominant-hand, and pose landmarks.

- Input length: 64 frames
- Hidden dimension: 320
- Attention heads: 8
- Transformer blocks: 2
- Parameters: 2,044,282
- Validation Top-1: 73.64%
- Test Top-1: 76.03%
- Test Top-5: 91.52%

Notebook: `notebooks/baseline/compact_transformer.ipynb`

### Enhanced Transformer

Extends the Transformer baseline with 128-frame sequences and richer kinematic descriptors.

- Input length: 128 frames
- Descriptor dimension: 609
- Latent dimension: 256
- Transformer blocks: 4
- Attention heads: 4
- Parameters: 2,366,970
- Validation Top-1: 74.12%
- Test Top-1: 75.25%
- Test Top-5: 90.65%

Notebook: `notebooks/transformer/enhanced_transformer.ipynb`

### Hybrid SqueezeFormer

Combines local Conv1D processing with global Transformer attention and class-balanced training.

- Input length: 64 frames
- Embedding dimension: 256
- Transformer blocks: 2
- Parameters: 2,681,594
- Validation Top-1: 67.46%
- Test Top-1: 73.78%
- Test Top-5: 92.51%

Notebook: `notebooks/baseline/hybrid_squeezeformer.ipynb`

### Enhanced Hybrid SqueezeFormer

Uses 128-frame 3D landmark sequences, explicit kinematic features, Conv1D local modeling, Transformer global attention, and Efficient Channel Attention (ECA).

- Input length: 128 frames
- 3D position, velocity, and acceleration features
- Embedding dimension: 192
- Conv1D + Transformer hybrid processing
- ECA channel recalibration
- Parameters: 1,915,288
- Validation Top-1: 74.98%
- Validation Top-5: 92.89%
- Test Top-1: **77.47%**
- Test Top-5: 91.78%

Training notebook: `notebooks/HSF/enhanced_hybrid_squeezeformer.ipynb`

Evaluation notebook: `notebooks/HSF/enhanced_hybrid_squeezeformer_evaluation.ipynb`

---

## Inference and Deployment

The repository also contains an inference workflow used for the study's end-to-end proof-of-concept demonstration:

```text
video input
    ↓
MediaPipe landmark extraction
    ↓
sign recognition
    ↓
gloss sequence
    ↓
language refinement
    ↓
text-to-speech output
```

A recorded end-to-end run reported approximately **54.88 s** total latency. Most of this time was associated with the MediaPipe visual front end rather than the classifier itself.

Therefore, the deployed system should be described as a **proof-of-concept assistive pipeline**, not as a real-time conversational sign-language interpreter.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Abduls100/GISLR-ASL-Recognition.git
cd GISLR-ASL-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r notebooks/requirements.txt
```

The experiments were developed primarily with TensorFlow/Keras in Kaggle environments. Exact package versions may differ between notebooks, so inspect each notebook's recorded environment information when reproducing an experiment.

---

## Running the Notebooks

Recommended order:

```text
1. notebooks/baseline/compact_transformer.ipynb
2. notebooks/transformer/enhanced_transformer.ipynb
3. notebooks/baseline/hybrid_squeezeformer.ipynb
4. notebooks/HSF/enhanced_hybrid_squeezeformer.ipynb
5. notebooks/HSF/enhanced_hybrid_squeezeformer_evaluation.ipynb
6. notebooks/inference/
```

The notebooks were originally developed using Kaggle-hosted computational resources, although local execution is also possible when the required dataset paths and dependencies are configured.

---

## Reproducibility Notes

To avoid ambiguity between experimental runs:

- the **model names in this README match the names used in the article**;
- the numerical results above are the **article-reported metrics**;
- validation metrics are reported separately from held-out/test metrics;
- older approximate claims such as `~90%`, `~98%`, `922K parameters`, and `2× faster` are **not used as article results** unless independently reproduced under the corresponding configuration;
- `testing-hybrid-model.ipynb` is retained as an experimental/debugging notebook and is not treated as one of the four primary article models.

When reproducing a result, record the exact notebook, checkpoint, input length, dataset/evaluation configuration, parameter count, validation metrics, and held-out/test metrics.

---

## Kaggle Notebooks

Historical Kaggle versions associated with the development process include:

- https://www.kaggle.com/code/abdulsamadibrahim/hybrid-methods-improvement-best
- https://www.kaggle.com/code/abdulsamadibrahim/1d-cnn-transformer-improved
- https://www.kaggle.com/code/abdulsamadibrahim/new-transformer
- https://www.kaggle.com/code/abdulsamadibrahim/inference

These may preserve earlier notebook names or intermediate experimental states. The **GitHub notebook names and article metrics in this README should be used as the current naming convention**.

---

## Citation

If you use this repository, please cite the associated article once its final bibliographic information is available.

```bibtex
@misc{ibrahim_gislr_asl_recognition_2026,
  author       = {Ibrahim, Abdulsamad},
  title        = {GISLR-ASL-Recognition: Transformer and Hybrid SqueezeFormer Models for Isolated Sign Language Recognition},
  year         = {2026},
  howpublished = {\url{https://github.com/Abduls100/GISLR-ASL-Recognition}},
  note         = {GitHub repository}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work uses the publicly available Google GISLR dataset and builds on the TensorFlow/Keras and MediaPipe ecosystems.
