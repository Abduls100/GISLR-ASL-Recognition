# 🤟 GISLR: American Sign Language Recognition with Deep Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](https://github.com)

# Advanced deep learning architectures for recognizing American Sign Language from video sequences using pose estimation landmarks. Comparing Hybrid SqueezeFormer (HSF) and pure Transformer architectures.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Comparison](#architecture-comparison)
3. [Project Structure](#project-structure)
4. [Notebook Organization](#notebook-organization)
5. [Quick Start](#quick-start)
6. [Installation](#installation--setup)
7. [Performance Metrics](#performance-metrics)
8. [Model Descriptions](#model-descriptions)
9. [Data Requirements](#data-requirements)
10. [Usage Examples](#usage-examples)
11. [Training Guide](#training-guide)
12. [Inference & Deployment](#inference--deployment)
13. [Results & Analysis](#results--analysis)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

This repository contains research and experimentation notebooks for **American Sign Language (ASL) recognition** using two distinct deep learning paradigms:

### Two Architectural Approaches:

1. **Hybrid SqueezeFormer (HSF)** - CNN + Transformer fusion
   - Combines depthwise separable convolutions with multi-head attention
   - Optimized for efficiency and performance balance
   - Production-ready models with multiple variants

2. **Pure Transformer** - Self-attention only
   - Traditional transformer architecture
   - Strong sequential modeling capabilities
   - Research-focused exploration

### Key Achievements

| Metric | HSF Baseline | HSF Production | Transformer |
|--------|--------------|----------------|-------------|
| **Top-1 Accuracy** | ~85% | ~90% | ~88% |
| **Top-5 Accuracy** | ~97% | ~98% | ~97% |
| **Model Parameters** | 2.7M | 922K | TBD |
| **Memory Usage** | Baseline | 48% reduction | Moderate |
| **Training Speed** | Baseline | 2× faster | Baseline |
| **Classes** | 250 | 250 | 250 |

---

## 🏗️ Project Structure

```
GISLR-ASL-Recognition/
│
├── 📁 notebooks/                          # Jupyter notebooks organized by architecture
│   ├── baseline/                          # Baseline models (research foundation)
│   │   ├── hsf-baseline/
│   │   │   ├── 1D-CNN-Transformer.ipynb   (HSF baseline - SqueezeFormer design)
│   │   │   └── README.md
│   │   │
│   │   └── transformer-baseline/
│   │       ├── model-train-transformer.ipynb  (Pure Transformer baseline)
│   │       └── README.md
│   │
│   ├── HSF/                            # Production-optimized HSF models 🏆
│   │   ├── hybrid-methods-improvement-best.ipynb   (Main HSF production)
│   │   ├── best-model-testing-hybrid.ipynb         (HSF evaluation)
│   │   └── README.md
│   │
│   ├── Transformer/                       # Experimental variants & research
│   │   ├── new-transformer-best.ipynb     (Latest HSF refinements)
│   │   ├── testing-hybrid-model.ipynb     (Validation & debugging)
│   │   └── README.md
│   │
│   └── inference/                         # Real-time inference & deployment
│       ├── inference.ipynb                (Inference pipeline)
│       ├── inference-testing-files/
│       │   ├── audio/
│       │   └── video/
│       └── README.md
│
├── .gitignore                             # Git ignore rules
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
```

---

## Architecture Comparison

### Hybrid SqueezeFormer (HSF) Architecture

**Strengths:**
- ✅ Efficient convolution operations (depthwise separable)
- ✅ Local feature extraction via Conv1D blocks
- ✅ Global context via Transformer blocks
- ✅ Smaller model size (922K parameters)
- ✅ Faster inference (2× speedup)
- ✅ Lower memory footprint (48% reduction)
- ✅ Production-ready with mixed precision (FP16)

**Architecture:**
```
Input (64/128 × 66 × 3)
    ↓
Landmark Embedding (320-D baseline / 192-D production)
    ↓
Depthwise Separable Conv1D Blocks (3-4 blocks)
    ↓
Multi-Head Self-Attention (Transformer block)
    ↓
Additional Conv1D + BiLSTM (production variant)
    ↓
Classification Head (250 classes)
```

**Key Components:**
- Conv kernel size: 17
- Transformer heads: 8 (baseline) / 4 (production)
- Stochastic depth: 0.0 → 0.2 (progressive)
- Training: 50-200 epochs, cosine/one-cycle schedule

---

### Pure Transformer Architecture

**Strengths:**
- ✅ Pure self-attention mechanism
- ✅ Strong sequential dependency modeling
- ✅ Proven transformer backbone
- ✅ Flexible architecture
- ✅ Good performance on long sequences
- ✅ Research-friendly design

**Architecture:**
```
Input (Sequence × Features)
    ↓
Embedding Layer
    ↓
Positional Encoding
    ↓
Transformer Encoder Blocks
    ↓
Global Pooling
    ↓
Classification Head (250 classes)
```

**Characteristics:**
- Pure attention-based design
- No convolutional layers
- Relies on transformer stacks for feature extraction
- Suitable for research and exploration

---

## 📊 Notebook Organization

### **Baseline Models **

#### Hybrid SqueezeFormer Baseline
**📄 notebooks/baseline/hsf-baseline/1D-CNN-Transformer.ipynb**

Purpose: Establish HSF baseline architecture

Contains:
- Original SqueezeFormer-style CNN-Transformer design
- Landmark extraction and feature engineering
- Comprehensive evaluation metrics
- Baseline for HSF improvements

Architecture:
```
Input: 64 frames × 66 landmarks × 3D
Embedding: 320-D
Backbone: 4 Conv1D + 2 Transformer blocks
Parameters: 2.7M
```

Performance:
- Top-1 Accuracy: 85.12%
- Top-5 Accuracy: 96.88%
- Training Time: ~25 hours

Key Features:
- ✅ Separate landmark-group embeddings (lips, hands, pose)
- ✅ Learned positional encoding
- ✅ Stochastic depth (progressive 0.0→0.2)
- ✅ Depthwise separable convolutions
- ✅ MixUp and feature masking augmentation

Use for:
- Understanding HSF fundamentals
- Research baseline
- Educational reference

---

#### Pure Transformer Baseline
**📄 notebooks/baseline/transformer-baseline/model-train-transformer.ipynb**

Purpose: Establish pure Transformer baseline architecture

Contains:
- Complete transformer training loop
- Multi-head self-attention implementation
- Learning rate scheduling
- Callbacks and monitoring

Architecture:
- Pure transformer encoder
- No convolutional layers
- Focus on sequential modeling

Key Features:
- ✅ Clean transformer implementation
- ✅ Training loop demonstration
- ✅ Advanced optimization techniques
- ✅ Comprehensive monitoring

Use for:
- Understanding transformer mechanics
- Comparison baseline
- Research exploration

---

### **Production Models (Hybrid SqueezeFormer) 🏆**

#### Main Production HSF Model
**📄 notebooks/hybrid/hybrid-methods-improvement-best.ipynb**

Purpose: Production-optimized HSF with best performance/efficiency balance

Improvements over baseline:
- ✅ Mixed precision training (FP16) - 2× speedup
- ✅ Longer sequences (128 vs 64 frames) - better temporal resolution
- ✅ Bidirectional LSTM for long-range dependencies
- ✅ ECA (Efficient Channel Attention)
- ✅ Learnable positional embeddings
- ✅ One-cycle learning rate with warmup

Architecture:
```
Input: 128 frames × N landmarks × 3D (FP16)
Embedding: 192-D (3× smaller than baseline)
Backbone: 5 Conv1D + 1 Transformer + BiLSTM
Parameters: 922K (3× smaller!)
```

Performance:
- Top-1 Accuracy: ~90%
- Top-5 Accuracy: ~98%
- Memory: 48% reduction
- Training Speed: 2× faster
- Training Time: ~50 hours (200 epochs)

Hyperparameters:
- Sequence Length: 128 frames
- Batch Size: 64
- Epochs: 200
- Learning Rate: 4e-4 (one-cycle, 30% warmup)
- Optimizer: AdamW (coupled weight decay)
- Precision: Mixed FP16/FP32

Key Features:
- ✅ Production-ready
- ✅ Optimal accuracy-efficiency tradeoff
- ✅ Streaming inference support
- ✅ GPU acceleration + CPU fallback
- ✅ Sophisticated augmentation pipeline

Use for:
- Production deployment
- Real-time inference
- Limited compute environments
- Best accuracy with efficiency

---

#### HSF Evaluation & Testing
**📄 notebooks/hybrid/best-model-testing-hybrid.ipynb**

Purpose: Comprehensive model evaluation and analysis

Includes:
- Top-K accuracy analysis (Top-1, Top-5, Top-10, Top-20)
- Confusion matrix (top 20 confused pairs)
- Per-class F1-score analysis
- Best/worst performing signs identification
- Detailed classification reports
- Performance visualizations

Output:
- Classification reports (CSV)
- Confusion analysis
- Per-class metrics
- Recommendations for improvement

Use for:
- Model quality assurance
- Performance analysis
- Identifying failure modes
- Generating reports


#### Model Testing & Debugging
**📄 notebooks/experiments/testing-hybrid-model.ipynb**

Purpose: Validation, debugging, and experimental analysis

Contains:
- Model validation procedures
- Debugging techniques
- Performance benchmarking
- Ablation studies

Use for:
- Understanding model behavior
- Debugging issues
- Experimental validation

---

### **Inference & Deployment**

#### Real-Time Inference Pipeline
**📄 notebooks/inference/inference.ipynb**

Purpose: Production inference and deployment

Features:
- ✅ Batch inference support
- ✅ Real-time frame processing
- ✅ Confidence scoring
- ✅ Multi-sign detection
- ✅ Streaming inference (prevents OOM)

Input Formats:
- Video files (MP4)
- Image sequences

Output:
- Predicted sign (text)
- Confidence score (0-100%)
- Frame-level predictions
- Performance metrics

Use Cases:
- Live sign language interpretation
- Video batch processing
- Integration with applications
- Real-world deployment

Testing Files:
Located in `inference-testing-files/` for development and validation.

---

## 🎯 Quick Start
### The implementations of this project can be done using two route: Use kaggle for resource maximization or use you local computer to deploy the system.


### Prerequisites
```bash
# Python 3.8+
python --version

# GPU support (optional but recommended)
nvidia-smi  # Check CUDA availability
```

### Installation

```bash
# Clone repository
git clone https://github.com/Abduls100/GISLR-ASL-Recognition.git
cd GISLR-ASL-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Recommended order:

# 1. Understanding Baselines
notebooks/baseline/hsf-baseline/1D-CNN-Transformer.ipynb 
notebooks/baseline/transformer-baseline/model-train-transformer.ipynb

# 2. HSF Models
notebooks/hybrid/hybrid-methods-improvement-best.ipynb
notebooks/hybrid/best-model-testing-hybrid.ipynb

# 3.Transformer Models
notebooks/experiments/new-transformer-best.ipynb

# 4. Deployment
notebooks/inference/inference.ipynb
```
###Notebooks link
https://www.kaggle.com/code/abdulsamadibrahim/hybrid-methods-improvement-best
https://www.kaggle.com/code/abdulsamadibrahim/1d-cnn-transformer-improved
https://www.kaggle.com/code/abdulsamadibrahim/new-transformer
https://www.kaggle.com/code/abdulsamadibrahim/inference

---

## 🔧 Installation & Setup

### System Requirements

| Component | Requirement | Recommended |
|-----------|-------------|------------|
| **OS** | Windows/macOS/Linux | Linux (Ubuntu 20.04+) |
| **Python** | 3.8+ | 3.9 or 3.10 |
| **GPU** | Optional | NVIDIA with CUDA 11.2+ |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 20GB | 50GB+ |


## Data Requirements

### Dataset Format

The notebooks expect the **Google GISLR dataset** format:

```
Dataset/
├── train/
│   ├── sign_1/
│   │   ├── frame_1.parquet
│   │   ├── frame_2.parquet
│   │   └── ...
│   ├── sign_2/
│   └── ...
│
├── train.csv
│   └── Columns: path, sign, participant_id
│
└── test/
    ├── video_1.mp4
    ├── video_2.mp4
    └── ...
```
This dataset is availble on Kaggle at https://www.kaggle.com/competitions/asl-signs

### Landmark Information

**Total Landmarks:** 543 (MediaPipe Holistic pose)

**Extracted Landmarks:**
- **Lips:** 40 points
- **Left Hand:** 21 points
- **Right Hand:** 21 points
- **Pose:** 5 points (shoulders, elbows, hips)
- **Total Used:** 87 landmarks per frame (optimized)

**Coordinates:**
- X, Y: Screen space [0, 1]
- Z: Depth estimation [-1, 1]

---



## Results & Analysis

### Key Findings

```
HSF Baseline Strengths:
✅ Efficient CNN feature extraction
✅ Strong local pattern recognition
✅ Stable training
✅ Good baseline performance (85%)

HSF Production Improvements:
✅ Better temporal understanding (128 vs 64 frames)
✅ BiLSTM for long-range dependencies
✅ Mixed precision optimization (2× speedup)
✅ Higher accuracy (~90%)
✅ Smaller footprint (3× fewer parameters)

Performance Insights:
✅ Top-5 performance is 8-10% better than Top-1
✅ Model is "close but not perfect" on difficult classes
✅ Confusion patterns suggest similar hand shapes are confused
✅ Training longer (200 vs 50 epochs) improves accuracy significantly
```

### Class-Level Analysis

```
Best 10 Performing Classes:
- Common, frequent signs (95%+ F1)
- Clear hand shapes
- Distinct movements

Challenging Classes:
- Fingerspelling (individual letters)
- Similar hand shapes (CAN vs ABLE)
- Subtle movement differences (WHERE vs WHEN)
- Short duration signs
```

---

## 🤝 Contributing

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- 🔍 **Architecture Improvements:** Advanced transformer variants
- 📊 **Data Augmentation:** Domain-specific techniques
- 🎯 **Evaluation:** Additional metrics and analysis tools
- 🚀 **Optimization:** Model compression, quantization
- 📱 **Deployment:** Mobile/edge device support
- 📚 **Documentation:** Tutorials and examples

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@repository{gislr_asl_recognition,
  title={GISLR: American Sign Language Recognition with Deep Learning},
  subtitle={Comparing Hybrid SqueezeFormer and Pure Transformer Architectures},
  author={Abdulsamad Ibrahim},
  year={2024},
  url={https://github.com/YourUsername/GISLR-ASL-Recognition}
}
```

---

## Acknowledgments

- 🙏 Google for the GISLR dataset
- 🙏 MediaPipe team for pose estimation
- 🙏 TensorFlow and Keras communities
- 🙏 Research papers on SqueezeFormer and Transformers

---

## 📞 Contact & Support

- **Issues:** Open an issue on GitHub
- **Discussions:** Use GitHub Discussions for questions


---

## 🗺️ Roadmap

### Phase 1 (Current) ✅
- ✅ HSF baseline architecture
- ✅ HSF production optimization
- ✅ Pure transformer baseline
- ✅ Comprehensive evaluation
- ✅ Production inference pipeline

### Phase 2 (Planned) 🎯
- 🎯 Advanced transformer variants
- 🎯 Multi-modal learning (video + audio)
- 🎯 Cross-architecture ensemble
- 🎯 Web app deployment

### Phase 3 (Future) 🚀
- 🚀 TFLite for mobile
- 🚀 Continuous learning
- 🚀 Transfer learning
- 🚀 Sign sequence generation

---

Made with ❤️ for American Sign Language Technology
