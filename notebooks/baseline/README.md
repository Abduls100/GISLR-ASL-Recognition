# Baseline Architectures

## Overview
Original SqueezeFormer-style CNN-Transformer and Transformer models for ASL recognition.

## Notebooks
- **1D-CNN-Transformer.ipynb** - Original baseline architecture (85.12% accuracy)
- **model-train-transformer.ipynb** - Transformer baseline architecture


## Hyperparameters
- Sequence Length: 64 frames
- Learning Rate: 1e-3 (cosine annealing)
- Optimizer: AdamW
- Label Smoothing: 0.05
- Weight Decay: 1e-5

## Data Requirements
- Input: 64 frames × 66 landmarks × 3D coordinates
- Output: 250 ASL signs
- Dataset: Google GISLR (train/val split)

## Usage
1. Load preprocessed data from train/test splits
2. Run training notebook
3. Monitor training/validation metrics
4. Save best weights when validation loss plateaus
5. Evaluate on test set

## Output Files
- `model.weights.h5` - Best model weights

## Next Steps
- Compare with Hybrid models for performance
- Experiment with different architectures
- Analyze per-class metrics

## References
- SqueezeFormer: Conformer with Squeezed Time-Channel Interaction
- Original paper: https://arxiv.org/abs/2301.01239

