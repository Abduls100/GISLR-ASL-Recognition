# Hybrid CNN-Transformer Models 

## Overview
Production-optimized hybrid architectures combining CNN, and Transformer for ASL recognition.

## Notebooks
- **hybrid-methods-improvement-best.ipynb** - Main production model
- **best-model-testing-hybrid.ipynb** - Comprehensive evaluation (Top-K, confusion matrix, F1)

## Key Metrics
- Top-1 Accuracy: ~90%
- Top-5 Accuracy: ~98%
- Macro F1-Score: ~0.89
- Parameters: 922K (3× smaller than baseline!)
- Memory: 48% reduction
- Training Speed: 2× faster (mixed FP16)
- Total Training Time: ~50 hours (200 epochs)

## Advantages Over Baseline
✅ Smaller model (3× fewer parameters)  
✅ Better accuracy (+4-7% Top-1)  
✅ Less memory usage (48% reduction)  
✅ Mixed precision (FP16 for speed)  
✅ Longer sequences (128 vs 64 frames)  
✅ BiLSTM for long-range dependencies  
✅ ECA for efficient attention  
✅ Production-ready  

## Hyperparameters

### Training
- Sequence Length: 128 frames
- Embedding Dim: 192
- Batch Size: 64
- Epochs: 200
- Learning Rate: 4e-4 (one-cycle with 30% warmup)
- Optimizer: AdamW
- Weight Decay: 0.05 × LR (coupled)
- Label Smoothing: 0.05

### Architecture
- Transformer Heads: 4
- LSTM Units: 128-D bidirectional
- Conv Kernel Size: 17
- Precision: Mixed FP16/FP32

### Augmentation
- Feature Masking: 20%
- Time Masking: Random frames
- MixUp: 20% probability (α=0.2)
- Noise Injection: Gaussian
- Low FPS Simulation: Video quality variations

## Evaluation
- Top-K accuracy (1, 3, 5, 10, 20)
- Confusion matrix (top 20 confused pairs)
- Per-class F1-scores
- Best/worst 10 classes
- Macro & weighted F1

## Use Cases
- **Production Deployment** - Optimal performance/efficiency
- **Real-time Inference** - Mixed precision for speed
- **Edge Devices** - Smaller model size
- **Limited Compute** - Lower memory requirements
- **POC Prototype Live Translation** - Streaming inference support

## Performance by Class
- Best: Common, frequent signs (95%+)
- Average: Regular signs (85-90%)
- Challenging: Similar handshapes, subtle movements (70-80%)

## Next Steps
- Deploy to production
- Test on live video streams
- Optimize for specific devices
- Fine-tune on domain-specific data
