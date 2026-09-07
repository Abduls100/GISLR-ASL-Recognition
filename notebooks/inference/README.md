# Inference & Deployment

## Overview
Real-time inference pipelines and deployment strategies for production ASL recognition.

## Notebook
- **inference.ipynb** - Complete inference pipeline with deployment using gradio

## Input Formats
- **Video Files:** MP4
- **Live Camera:** Webcam stream (Not Supported at the moments)
- **Image Sequences:** Frame-by-frame processing

## Output Format
```python
{
  "sign": "HELLO",
  "confidence": 0.95,
  "top_5": [
    {"sign": "HELLO", "prob": 0.95},
    {"sign": "HI", "prob": 0.03},
    ...
  ],
  "timestamp": 1.23,
  "frame_id": 45
}
```

## Features
✅ Batch inference support  
✅ Real-time frame processing  
✅ Confidence scoring  
✅ GPU acceleration  
✅ CPU fallback support  

## Testing Files
Located in `inference-testing-files/`:
- **audio/** - Sample audio files for testing
- **video/** - Sample video files for testing

Use these for development and validation before production deployment.

## Deployment Options
1. **Local Inference** - Single GPU machine
2. **Batch Processing** - Process multiple videos
3. **Web Service** - Gradio interface
4. **Mobile** - TFLite conversion

## Next Steps
1. Test with sample videos
2. Fine-tune confidence threshold
3. Deploy to target environment
4. Monitor performance metrics
5. Collect user feedback

