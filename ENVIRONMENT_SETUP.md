# Environment Setup Guide

This project provides environment setup options for different platforms.

## Environment Files

### 1. `env.yaml` - Apple Silicon (M1/M2/M3 Macs)
```bash
conda env create -f env.yaml
conda activate tf_metal
```
- Optimized for Apple Silicon with Metal Performance Shaders
- Uses `tensorflow-macos` and `tensorflow-metal` for GPU acceleration

### 2. `deepml.yaml` - Linux with GPU Support
```bash
conda env create -f deepml.yaml
conda activate deepml
```
- Uses `tensorflow[and-cuda]` for automatic GPU support
- TensorFlow handles CUDA dependencies automatically
- Optimized for Linux systems with NVIDIA GPUs

### 3. `requirements.txt` - Pip-only installation
```bash
pip install -r requirements.txt
```
- Pure pip installation without conda
- Works with any Python environment manager
- CPU-only by default

## Platform-Specific Recommendations

### Apple Silicon (M1/M2/M3)
- **Recommended**: `env.yaml`
- **Alternative**: `requirements.txt` (if Metal support issues)

### Linux with NVIDIA GPU
- **Recommended**: `deepml.yaml`
- **Alternative**: `requirements.txt` (if CUDA issues)

### Linux without GPU / Windows
- **Recommended**: `requirements.txt`
- Most cloud platforms have TensorFlow pre-installed

### Cloud Computing (Google Colab, AWS, etc.)
- **Recommended**: `requirements.txt`
- Most cloud platforms have TensorFlow pre-installed

## Verification

After setting up your environment, verify the installation:

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import Bio

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

## Troubleshooting

### Apple Silicon Issues
- If Metal support doesn't work, use `requirements.txt`
- Ensure you're using conda-forge channel

### CUDA Issues on Linux
- Check NVIDIA driver compatibility
- Use `requirements.txt` for CPU-only fallback
- Verify CUDA toolkit version matches TensorFlow requirements

### Memory Issues
- Reduce batch size in `config.py`
- Use CPU-only version for large datasets
- Consider data preprocessing optimizations