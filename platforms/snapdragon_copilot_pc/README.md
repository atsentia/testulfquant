# Snapdragon Copilot PC Quantization

4-bit quantization of GPT-OSS-20B optimized for Snapdragon Elite X processors with 32GB RAM.

## Overview

This implementation uses standard PyTorch quantization for optimal compatibility with Snapdragon Elite X processors. The 4-bit quantization reduces the model from ~42GB to ~10.5GB while maintaining good performance on CPU.

## Requirements

### Hardware
- **CPU**: Snapdragon Elite X processor
- **RAM**: 32GB (16GB available for model inference)
- **Storage**: ~15GB free space (original + quantized model)

### Software
- **OS**: Windows on ARM or Linux ARM64
- **Python**: 3.8+
- **PyTorch**: 1.13+ with ARM64 support

## Installation

```bash
# Install PyTorch for ARM64
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
pip install transformers safetensors accelerate
```

## Quick Start

### 1. Download GPT-OSS-20B Model

```bash
# Download model in SafeTensors format
huggingface-cli download openai/gpt-oss-20b \
  --local-dir ./models/gpt-oss-20b \
  --local-dir-use-symlinks False
```

### 2. Analyze Model (Optional)

```bash
# Check model structure and memory requirements
python ../../utils/safetensors_loader.py ./models/gpt-oss-20b
```

### 3. Run Quantization

```bash
# Dynamic quantization (recommended for Snapdragon)
python quantize_snapdragon.py \
  --model-path ./models/gpt-oss-20b \
  --output-path ./snapdragon_quantized \
  --method dynamic

# With performance benchmark
python quantize_snapdragon.py \
  --model-path ./models/gpt-oss-20b \
  --output-path ./snapdragon_quantized \
  --method dynamic \
  --benchmark
```

## Quantization Methods

### Dynamic Quantization (Recommended)
- **Pros**: No calibration needed, fast quantization, good performance
- **Cons**: Slightly lower accuracy than static quantization
- **Usage**: `--method dynamic`

### Static Quantization
- **Pros**: Better accuracy, optimal performance
- **Cons**: Requires calibration data, longer quantization time
- **Usage**: `--method static --calibration-samples 100`

## Performance Expectations

### Snapdragon Elite X Performance
- **Model Size**: 10.5GB (fits comfortably in 32GB RAM)
- **Inference Speed**: ~15-30 tokens/second (depends on sequence length)
- **Memory Usage**: ~12-14GB peak (including overhead)
- **Power Efficiency**: Optimized for ARM64 architecture

### Benchmarking

```bash
# Run comprehensive benchmark
python quantize_snapdragon.py \
  --model-path ./models/gpt-oss-20b \
  --output-path ./snapdragon_quantized \
  --benchmark
```

Expected results:
- **Compression**: 4x model size reduction
- **Speed**: Comparable to FP16 on similar hardware
- **Accuracy**: <2% degradation on most tasks

## Optimization Tips

### Memory Optimization
1. **Close other applications** before quantization
2. **Use swap file** if needed during quantization
3. **Monitor memory usage** with Task Manager/htop

### Performance Optimization
1. **Use CPU affinity** for consistent performance
2. **Disable CPU throttling** during inference
3. **Optimize Windows power settings** for performance

### Snapdragon-Specific
1. **Use ARM64-native PyTorch** for best performance
2. **Enable NEON optimizations** in PyTorch build
3. **Consider ONNX Runtime** for production deployment

## Deployment

### Standalone Application
```python
import torch
from transformers import AutoTokenizer

# Load quantized model
checkpoint = torch.load("snapdragon_quantized/quantized_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/gpt-oss-20b")

# Run inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
```

### Integration with Copilot PC
- Integrate with Windows Copilot framework
- Use WinRT APIs for system integration
- Optimize for battery life and thermal management

## Troubleshooting

### Memory Issues
- **Symptom**: Out of memory during quantization
- **Solution**: Close other applications, add swap space, or use a machine with more RAM

### Performance Issues
- **Symptom**: Slow inference
- **Solution**: Check CPU usage, disable background services, verify ARM64 PyTorch installation

### Compatibility Issues
- **Symptom**: Import errors or crashes
- **Solution**: Verify PyTorch ARM64 installation, check Python version compatibility

## Advanced Configuration

### Custom Quantization Settings
```python
from pytorch_4bit import PyTorch4BitQuantizer

quantizer = PyTorch4BitQuantizer(
    quantization_method="dynamic",
    dtype=torch.qint8,
    observer_type="minmax"
)
```

### Performance Profiling
```bash
# Profile with PyTorch profiler
python quantize_snapdragon.py --profile
```

## Integration Examples

### Windows Service
```python
# Example Windows service integration
import win32serviceutil
import win32service

class GPTService(win32serviceutil.ServiceFramework):
    _svc_name_ = "GPT-OSS-Quantized"
    _svc_display_name_ = "GPT-OSS Quantized Inference"
    
    def SvcDoRun(self):
        # Load and run quantized model
        pass
```

### REST API
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
# Load quantized model here

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.json['prompt']
    # Run inference and return results
    return jsonify({'generated_text': result})
```

## File Structure

```
snapdragon_copilot_pc/
├── pytorch_4bit.py          # Core quantization implementation
├── quantize_snapdragon.py   # Main quantization script
├── README.md                # This file
└── examples/                # Usage examples
    ├── simple_inference.py  # Basic inference example
    ├── batch_processing.py  # Batch inference example
    └── api_server.py         # REST API server
```

## Support

For Snapdragon-specific issues:
- Check Qualcomm Developer documentation
- Verify ARM64 PyTorch installation
- Monitor CPU temperature and throttling
- Use Windows Performance Toolkit for debugging