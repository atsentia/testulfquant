# CLAUDE.md

This file provides guidance to Claude Code when working with the 1.5-bit quantized GPT-OSS-20B inference project.

## Project Overview

This project implements post-training quantization of OpenAI's GPT-OSS-20B model to 1.58-bit (ternary weights) using the PT-BitNet approach. The goal is to reduce model size from ~42GB to ~1.5GB while maintaining performance.

## Key Technical Details

### Quantization Approach
- **PT-BitNet**: Post-training quantization without any fine-tuning
- **Ternary Weights**: {-1, 0, +1} using absmean quantization
- **Activations**: 8-bit quantization using absmax
- **Two-Stage Process**:
  1. Weight distribution transformation
  2. Block-wise weight optimization

### Model Details
- **Source Model**: GPT-OSS-20B (21B total, 3.6B active parameters)
- **Original Size**: ~42GB
- **Target Size**: ~1.5GB
- **Expected Speedup**: 2-3x on CPU
- **Accuracy Target**: <5% degradation

## Implementation Priority

1. **Core Quantization** (`quantization/pt_bitnet.py`)
   - Implement absmean quantization function
   - Two-stage weight transformation
   - Block-wise optimization

2. **BitLinear Layer** (`quantization/bitlinear.py`)
   - Replace nn.Linear with ternary operations
   - Handle forward pass with lookup tables
   - Maintain gradient flow for calibration

3. **Model Loading** (`scripts/download_model.py`)
   - Download from HuggingFace Hub
   - Memory-mapped loading for large model
   - Architecture analysis

4. **Quantization Pipeline** (`quantization/quantize_model.py`)
   - Load pre-trained weights
   - Apply PT-BitNet quantization
   - Save in efficient format

5. **Inference Engine** (`inference/engine.py`)
   - Load quantized model
   - Efficient ternary operations
   - Token generation

## Testing Requirements

### Unit Tests
- Test quantization preserves weight distribution
- Verify ternary values {-1, 0, +1}
- Test BitLinear layer correctness

### Integration Tests
- End-to-end quantization pipeline
- Model loading and saving
- Inference output validation

### Performance Tests
- Memory usage measurement
- Inference speed benchmarks
- Accuracy metrics (perplexity)

## Important Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/performance.py

# Quantize model (once implemented)
python quantization/quantize_model.py \
  --model openai/gpt-oss-20b \
  --output models/gpt-oss-20b-ternary

# Run inference (once implemented)
python inference/run_inference.py \
  --model models/gpt-oss-20b-ternary \
  --prompt "Test prompt"
```

## Resource Considerations

- **Development Machine**: Needs 64GB+ RAM for quantization
- **Large Memory Machine**: 128GB+ recommended for faster quantization
- **Inference**: Can run on 16GB after quantization

## Key Papers and References

1. **PT-BitNet**: Post-training quantization to ternary weights
2. **BitNet b1.58**: Original 1.58-bit training approach
3. **GPT-OSS-20B**: OpenAI's open-source model

## Implementation Notes

- Focus on test-driven development
- Validate each component independently
- Use memory-mapped files for large model handling
- Implement efficient ternary storage (2 bits per weight)
- Keep residual connections in full precision
- Don't quantize embedding and output layers

## Next Steps on Larger Machine

1. Download full GPT-OSS-20B model
2. Run quantization pipeline
3. Validate quantized model accuracy
4. Optimize inference performance
5. Run comprehensive benchmarks