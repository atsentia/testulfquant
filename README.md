# 1.5-Bit Quantized GPT-OSS-20B Inference

Implementation of post-training quantization to 1.58-bit (ternary weights) for OpenAI's GPT-OSS-20B model using the PT-BitNet approach, enabling efficient inference with minimal accuracy loss.

## Project Goals

- Quantize GPT-OSS-20B from ~42GB to ~1.5GB using ternary weights {-1, 0, +1}
- Achieve 2-3x inference speedup on CPU
- Maintain accuracy with <5% degradation on key metrics
- No fine-tuning required - pure post-training quantization

## Implementation Plan

### Phase 1: Environment Setup & Dependencies ✅
- [x] Create project structure
- [x] Setup Python dependencies

### Phase 2: Model Download & Preparation
- [ ] Download GPT-OSS-20B from HuggingFace
- [ ] Analyze model architecture
- [ ] Map quantizable components

### Phase 3: PT-BitNet Quantization Implementation
- [ ] Implement ternary quantization algorithm
  - Absmean quantization for weights → {-1, 0, +1}
  - 8-bit quantization for activations
  - Two-stage weight transformation
- [ ] Create BitLinear layers
- [ ] Build quantization pipeline

### Phase 4: Inference Engine Development
- [ ] Python inference implementation
- [ ] Optimized ternary operations
- [ ] Optional: C++ kernel optimization

### Phase 5: Comprehensive Testing Suite
- [ ] Unit tests for quantization
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Accuracy validation

### Phase 6: Optimization & Validation
- [ ] Memory optimization (2 bits per weight)
- [ ] Speed optimization
- [ ] Accuracy validation against original model

## Technical Approach

### PT-BitNet Post-Training Quantization
PT-BitNet enables quantization without any training or fine-tuning:
1. **Stage 1**: Transform weight distribution to be quantization-friendly
2. **Stage 2**: Block-wise weight optimization
3. **Result**: Ternary weights {-1, 0, +1} with minimal accuracy loss

### Key Features
- No training required - works on pre-trained models
- Scales to large models (tested up to 70B parameters)
- Significant memory reduction (~96% for weights)
- Maintains model capabilities

## Resource Requirements

### Development
- **Memory**: 64GB RAM minimum (128GB+ recommended for quantization)
- **Storage**: ~50GB for original + quantized models
- **GPU**: Optional but beneficial for faster processing

### Inference (After Quantization)
- **Memory**: ~16GB RAM for quantized model
- **CPU**: Modern x86_64 or ARM processor
- **Storage**: ~1.5GB for quantized model

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model (to be implemented)
python scripts/download_model.py

# Run quantization (to be implemented)
python quantization/quantize_model.py --model gpt-oss-20b --output models/gpt-oss-20b-ternary

# Run inference (to be implemented)
python inference/run_inference.py --model models/gpt-oss-20b-ternary --prompt "Hello, world!"
```

## Project Structure

```
1.5bit-quantized-gpt-oss-inference/
├── quantization/       # Quantization implementation
│   ├── pt_bitnet.py   # Main quantization algorithm
│   └── bitlinear.py   # BitLinear layer implementation
├── inference/          # Inference engine
│   └── engine.py      # Python inference implementation
├── tests/             # Test suite
│   ├── test_quantization.py
│   └── test_inference.py
├── benchmarks/        # Performance testing
│   └── performance.py
├── models/            # Model storage
├── scripts/           # Utility scripts
│   └── download_model.py
└── requirements.txt   # Python dependencies
```

## References

- [PT-BitNet Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4987078)
- [Microsoft BitNet](https://github.com/microsoft/BitNet)
- [OpenAI GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764)

## License

Apache 2.0 (matching GPT-OSS-20B license)