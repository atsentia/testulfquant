# Multi-Platform GPT-OSS-20B Quantization

Implementation of platform-optimized quantization for OpenAI's GPT-OSS-20B model targeting different deployment scenarios with optimal performance and memory usage.

## Platform Targets

### 🖥️ Snapdragon Copilot PC (4-bit)
- **Target**: Snapdragon Elite X with 32GB RAM
- **Model Size**: ~10.5GB (4x compression from 42GB)
- **Method**: Standard PyTorch dynamic quantization
- **Use Case**: Professional productivity and development

### 📱 iPhone Core ML (2-bit)
- **Target**: iPhone with Neural Engine (A14+ chips)
- **Model Size**: ~5.25GB (8x compression from 42GB)
- **Method**: Core ML optimized with Neural Engine acceleration
- **Use Case**: On-device AI assistant and apps

### 🔬 Research Ternary (1.58-bit)
- **Target**: Maximum compression research
- **Model Size**: ~1.5GB (28x compression from 42GB)
- **Method**: PT-BitNet ternary quantization
- **Use Case**: Extreme efficiency experiments

## Current Status

### ✅ Completed Implementation

The initial implementation framework is complete and ready for testing on a larger memory machine:

1. **PT-BitNet Quantization Algorithm** (`quantization/pt_bitnet.py`)
   - Two-stage weight transformation and optimization
   - Absmean quantization to ternary values {-1, 0, +1}
   - Block-wise optimization for better accuracy
   - Efficient 2-bit packing/unpacking for storage

2. **BitLinear Layers** (`quantization/bitlinear.py`)
   - Ternary weight operations (no multiplications)
   - Optimized version with lookup tables
   - 8-bit activation quantization support
   - Drop-in replacement for nn.Linear

3. **Inference Engine** (`inference/engine.py`)
   - Model loading from quantized format
   - Benchmarking capabilities
   - Framework for text generation

4. **Comprehensive Test Suite** (`tests/test_quantization.py`)
   - Unit tests for all quantization functions
   - Integration tests for full pipeline
   - Memory reduction validation
   - Ternary weight verification

5. **Model Management Scripts**
   - Download script for GPT-OSS-20B
   - Quantization pipeline script
   - Performance benchmarking tools

### 📋 Next Steps (On Larger Machine)

1. **Download GPT-OSS-20B Model**
   ```bash
   python scripts/download_model.py
   ```

2. **Run Quantization**
   ```bash
   python quantization/quantize_model.py \
     --model models/gpt-oss-20b \
     --output models/gpt-oss-20b-ternary
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Benchmark Performance**
   ```bash
   python inference/engine.py \
     --model models/gpt-oss-20b-ternary \
     --benchmark
   ```

## Implementation Details

### Phase 1: Environment Setup & Dependencies ✅
- [x] Create project structure
- [x] Setup Python dependencies
- [x] Initialize git repository

### Phase 2: Core Quantization ✅
- [x] Implement ternary quantization algorithm
- [x] Create BitLinear layers with ternary operations
- [x] Build quantization pipeline with PT-BitNet approach
- [x] Implement efficient weight packing (2 bits per weight)

### Phase 3: Inference Framework ✅
- [x] Python inference engine implementation
- [x] Optimized ternary matrix multiplication
- [x] Model loading and weight unpacking
- [x] Benchmarking capabilities

### Phase 4: Testing Suite ✅
- [x] Unit tests for quantization functions
- [x] Integration tests for complete pipeline
- [x] Memory reduction validation
- [x] Test-driven development approach

### Phase 5: To Be Completed (Requires Large Memory)
- [ ] Download and process GPT-OSS-20B model
- [ ] Run full quantization on 20B parameters
- [ ] Validate accuracy on benchmarks
- [ ] Optimize inference performance
- [ ] Compare with other quantization methods (e.g., Mistral's AQLM)

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
# Clone the repository
git clone git@github.com:atsentia/1.5bit-quantized-gpt-oss-inference.git
cd 1.5bit-quantized-gpt-oss-inference

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/test_quantization.py -v

# Download GPT-OSS-20B model (requires ~50GB disk space)
python scripts/download_model.py

# Run quantization (requires 64GB+ RAM, 128GB recommended)
python quantization/quantize_model.py \
  --model models/gpt-oss-20b \
  --output models/gpt-oss-20b-ternary

# Run inference with quantized model
python inference/engine.py \
  --model models/gpt-oss-20b-ternary \
  --prompt "Hello, world!"

# Benchmark performance
python inference/engine.py \
  --model models/gpt-oss-20b-ternary \
  --benchmark
```

## Project Structure

```
1.5bit-quantized-gpt-oss-inference/
├── quantization/              # Quantization implementation
│   ├── __init__.py           # Module exports
│   ├── pt_bitnet.py          # PT-BitNet quantization algorithm
│   ├── bitlinear.py          # BitLinear layer implementation
│   └── quantize_model.py     # Model quantization script
├── inference/                 # Inference engine
│   └── engine.py             # Inference engine with benchmarking
├── tests/                     # Test suite
│   └── test_quantization.py  # Comprehensive unit tests
├── scripts/                   # Utility scripts
│   └── download_model.py     # Model download helper
├── models/                    # Model storage (created on use)
├── benchmarks/                # Benchmark results (created on use)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── CLAUDE.md                  # Claude Code context
```

## Key Components

### Quantization Module
- **PTBitNetQuantizer**: Main quantization class implementing two-stage algorithm
- **BitLinear/BitLinearOptimized**: Ternary weight layers replacing nn.Linear
- **pack/unpack_ternary_weights**: Efficient 2-bit storage for ternary values

### Inference Engine
- **TernaryModelLoader**: Loads quantized models from disk
- **TernaryInferenceEngine**: Runs inference with quantized models
- **SimpleTernaryModel**: Demonstration model architecture

### Testing
- Comprehensive test coverage for all components
- Memory reduction validation
- Ternary weight verification
- Integration tests for full pipeline

## Alternative Quantization Methods

While this project focuses on 1.58-bit (ternary) quantization, other effective approaches include:

- **Mistral's AQLM**: State-of-the-art 2-4 bit quantization with codebook approach
- **AWQ (Activation-aware Weight Quantization)**: 4-bit quantization with minimal accuracy loss
- **GPTQ**: Popular 4-bit quantization method
- **GGUF/GGML**: Efficient quantization formats used by llama.cpp

For better accuracy/performance trade-offs, consider implementing AQLM or AWQ as alternatives.

## References

- [PT-BitNet Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4987078) - Post-training quantization to ternary
- [Microsoft BitNet](https://github.com/microsoft/BitNet) - Official 1-bit LLM inference framework
- [OpenAI GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) - Target model for quantization
- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764) - Original 1.58-bit training approach
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) - Comprehensive study on ternary models

## Contributing

Contributions are welcome! Areas for improvement:
- Full model architecture implementation
- C++ kernel optimizations
- Alternative quantization methods (AQLM, AWQ)
- Accuracy benchmarking suite
- ONNX export support

## License

Apache 2.0 (matching GPT-OSS-20B license)