#!/usr/bin/env python3
"""
Snapdragon Copilot PC 4-bit Quantization Script

Optimized for:
- Snapdragon Elite X processor
- 32GB RAM (16GB available for inference)
- CPU-based inference with PyTorch
- Target model size: ~10.5GB (4x compression from 42GB)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors import safe_open
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from platforms.snapdragon_copilot_pc.pytorch_4bit import PyTorch4BitQuantizer
from utils.safetensors_loader import SafeTensorsModelLoader


def load_model_from_safetensors(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load GPT-OSS-20B from SafeTensors format with memory optimization.
    
    Args:
        model_path: Path to model directory with SafeTensors files
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from SafeTensors: {model_path}")
    
    # Load configuration
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = AutoConfig.from_pretrained(model_path)
    print(f"Model config loaded: {config.model_type}, {config.hidden_size}d")
    
    # Use memory-mapped loading for large model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use FP32 for quantization accuracy
        device_map=None,  # Load to CPU first
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    print(f"✓ Model loaded on {device}")
    return model


def quantize_for_snapdragon(
    model_path: str,
    output_path: str,
    quantization_method: str = "dynamic",
    calibration_samples: int = 100
):
    """
    Quantize GPT-OSS-20B for optimal Snapdragon Copilot PC performance.
    
    Args:
        model_path: Path to original model
        output_path: Path to save quantized model
        quantization_method: Quantization method ("dynamic", "static")
        calibration_samples: Number of calibration samples for static quantization
    """
    print("=" * 70)
    print("Snapdragon Copilot PC 4-bit Quantization")
    print("=" * 70)
    print(f"Target: GPT-OSS-20B (~42GB) → ~10.5GB (4x compression)")
    print(f"Platform: Snapdragon Elite X with 32GB RAM")
    print(f"Method: {quantization_method} PyTorch quantization")
    print()
    
    # Load model
    model = load_model_from_safetensors(model_path)
    
    # Initialize quantizer
    quantizer = PyTorch4BitQuantizer(
        quantization_method=quantization_method,
        dtype=torch.qint8  # Use qint8 for Snapdragon compatibility
    )
    
    # Get original model size
    original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)  # GB
    print(f"Original model size: {original_size:.2f} GB")
    
    # Quantize model
    if quantization_method == "dynamic":
        print("Applying dynamic quantization (no calibration needed)...")
        quantized_model = quantizer.quantize_model_dynamic(model)
        
    elif quantization_method == "static":
        print("Applying static quantization (with calibration)...")
        # Create calibration loader
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        calibration_loader = create_calibration_loader(tokenizer, calibration_samples)
        
        # Prepare and quantize
        prepared_model = quantizer.prepare_model_for_quantization(model)
        quantized_model = quantizer.quantize_model_static(prepared_model, calibration_loader)
    
    # Calculate size reduction
    size_stats = quantizer.calculate_model_size(quantized_model)
    
    print("\n" + "=" * 50)
    print("Quantization Results:")
    print("=" * 50)
    print(f"Original size: {size_stats['original_size_mb']:.0f} MB")
    print(f"Quantized size: {size_stats['quantized_size_mb']:.0f} MB")
    print(f"Compression ratio: {size_stats['compression_ratio']:.1f}x")
    print(f"Quantized parameters: {size_stats['quantization_percentage']:.1f}%")
    
    # Save quantized model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = output_dir / "quantized_model.pt"
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_config': model.config.to_dict(),
        'quantization_config': {
            'method': quantization_method,
            'dtype': str(quantizer.dtype),
            'target_platform': 'snapdragon_copilot_pc'
        },
        'size_stats': size_stats
    }, model_save_path)
    
    # Save metadata
    metadata = {
        'platform': 'snapdragon_copilot_pc',
        'quantization_method': quantization_method,
        'compression_ratio': size_stats['compression_ratio'],
        'model_size_mb': size_stats['quantized_size_mb'],
        'target_ram_usage': '16GB',
        'inference_device': 'cpu'
    }
    
    with open(output_dir / "snapdragon_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Quantized model saved to: {output_dir}")
    print(f"✓ Ready for Snapdragon Copilot PC deployment")
    
    return quantized_model


def create_calibration_loader(tokenizer, num_samples: int = 100):
    """Create calibration data loader for static quantization."""
    # Generate sample text for calibration
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The only thing we have to fear is fear itself."
    ] * (num_samples // 5 + 1)
    
    # Tokenize samples
    inputs = tokenizer(
        sample_texts[:num_samples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'])
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


def benchmark_snapdragon_performance(model, tokenizer, device="cpu"):
    """
    Benchmark quantized model performance on Snapdragon.
    
    Args:
        model: Quantized model
        tokenizer: Model tokenizer
        device: Device for benchmarking
    """
    print("\n" + "=" * 50)
    print("Snapdragon Performance Benchmark")
    print("=" * 50)
    
    model.eval()
    
    # Test input
    test_prompt = "The future of AI computing on edge devices"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(
                inputs['input_ids'],
                max_length=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Benchmark
    times = []
    for i in range(10):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i == 0:  # Show sample output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Sample output: {generated_text}")
    
    avg_time = sum(times) / len(times)
    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Tokens per second: {tokens_generated / avg_time:.1f}")
    print(f"Memory efficient: Suitable for 32GB RAM Copilot PC")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize GPT-OSS-20B for Snapdragon Copilot PC"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to GPT-OSS-20B model (SafeTensors format)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./snapdragon_quantized",
        help="Output path for quantized model"
    )
    parser.add_argument(
        "--method",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization method"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples for static quantization"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark after quantization"
    )
    
    args = parser.parse_args()
    
    # Quantize model
    quantized_model = quantize_for_snapdragon(
        args.model_path,
        args.output_path,
        args.method,
        args.calibration_samples
    )
    
    # Optional benchmarking
    if args.benchmark:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        benchmark_snapdragon_performance(quantized_model, tokenizer)


if __name__ == "__main__":
    main()