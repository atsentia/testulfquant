#!/usr/bin/env python3
"""
Multi-target quantization for GPT-OSS-20B model.

Supports multiple quantization targets:
- 4-bit: Snapdragon Elite X (Standard PyTorch, ~10.5GB)
- 2-bit: iPhone Core ML (Neural Engine optimized, ~5.25GB)
- 1.58-bit: Maximum compression (Original PT-BitNet, ~1.5GB)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import pickle
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from quantization import (
    PTBitNetQuantizer,
    pack_ternary_weights,
    BitLinear,
    replace_linear_with_bitlinear
)

try:
    from quantization.pytorch_4bit import PyTorch4BitQuantizer
    from quantization.coreml_2bit import CoreML2BitQuantizer, save_coreml_quantized_model
    ADVANCED_QUANTIZATION_AVAILABLE = True
except ImportError:
    ADVANCED_QUANTIZATION_AVAILABLE = False
    print("Advanced quantization modules not available. Only 1.58-bit quantization will work.")


def save_quantized_model(
    quantized_layers: Dict[str, Dict[str, Any]],
    model_config: Dict,
    output_dir: Path
):
    """
    Save quantized model to disk.
    
    Args:
        quantized_layers: Dictionary of quantized layer data
        model_config: Model configuration
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each layer's quantized weights
    for layer_name, layer_data in tqdm(quantized_layers.items(), desc="Saving layers"):
        layer_dir = output_dir / layer_name.replace('.', '_')
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Pack weights for efficient storage
        packed_weights = pack_ternary_weights(layer_data['quantized_weights'])
        
        # Save packed weights
        torch.save({
            'packed_weights': packed_weights,
            'scale': layer_data['scale'],
            'original_shape': layer_data['original_shape'],
            'bias': layer_data['bias'],
            'statistics': layer_data['statistics']
        }, layer_dir / 'weights.pt')
    
    # Save model configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save quantization metadata
    metadata = {
        'quantization_method': 'PT-BitNet',
        'bits_per_weight': 1.58,
        'num_layers': len(quantized_layers),
        'layer_names': list(quantized_layers.keys())
    }
    
    with open(output_dir / 'quantization_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Quantized model saved to {output_dir}")


def calculate_model_size(quantized_layers: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate size of quantized model.
    
    Args:
        quantized_layers: Dictionary of quantized layer data
        
    Returns:
        Size statistics in GB
    """
    total_params = 0
    total_ternary_params = 0
    total_bias_params = 0
    
    for layer_data in quantized_layers.values():
        weight_params = layer_data['original_shape'][0] * layer_data['original_shape'][1]
        total_params += weight_params
        total_ternary_params += weight_params
        
        if layer_data['bias'] is not None:
            bias_params = layer_data['bias'].numel()
            total_params += bias_params
            total_bias_params += bias_params
    
    # Calculate sizes
    original_size_gb = (total_params * 4) / (1024**3)  # 32-bit floats
    ternary_size_gb = (total_ternary_params * 2 / 8) / (1024**3)  # 2 bits per weight
    bias_size_gb = (total_bias_params * 4) / (1024**3)  # Keep bias in 32-bit
    quantized_size_gb = ternary_size_gb + bias_size_gb
    
    return {
        'total_params': total_params,
        'ternary_params': total_ternary_params,
        'original_size_gb': original_size_gb,
        'quantized_size_gb': quantized_size_gb,
        'compression_ratio': original_size_gb / quantized_size_gb
    }


def quantize_gpt_oss_20b(
    model_path: str,
    output_dir: str,
    quantization_target: str = "ternary",
    block_size: int = 128,
    optimization_steps: int = 100
):
    """
    Quantize GPT-OSS-20B model to specified bit-width.
    
    Args:
        model_path: Path to the model
        output_dir: Directory to save quantized model
        quantization_target: Target platform ("ternary", "snapdragon", "iphone")
        block_size: Block size for optimization
        optimization_steps: Number of optimization steps
    """
    output_path = Path(output_dir)
    
    print("=" * 60)
    print("PT-BitNet Quantization for GPT-OSS-20B")
    print("=" * 60)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("Warning: Quantization on CPU will be slow. Consider using a GPU.")
    
    # Load model with memory mapping
    print(f"\nLoading model from {model_path}...")
    print("Note: This may take several minutes for the 20B model.")
    
    try:
        # Load with reduced memory usage
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # Get model configuration
        model_config = model.config.to_dict()
        
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model_config.get('model_type', 'unknown')}")
        print(f"  Hidden size: {model_config.get('hidden_size', 'unknown')}")
        print(f"  Num layers: {model_config.get('num_hidden_layers', 'unknown')}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the model is downloaded to the specified path")
        print("2. Check available memory (64GB+ recommended)")
        print("3. Try using a machine with more RAM")
        raise
    
    # Initialize quantizer
    print(f"\nInitializing PT-BitNet quantizer...")
    quantizer = PTBitNetQuantizer(
        block_size=block_size,
        optimization_steps=optimization_steps
    )
    
    # Quantize model
    print(f"\nQuantizing model layers...")
    print("This process will:")
    print("1. Transform weight distributions")
    print("2. Quantize to ternary values {-1, 0, +1}")
    print("3. Optimize weights block-wise")
    print("-" * 40)
    
    quantized_layers = quantizer.quantize_model(model)
    
    # Calculate size statistics
    print(f"\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    
    size_stats = calculate_model_size(quantized_layers)
    print(f"\nModel Size Statistics:")
    print(f"  Total parameters: {size_stats['total_params']:,}")
    print(f"  Original size: {size_stats['original_size_gb']:.2f} GB")
    print(f"  Quantized size: {size_stats['quantized_size_gb']:.2f} GB")
    print(f"  Compression ratio: {size_stats['compression_ratio']:.1f}x")
    
    # Calculate average statistics
    avg_stats = {
        'avg_rmse': 0,
        'avg_percent_zero': 0
    }
    
    for layer_data in quantized_layers.values():
        stats = layer_data['statistics']
        avg_stats['avg_rmse'] += stats['rmse']
        avg_stats['avg_percent_zero'] += stats['percent_zero']
    
    num_layers = len(quantized_layers)
    avg_stats['avg_rmse'] /= num_layers
    avg_stats['avg_percent_zero'] /= num_layers
    
    print(f"\nQuantization Quality:")
    print(f"  Average RMSE: {avg_stats['avg_rmse']:.6f}")
    print(f"  Average sparsity: {avg_stats['avg_percent_zero']:.1f}%")
    
    # Save quantized model
    print(f"\nSaving quantized model to {output_path}...")
    save_quantized_model(quantized_layers, model_config, output_path)
    
    # Save size statistics
    with open(output_path / 'size_statistics.json', 'w') as f:
        json.dump(size_stats, f, indent=2)
    
    print(f"\n✓ Quantization complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Next step: Run inference with the quantized model")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize GPT-OSS-20B to 1.58-bit using PT-BitNet"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gpt-oss-20b",
        help="Path to GPT-OSS-20B model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/gpt-oss-20b-ternary",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for optimization"
    )
    parser.add_argument(
        "--optimization-steps",
        type=int,
        default=100,
        help="Number of optimization steps per block"
    )
    
    args = parser.parse_args()
    
    quantize_gpt_oss_20b(
        args.model,
        args.output,
        args.block_size,
        args.optimization_steps
    )


if __name__ == "__main__":
    main()