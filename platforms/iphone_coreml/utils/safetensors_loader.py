"""
SafeTensors Model Loading Utilities

Handles loading and preprocessing of GPT-OSS-20B model from SafeTensors format.
Optimized for memory efficiency during quantization process.
"""

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import mmap
import os


class SafeTensorsModelLoader:
    """
    Efficient loader for SafeTensors format models.
    
    Provides memory-mapped loading and preprocessing for quantization.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize SafeTensors loader.
        
        Args:
            model_path: Path to model directory containing SafeTensors files
        """
        self.model_path = Path(model_path)
        self.safetensors_files = self._find_safetensors_files()
        self.config = self._load_config()
        
    def _find_safetensors_files(self) -> List[Path]:
        """Find all SafeTensors files in the model directory."""
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            # Check for HuggingFace format
            safetensors_files = list(self.model_path.glob("model*.safetensors"))
        
        if not safetensors_files:
            raise FileNotFoundError(f"No SafeTensors files found in {self.model_path}")
        
        # Sort files for consistent loading order
        safetensors_files.sort()
        return safetensors_files
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information without loading weights."""
        total_size = sum(f.stat().st_size for f in self.safetensors_files)
        
        return {
            'model_type': self.config.get('model_type', 'unknown'),
            'hidden_size': self.config.get('hidden_size', 'unknown'),
            'num_layers': self.config.get('num_hidden_layers', 'unknown'),
            'vocab_size': self.config.get('vocab_size', 'unknown'),
            'total_size_gb': total_size / (1024**3),
            'safetensors_files': [f.name for f in self.safetensors_files],
            'num_files': len(self.safetensors_files)
        }
    
    def load_weights_memory_mapped(self) -> Dict[str, torch.Tensor]:
        """
        Load model weights using memory mapping for efficiency.
        
        Returns:
            Dictionary of parameter names to tensors
        """
        print(f"Loading weights from {len(self.safetensors_files)} SafeTensors files...")
        
        weights = {}
        
        for i, safetensors_file in enumerate(self.safetensors_files):
            print(f"Loading file {i+1}/{len(self.safetensors_files)}: {safetensors_file.name}")
            
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        
        print(f"✓ Loaded {len(weights)} parameters")
        return weights
    
    def load_weights_selective(
        self, 
        layer_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load only specific layers matching patterns.
        
        Args:
            layer_patterns: Patterns to include (e.g., ['transformer.h.', 'lm_head'])
            exclude_patterns: Patterns to exclude (e.g., ['embed', 'ln'])
            
        Returns:
            Dictionary of filtered parameter names to tensors
        """
        if layer_patterns is None:
            layer_patterns = ['transformer.h.']  # Default to transformer layers
        
        if exclude_patterns is None:
            exclude_patterns = ['embed', 'wte', 'wpe']  # Default exclusions
        
        weights = {}
        
        for safetensors_file in self.safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # Check inclusion patterns
                    include = any(pattern in key for pattern in layer_patterns)
                    
                    # Check exclusion patterns
                    exclude = any(pattern in key.lower() for pattern in exclude_patterns)
                    
                    if include and not exclude:
                        weights[key] = f.get_tensor(key)
        
        print(f"✓ Loaded {len(weights)} parameters (selective)")
        return weights
    
    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about model layers without loading weights.
        
        Returns:
            Dictionary with layer information
        """
        layer_info = {}
        
        for safetensors_file in self.safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_info = f.get_tensor(key)
                    layer_info[key] = {
                        'shape': list(tensor_info.shape),
                        'dtype': str(tensor_info.dtype),
                        'numel': tensor_info.numel(),
                        'size_mb': tensor_info.numel() * tensor_info.element_size() / (1024**2)
                    }
        
        return layer_info
    
    def check_memory_requirements(self) -> Dict[str, float]:
        """
        Estimate memory requirements for loading and quantization.
        
        Returns:
            Memory requirements in GB
        """
        layer_info = self.get_layer_info()
        
        # Calculate sizes
        original_size_gb = sum(info['size_mb'] for info in layer_info.values()) / 1024
        
        # Estimate quantization memory overhead (2x for safety)
        quantization_overhead_gb = original_size_gb * 2
        
        # Estimate final quantized sizes
        ternary_size_gb = original_size_gb * (1.58 / 32)  # 1.58-bit
        fourbit_size_gb = original_size_gb * (4 / 32)     # 4-bit
        twobit_size_gb = original_size_gb * (2 / 32)      # 2-bit
        
        return {
            'original_model_gb': original_size_gb,
            'quantization_peak_gb': quantization_overhead_gb,
            'ternary_final_gb': ternary_size_gb,
            'fourbit_final_gb': fourbit_size_gb,
            'twobit_final_gb': twobit_size_gb,
            'recommended_ram_gb': max(64, quantization_overhead_gb * 1.5)  # 50% safety margin
        }
    
    def preprocess_for_quantization(
        self,
        target_platform: str = "snapdragon",
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess SafeTensors for optimal quantization.
        
        Args:
            target_platform: Target platform ("snapdragon", "iphone", "generic")
            dtype: Target dtype for quantization
            
        Returns:
            Preprocessed weights dictionary
        """
        print(f"Preprocessing SafeTensors for {target_platform} quantization...")
        
        # Load weights based on platform requirements
        if target_platform == "snapdragon":
            # Load all linear layers, skip embeddings
            weights = self.load_weights_selective(
                layer_patterns=['transformer.h.', 'lm_head'],
                exclude_patterns=['embed', 'wte', 'wpe', 'ln_']
            )
        elif target_platform == "iphone":
            # More aggressive filtering for memory constraints
            weights = self.load_weights_selective(
                layer_patterns=['transformer.h.'],
                exclude_patterns=['embed', 'wte', 'wpe', 'ln_', 'lm_head']
            )
        else:
            # Load all weights
            weights = self.load_weights_memory_mapped()
        
        # Convert dtype if needed
        if dtype != torch.float32:
            print(f"Converting weights to {dtype}...")
            weights = {k: v.to(dtype) for k, v in weights.items()}
        
        return weights
    
    def validate_safetensors_integrity(self) -> bool:
        """
        Validate SafeTensors file integrity.
        
        Returns:
            True if all files are valid
        """
        try:
            for safetensors_file in self.safetensors_files:
                with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                    # Try to access first tensor
                    keys = list(f.keys())
                    if keys:
                        _ = f.get_tensor(keys[0])
            
            print("✓ All SafeTensors files are valid")
            return True
            
        except Exception as e:
            print(f"✗ SafeTensors validation failed: {e}")
            return False


def convert_pytorch_to_safetensors(
    pytorch_model_path: str,
    output_path: str,
    max_shard_size: str = "5GB"
):
    """
    Convert PyTorch model to SafeTensors format.
    
    Args:
        pytorch_model_path: Path to PyTorch model
        output_path: Output path for SafeTensors
        max_shard_size: Maximum size per shard
    """
    from transformers import AutoModelForCausalLM
    
    print(f"Converting PyTorch model to SafeTensors...")
    print(f"Input: {pytorch_model_path}")
    print(f"Output: {output_path}")
    
    # Load PyTorch model
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_model_path,
        torch_dtype=torch.float16,  # Use FP16 for smaller files
        low_cpu_mem_usage=True
    )
    
    # Save as SafeTensors
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size
    )
    
    print(f"✓ Model converted to SafeTensors: {output_path}")


def analyze_safetensors_model(model_path: str):
    """
    Analyze SafeTensors model and print detailed information.
    
    Args:
        model_path: Path to SafeTensors model directory
    """
    loader = SafeTensorsModelLoader(model_path)
    
    # Basic model info
    model_info = loader.get_model_info()
    print("=" * 60)
    print("SafeTensors Model Analysis")
    print("=" * 60)
    print(f"Model Type: {model_info['model_type']}")
    print(f"Hidden Size: {model_info['hidden_size']}")
    print(f"Layers: {model_info['num_layers']}")
    print(f"Vocab Size: {model_info['vocab_size']}")
    print(f"Total Size: {model_info['total_size_gb']:.2f} GB")
    print(f"SafeTensors Files: {model_info['num_files']}")
    
    # Memory requirements
    memory_req = loader.check_memory_requirements()
    print(f"\nMemory Requirements:")
    print(f"  Original Model: {memory_req['original_model_gb']:.2f} GB")
    print(f"  Quantization Peak: {memory_req['quantization_peak_gb']:.2f} GB")
    print(f"  Recommended RAM: {memory_req['recommended_ram_gb']:.0f} GB")
    
    print(f"\nQuantized Sizes:")
    print(f"  4-bit (Snapdragon): {memory_req['fourbit_final_gb']:.2f} GB")
    print(f"  2-bit (iPhone): {memory_req['twobit_final_gb']:.2f} GB")
    print(f"  1.58-bit (Ternary): {memory_req['ternary_final_gb']:.2f} GB")
    
    # Layer breakdown
    layer_info = loader.get_layer_info()
    linear_layers = {k: v for k, v in layer_info.items() 
                    if 'weight' in k and len(v['shape']) == 2}
    
    print(f"\nLinear Layers: {len(linear_layers)}")
    total_linear_params = sum(info['numel'] for info in linear_layers.values())
    print(f"Linear Parameters: {total_linear_params:,}")
    
    # Validate integrity
    loader.validate_safetensors_integrity()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python safetensors_loader.py <model_path>")
        sys.exit(1)
    
    analyze_safetensors_model(sys.argv[1])