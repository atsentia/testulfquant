"""
Core ML 2-bit Quantization for iPhone Deployment

Optimized 2-bit quantization specifically for Apple's Core ML framework.
Target: GPT-OSS-20B (~42GB) -> ~5.25GB for iPhone deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import coremltools as ct
from coremltools.models.neural_network import NeuralNetwork
from coremltools.models.utils import save_spec


class CoreML2BitQuantizer:
    """
    2-bit quantization optimized for Apple's Core ML and Neural Engine.
    
    Uses 4 levels: {-1.5, -0.5, +0.5, +1.5} for optimal iPhone performance.
    Integrates with Core ML's quantization pipeline for Neural Engine acceleration.
    """
    
    def __init__(
        self,
        weight_threshold: float = 0.1,
        activation_bits: int = 8,
        use_neural_engine: bool = True,
        calibration_method: str = "entropy"
    ):
        """
        Initialize Core ML 2-bit quantizer.
        
        Args:
            weight_threshold: Threshold for weight clipping
            activation_bits: Bits for activation quantization (8 or 16)
            use_neural_engine: Whether to optimize for Neural Engine
            calibration_method: Calibration method ('entropy', 'minmax')
        """
        self.weight_threshold = weight_threshold
        self.activation_bits = activation_bits
        self.use_neural_engine = use_neural_engine
        self.calibration_method = calibration_method
        
        # 2-bit quantization levels optimized for Neural Engine
        self.quantization_levels = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        self.bits_per_weight = 2.0
    
    def entropy_calibration(self, weights: torch.Tensor) -> Tuple[float, float]:
        """
        Use entropy-based calibration for optimal quantization range.
        
        Args:
            weights: Weight tensor to calibrate
            
        Returns:
            Optimal min and max values for quantization
        """
        # Compute histogram
        hist, bin_edges = torch.histogram(weights.flatten(), bins=2048)
        hist = hist.float()
        
        # Find optimal threshold using KL divergence
        best_threshold = None
        min_kl_div = float('inf')
        
        for i in range(len(bin_edges) - 1):
            threshold = bin_edges[i].item()
            
            # Clamp weights to threshold
            clamped = torch.clamp(weights, -threshold, threshold)
            
            # Compute KL divergence (simplified)
            if torch.var(clamped) > 0:
                kl_div = torch.mean((weights - clamped) ** 2)
                
                if kl_div < min_kl_div:
                    min_kl_div = kl_div
                    best_threshold = threshold
        
        return -best_threshold, best_threshold
    
    def quantize_weights_2bit(
        self, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Quantize weights to 2-bit (4 levels) for Core ML.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Quantized weights and quantization parameters
        """
        if self.calibration_method == "entropy":
            w_min, w_max = self.entropy_calibration(weights)
        else:
            # Min-max calibration
            w_min = weights.min().item()
            w_max = weights.max().item()
        
        # Apply weight clipping
        if self.weight_threshold > 0:
            std = weights.std().item()
            clip_val = self.weight_threshold * std
            w_min = max(w_min, -clip_val)
            w_max = min(w_max, clip_val)
        
        # Make symmetric for Neural Engine optimization
        if self.use_neural_engine:
            abs_max = max(abs(w_min), abs(w_max))
            w_min, w_max = -abs_max, abs_max
        
        # Quantize to 4 levels
        scale = (w_max - w_min) / 3.0  # 3 intervals for 4 levels
        
        # Map weights to quantization levels
        normalized = (weights - w_min) / scale
        level_indices = torch.round(normalized).clamp(0, 3).long()
        
        # Get actual quantized values
        quantized = self.quantization_levels[level_indices]
        
        # Rescale to original range
        quantized = quantized * scale / 3.0 + (w_min + w_max) / 2
        
        quant_params = {
            'scale': scale,
            'w_min': w_min,
            'w_max': w_max,
            'levels': self.quantization_levels.tolist()
        }
        
        return quantized, quant_params
    
    def create_coreml_config(
        self, 
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> ct.PassPipeline:
        """
        Create Core ML optimization pipeline for 2-bit quantization.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape
            
        Returns:
            Core ML optimization pipeline
        """
        # Create quantization configuration
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,  # Use int8 for 2-bit storage
            granularity="per_channel"
        )
        
        # Configure for Neural Engine
        if self.use_neural_engine:
            config = ct.optimize.coreml.OptimizationConfig(
                global_config=ct.optimize.coreml.OpPalettizerConfig(
                    mode="kmeans",
                    lut_dtype=np.int8,
                    nbits=2
                )
            )
        else:
            config = ct.optimize.coreml.OptimizationConfig(
                global_config=op_config
            )
        
        return config
    
    def convert_to_coreml(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str
    ):
        """
        Convert quantized model to Core ML format.
        
        Args:
            model: Quantized PyTorch model
            input_shape: Shape of input tensor
            output_path: Path to save Core ML model
        """
        # Create sample input
        sample_input = torch.randint(0, 1000, input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, sample_input)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape)],
            minimum_deployment_target=ct.target.iOS15,  # Neural Engine support
            compute_units=ct.ComputeUnit.ALL if self.use_neural_engine else ct.ComputeUnit.CPU_ONLY
        )
        
        # Apply 2-bit quantization optimization
        config = self.create_coreml_config(model, input_shape)
        
        # Optimize for Neural Engine
        if self.use_neural_engine:
            optimized_model = ct.optimize.coreml.palettize_weights(
                coreml_model,
                config=config
            )
        else:
            optimized_model = ct.optimize.coreml.linear_quantize_weights(
                coreml_model,
                config=config
            )
        
        # Save the model
        optimized_model.save(output_path)
        print(f"Core ML model saved to {output_path}")
        
        return optimized_model
    
    def quantize_layer(self, layer: nn.Linear) -> Dict[str, Any]:
        """
        Quantize a single linear layer for Core ML.
        
        Args:
            layer: Linear layer to quantize
            
        Returns:
            Dictionary containing quantized weights and metadata
        """
        weights = layer.weight.data.clone()
        
        # Apply 2-bit quantization
        quantized_weights, quant_params = self.quantize_weights_2bit(weights)
        
        # Calculate statistics
        stats = self.calculate_statistics(weights, quantized_weights)
        
        return {
            'quantized_weights': quantized_weights,
            'quant_params': quant_params,
            'original_shape': weights.shape,
            'statistics': stats,
            'bias': layer.bias.data.clone() if layer.bias is not None else None,
            'quantization_mode': '2bit_coreml'
        }
    
    def calculate_statistics(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate quantization statistics."""
        mse = nn.MSELoss()(quantized, original).item()
        
        # Level distribution
        level_counts = {}
        for level in self.quantization_levels:
            count = (torch.abs(quantized - level) < 1e-6).sum().item()
            level_counts[f'level_{level.item():.1f}'] = count
        
        total_weights = quantized.numel()
        level_percentages = {
            k: 100.0 * v / total_weights for k, v in level_counts.items()
        }
        
        stats = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'compression_ratio': 32.0 / 2.0,  # 32-bit to 2-bit
            'bits_per_weight': 2.0,
            **level_percentages
        }
        
        return stats
    
    def quantize_model(self, model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Quantize all linear layers in a model for Core ML.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Dictionary mapping layer names to quantized weights
        """
        quantized_layers = {}
        skip_layers = ['embed', 'lm_head', 'wte', 'wpe']
        
        print("Quantizing model to 2-bit for Core ML deployment")
        
        for name, module in tqdm(model.named_modules(), desc="Quantizing layers"):
            if isinstance(module, nn.Linear):
                # Skip embedding and output layers
                if any(skip in name.lower() for skip in skip_layers):
                    print(f"Skipping {name} (embedding/output layer)")
                    continue
                
                print(f"Quantizing {name}: {module.weight.shape}")
                quantized_layers[name] = self.quantize_layer(module)
                
                # Print statistics
                stats = quantized_layers[name]['statistics']
                print(f"  RMSE: {stats['rmse']:.6f}")
                print(f"  Compression: {stats['compression_ratio']:.1f}x")
                
                # Print level distribution
                level_info = [f"{k}: {v:.1f}%" for k, v in stats.items() 
                             if k.startswith('level_')]
                print(f"  Distribution: {', '.join(level_info)}")
        
        return quantized_layers
    
    def estimate_model_size(self, quantized_layers: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Estimate Core ML model size after quantization.
        
        Args:
            quantized_layers: Dictionary of quantized layers
            
        Returns:
            Size estimates in MB
        """
        total_params = 0
        quantized_params = 0
        
        for layer_data in quantized_layers.values():
            shape = layer_data['original_shape']
            layer_params = shape[0] * shape[1]
            total_params += layer_params
            quantized_params += layer_params
        
        # Calculate sizes
        original_size_mb = total_params * 4 / (1024 * 1024)  # FP32
        quantized_size_mb = quantized_params * 0.25 / (1024 * 1024)  # 2-bit
        
        # Add overhead for Core ML format
        coreml_overhead = quantized_size_mb * 0.1  # ~10% overhead
        final_size_mb = quantized_size_mb + coreml_overhead
        
        return {
            'original_size_mb': original_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'coreml_size_mb': final_size_mb,
            'compression_ratio': original_size_mb / final_size_mb,
            'total_parameters': total_params,
            'quantized_parameters': quantized_params
        }


class CoreMLLinear2Bit(nn.Module):
    """
    2-bit Linear layer optimized for Core ML Neural Engine.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantization_levels: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use 2-bit quantization levels
        if quantization_levels is None:
            self.register_buffer('levels', torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        else:
            self.register_buffer('levels', quantization_levels)
        
        # Store quantized weights as indices (0-3)
        self.register_buffer('weight_indices', torch.zeros(
            out_features, in_features, dtype=torch.uint8
        ))
        
        # Weight scaling factors
        self.register_buffer('weight_scales', torch.ones(out_features))
        
        # Bias (kept in full precision)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def set_quantized_weights(
        self,
        weight_indices: torch.Tensor,
        scales: torch.Tensor
    ):
        """Set quantized weights from indices and scales."""
        self.weight_indices.data = weight_indices
        self.weight_scales.data = scales
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using 2-bit quantized weights."""
        # Reconstruct weights from indices
        weights = self.levels[self.weight_indices]  # Shape: [out_features, in_features]
        weights = weights * self.weight_scales.unsqueeze(1)
        
        # Standard linear operation
        output = torch.nn.functional.linear(x, weights, self.bias)
        
        return output


def save_coreml_quantized_model(
    quantized_layers: Dict[str, Dict[str, Any]],
    model_config: Dict,
    output_path: str
):
    """
    Save quantized model in Core ML compatible format.
    
    Args:
        quantized_layers: Dictionary of quantized layers
        model_config: Model configuration
        output_path: Output file path
    """
    import pickle
    
    save_data = {
        'quantized_layers': quantized_layers,
        'model_config': model_config,
        'quantization_type': '2bit_coreml',
        'target_platform': 'ios_neural_engine'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Core ML quantized model saved to {output_path}")