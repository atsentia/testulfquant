"""
Multi-bit Quantization: 2-bit, 4-bit, and 1.58-bit (Ternary) Support

This module implements quantization to different bit-widths:
- 2-bit: 4 levels {-1.5, -0.5, +0.5, +1.5} for iPhone/CoreML
- 4-bit: 16 levels for Snapdragon Elite X 
- 1.58-bit: Ternary {-1, 0, +1} (existing implementation)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from tqdm import tqdm
from enum import Enum


class QuantizationMode(Enum):
    """Supported quantization modes."""
    TERNARY = "1.58bit"  # {-1, 0, +1}
    TWO_BIT = "2bit"     # 4 levels
    FOUR_BIT = "4bit"    # 16 levels


class MultiBitQuantizer:
    """
    Universal quantizer supporting 2-bit, 4-bit, and ternary quantization.
    
    Optimized for different deployment targets:
    - 2-bit: iPhone/CoreML with ultra-low memory
    - 4-bit: Snapdragon Elite X with balanced performance
    - Ternary: Maximum compression with acceptable quality loss
    """
    
    def __init__(
        self,
        mode: QuantizationMode = QuantizationMode.FOUR_BIT,
        block_size: int = 128,
        optimization_steps: int = 100,
        learning_rate: float = 0.01,
        use_symmetric: bool = True,
        calibration_method: str = "minmax"
    ):
        """
        Initialize multi-bit quantizer.
        
        Args:
            mode: Quantization bit-width mode
            block_size: Size of blocks for block-wise optimization
            optimization_steps: Number of optimization steps per block
            learning_rate: Learning rate for weight optimization
            use_symmetric: Whether to use symmetric quantization
            calibration_method: Method for determining quantization range ('minmax', 'percentile')
        """
        self.mode = mode
        self.block_size = block_size
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate
        self.use_symmetric = use_symmetric
        self.calibration_method = calibration_method
        
        # Set quantization parameters based on mode
        self._setup_quantization_params()
    
    def _setup_quantization_params(self):
        """Setup quantization parameters based on bit-width."""
        if self.mode == QuantizationMode.TERNARY:
            self.num_levels = 3
            self.levels = torch.tensor([-1.0, 0.0, 1.0])
            self.bits_per_weight = 1.58
        elif self.mode == QuantizationMode.TWO_BIT:
            self.num_levels = 4
            self.levels = torch.tensor([-1.5, -0.5, 0.5, 1.5])
            self.bits_per_weight = 2.0
        elif self.mode == QuantizationMode.FOUR_BIT:
            self.num_levels = 16
            if self.use_symmetric:
                # Symmetric 4-bit: -7.5 to +7.5 in steps of 1
                self.levels = torch.linspace(-7.5, 7.5, 16)
            else:
                # Asymmetric 4-bit: will be computed per-tensor
                self.levels = None
            self.bits_per_weight = 4.0
    
    def compute_quantization_range(
        self, 
        weights: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute quantization range based on calibration method.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Min and max values for quantization range
        """
        if self.calibration_method == "minmax":
            w_min = weights.min().item()
            w_max = weights.max().item()
        elif self.calibration_method == "percentile":
            w_min = torch.quantile(weights, 0.01).item()
            w_max = torch.quantile(weights, 0.99).item()
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        if self.use_symmetric:
            # Make symmetric around zero
            abs_max = max(abs(w_min), abs(w_max))
            w_min, w_max = -abs_max, abs_max
        
        return w_min, w_max
    
    def quantize_weights(
        self, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Quantize weights to specified bit-width.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Quantized weights and quantization parameters
        """
        if self.mode == QuantizationMode.TERNARY:
            return self._quantize_ternary(weights)
        elif self.mode == QuantizationMode.TWO_BIT:
            return self._quantize_2bit(weights)
        elif self.mode == QuantizationMode.FOUR_BIT:
            return self._quantize_4bit(weights)
    
    def _quantize_ternary(
        self, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Quantize to ternary values {-1, 0, +1}."""
        # Use absmean quantization like original implementation
        scale = weights.abs().mean()
        if scale == 0:
            return torch.zeros_like(weights), {"scale": 1.0}
        
        normalized = weights / scale
        
        # Apply ternary quantization with threshold
        quantized = torch.sign(normalized)
        threshold = torch.quantile(normalized.abs(), 0.95)
        quantized[normalized.abs() < threshold] = 0
        
        return quantized, {"scale": scale.item()}
    
    def _quantize_2bit(
        self, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Quantize to 2-bit (4 levels)."""
        w_min, w_max = self.compute_quantization_range(weights)
        
        # Compute scale and zero point
        scale = (w_max - w_min) / (self.num_levels - 1)
        zero_point = -w_min / scale
        
        # Quantize
        q_weights = torch.round(weights / scale + zero_point)
        q_weights = torch.clamp(q_weights, 0, self.num_levels - 1)
        
        # Map to actual quantized values
        level_mapping = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        quantized = level_mapping[q_weights.long()]
        
        return quantized, {
            "scale": scale,
            "zero_point": zero_point,
            "w_min": w_min,
            "w_max": w_max
        }
    
    def _quantize_4bit(
        self, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Quantize to 4-bit (16 levels)."""
        w_min, w_max = self.compute_quantization_range(weights)
        
        # Compute scale and zero point
        if self.use_symmetric:
            scale = max(abs(w_min), abs(w_max)) / 7.5  # Map to [-7.5, 7.5]
            zero_point = 0
        else:
            scale = (w_max - w_min) / (self.num_levels - 1)
            zero_point = -w_min / scale
        
        # Quantize
        if self.use_symmetric:
            q_weights = torch.round(weights / scale)
            q_weights = torch.clamp(q_weights, -7.5, 7.5)
            quantized = q_weights * scale
        else:
            q_weights = torch.round(weights / scale + zero_point)
            q_weights = torch.clamp(q_weights, 0, self.num_levels - 1)
            quantized = (q_weights - zero_point) * scale
        
        return quantized, {
            "scale": scale,
            "zero_point": zero_point,
            "w_min": w_min,
            "w_max": w_max,
            "symmetric": self.use_symmetric
        }
    
    def optimize_quantized_weights(
        self,
        original_weights: torch.Tensor,
        quantized_weights: torch.Tensor,
        quant_params: Dict[str, float]
    ) -> torch.Tensor:
        """
        Optimize quantized weights using gradient-based fine-tuning.
        
        Args:
            original_weights: Original weight tensor
            quantized_weights: Initial quantized weights
            quant_params: Quantization parameters
            
        Returns:
            Optimized quantized weights
        """
        if self.optimization_steps == 0:
            return quantized_weights
        
        # Create learnable parameters
        if self.mode == QuantizationMode.TERNARY:
            # For ternary, optimize the scale factor
            scale = torch.tensor(quant_params["scale"], requires_grad=True)
            optimizer = torch.optim.Adam([scale], lr=self.learning_rate)
            
            for _ in range(self.optimization_steps):
                optimizer.zero_grad()
                
                # Reconstruct weights
                reconstructed = quantized_weights * scale
                loss = nn.MSELoss()(reconstructed, original_weights)
                
                loss.backward()
                optimizer.step()
                
                # Keep scale positive
                with torch.no_grad():
                    scale.data = torch.abs(scale.data)
            
            return quantized_weights * scale.detach()
        
        else:
            # For multi-bit, optimize within quantization levels
            optimized = quantized_weights.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([optimized], lr=self.learning_rate)
            
            for _ in range(self.optimization_steps):
                optimizer.zero_grad()
                
                # Reconstruction loss
                loss = nn.MSELoss()(optimized, original_weights)
                
                # Regularization to keep values at quantization levels
                if self.mode == QuantizationMode.TWO_BIT:
                    reg_loss = self._compute_level_regularization(optimized, self.levels)
                elif self.mode == QuantizationMode.FOUR_BIT:
                    if self.use_symmetric:
                        reg_loss = self._compute_level_regularization(optimized, self.levels)
                    else:
                        # For asymmetric, regularize to quantized values
                        reg_loss = 0
                
                total_loss = loss + 0.1 * reg_loss
                total_loss.backward()
                optimizer.step()
                
                # Project back to valid quantization levels
                with torch.no_grad():
                    if self.mode == QuantizationMode.TWO_BIT:
                        optimized.data = self._project_to_levels(optimized.data, self.levels)
                
            return optimized.detach()
    
    def _compute_level_regularization(
        self, 
        weights: torch.Tensor, 
        levels: torch.Tensor
    ) -> torch.Tensor:
        """Compute regularization loss to keep weights at quantization levels."""
        # Find closest level for each weight
        distances = torch.abs(weights.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0))
        closest_levels = levels[torch.argmin(distances, dim=-1)]
        
        # Regularization loss
        return nn.MSELoss()(weights, closest_levels)
    
    def _project_to_levels(
        self, 
        weights: torch.Tensor, 
        levels: torch.Tensor
    ) -> torch.Tensor:
        """Project weights to nearest quantization levels."""
        distances = torch.abs(weights.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0))
        closest_indices = torch.argmin(distances, dim=-1)
        return levels[closest_indices]
    
    def quantize_layer(self, layer: nn.Linear) -> Dict[str, Any]:
        """
        Quantize a single linear layer.
        
        Args:
            layer: Linear layer to quantize
            
        Returns:
            Dictionary containing quantized weights and metadata
        """
        weights = layer.weight.data.clone()
        
        # Apply initial quantization
        quantized_weights, quant_params = self.quantize_weights(weights)
        
        # Optimize quantized weights
        if self.optimization_steps > 0:
            quantized_weights = self.optimize_quantized_weights(
                weights, quantized_weights, quant_params
            )
        
        # Calculate statistics
        stats = self.calculate_statistics(weights, quantized_weights)
        
        return {
            'quantized_weights': quantized_weights,
            'quant_params': quant_params,
            'original_shape': weights.shape,
            'statistics': stats,
            'bias': layer.bias.data.clone() if layer.bias is not None else None,
            'mode': self.mode.value
        }
    
    def calculate_statistics(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate quantization statistics.
        
        Args:
            original: Original weight tensor
            quantized: Quantized weight tensor
            
        Returns:
            Dictionary of statistics
        """
        mse = nn.MSELoss()(quantized, original).item()
        
        # Calculate compression ratio
        original_bits = 32  # FP32
        compression_ratio = original_bits / self.bits_per_weight
        
        stats = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'compression_ratio': compression_ratio,
            'bits_per_weight': self.bits_per_weight,
            'mode': self.mode.value
        }
        
        # Add mode-specific statistics
        if self.mode == QuantizationMode.TERNARY:
            ternary_stats = self._calculate_ternary_stats(quantized)
            stats.update(ternary_stats)
        elif self.mode in [QuantizationMode.TWO_BIT, QuantizationMode.FOUR_BIT]:
            level_stats = self._calculate_level_stats(quantized)
            stats.update(level_stats)
        
        return stats
    
    def _calculate_ternary_stats(self, weights: torch.Tensor) -> Dict[str, float]:
        """Calculate statistics for ternary weights."""
        neg_ones = (weights == -1).sum().item()
        zeros = (weights == 0).sum().item()
        pos_ones = (weights == 1).sum().item()
        total = weights.numel()
        
        return {
            'percent_neg_one': 100 * neg_ones / total,
            'percent_zero': 100 * zeros / total,
            'percent_pos_one': 100 * pos_ones / total,
        }
    
    def _calculate_level_stats(self, weights: torch.Tensor) -> Dict[str, float]:
        """Calculate statistics for multi-bit weights."""
        unique_values = torch.unique(weights)
        return {
            'unique_levels': len(unique_values),
            'weight_min': weights.min().item(),
            'weight_max': weights.max().item(),
            'weight_std': weights.std().item()
        }
    
    def quantize_model(self, model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Quantize all linear layers in a model.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Dictionary mapping layer names to quantized weights
        """
        quantized_layers = {}
        
        print(f"Quantizing model to {self.mode.value} ({self.bits_per_weight} bits per weight)")
        
        for name, module in tqdm(model.named_modules(), desc="Quantizing layers"):
            if isinstance(module, nn.Linear):
                # Skip embedding and output layers
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    print(f"Skipping {name} (embedding/output layer)")
                    continue
                
                print(f"Quantizing {name}: {module.weight.shape}")
                quantized_layers[name] = self.quantize_layer(module)
                
                # Print statistics
                stats = quantized_layers[name]['statistics']
                print(f"  RMSE: {stats['rmse']:.6f}")
                print(f"  Compression: {stats['compression_ratio']:.1f}x")
                
                if self.mode == QuantizationMode.TERNARY:
                    print(f"  Distribution: -1: {stats['percent_neg_one']:.1f}%, "
                          f"0: {stats['percent_zero']:.1f}%, "
                          f"+1: {stats['percent_pos_one']:.1f}%")
                else:
                    print(f"  Levels: {stats['unique_levels']}, "
                          f"Range: [{stats['weight_min']:.3f}, {stats['weight_max']:.3f}]")
        
        return quantized_layers


def pack_multi_bit_weights(
    weights: torch.Tensor, 
    mode: QuantizationMode
) -> torch.Tensor:
    """
    Pack quantized weights efficiently based on bit-width.
    
    Args:
        weights: Quantized weight tensor
        mode: Quantization mode
        
    Returns:
        Packed weight tensor
    """
    if mode == QuantizationMode.TERNARY:
        return pack_ternary_weights(weights)
    elif mode == QuantizationMode.TWO_BIT:
        return pack_2bit_weights(weights)
    elif mode == QuantizationMode.FOUR_BIT:
        return pack_4bit_weights(weights)


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights using 2 bits per weight."""
    flat = weights.flatten()
    encoded = torch.zeros_like(flat, dtype=torch.uint8)
    
    # Encode: -1->0, 0->1, +1->2
    encoded[flat == -1] = 0
    encoded[flat == 0] = 1
    encoded[flat == 1] = 2
    
    # Pack 4 weights per byte
    num_bytes = (flat.numel() + 3) // 4
    packed = torch.zeros(num_bytes, dtype=torch.uint8)
    
    for i in range(0, flat.numel(), 4):
        byte_idx = i // 4
        for j in range(min(4, flat.numel() - i)):
            packed[byte_idx] |= encoded[i + j] << (j * 2)
    
    return packed


def pack_2bit_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit weights using 2 bits per weight."""
    flat = weights.flatten()
    
    # Map to indices: -1.5->0, -0.5->1, +0.5->2, +1.5->3
    level_to_idx = {-1.5: 0, -0.5: 1, 0.5: 2, 1.5: 3}
    encoded = torch.zeros_like(flat, dtype=torch.uint8)
    
    for i, weight in enumerate(flat):
        encoded[i] = level_to_idx[weight.item()]
    
    # Pack 4 weights per byte
    num_bytes = (flat.numel() + 3) // 4
    packed = torch.zeros(num_bytes, dtype=torch.uint8)
    
    for i in range(0, flat.numel(), 4):
        byte_idx = i // 4
        for j in range(min(4, flat.numel() - i)):
            packed[byte_idx] |= encoded[i + j] << (j * 2)
    
    return packed


def pack_4bit_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit weights using 4 bits per weight."""
    flat = weights.flatten()
    
    # Quantize to 4-bit indices (0-15)
    w_min, w_max = flat.min(), flat.max()
    scale = (w_max - w_min) / 15
    encoded = torch.round((flat - w_min) / scale).clamp(0, 15).to(torch.uint8)
    
    # Pack 2 weights per byte
    num_bytes = (flat.numel() + 1) // 2
    packed = torch.zeros(num_bytes, dtype=torch.uint8)
    
    for i in range(0, flat.numel(), 2):
        byte_idx = i // 2
        packed[byte_idx] = encoded[i]
        if i + 1 < flat.numel():
            packed[byte_idx] |= encoded[i + 1] << 4
    
    return packed