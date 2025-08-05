"""
PT-BitNet: Post-Training Quantization to 1.58-bit (Ternary Weights)

This module implements the PT-BitNet quantization algorithm that converts
pre-trained model weights to ternary values {-1, 0, +1} without any training.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm


class PTBitNetQuantizer:
    """
    Post-training quantization to 1.58-bit using PT-BitNet approach.
    
    This quantizer implements a two-stage algorithm:
    1. Weight distribution transformation to be quantization-friendly
    2. Block-wise weight optimization
    """
    
    def __init__(
        self,
        block_size: int = 128,
        optimization_steps: int = 100,
        learning_rate: float = 0.01,
        threshold_percentile: float = 0.95
    ):
        """
        Initialize PT-BitNet quantizer.
        
        Args:
            block_size: Size of blocks for block-wise optimization
            optimization_steps: Number of optimization steps per block
            learning_rate: Learning rate for weight optimization
            threshold_percentile: Percentile for determining zero threshold
        """
        self.block_size = block_size
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
    
    def absmean_quantize(self, weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Quantize weights to ternary values using absmean quantization.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Quantized weights and scaling factor
        """
        # Calculate scaling factor using absolute mean
        scale = weights.abs().mean()
        
        # Avoid division by zero
        if scale == 0:
            return torch.zeros_like(weights), 1.0
        
        # Normalize weights
        normalized = weights / scale
        
        # Apply ternary quantization
        # Values close to 0 become 0, others become -1 or +1
        quantized = torch.sign(normalized)
        threshold = torch.quantile(normalized.abs(), self.threshold_percentile / 100)
        quantized[normalized.abs() < threshold] = 0
        
        return quantized, scale.item()
    
    def transform_distribution(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: Transform weight distribution to be quantization-friendly.
        
        Args:
            weights: Original weight tensor
            
        Returns:
            Transformed weight tensor
        """
        # Center the weights
        centered = weights - weights.mean()
        
        # Apply soft clipping to reduce outliers
        std = centered.std()
        clip_value = 3 * std
        transformed = torch.clamp(centered, -clip_value, clip_value)
        
        # Normalize to unit variance
        transformed = transformed / transformed.std()
        
        return transformed
    
    def optimize_block(
        self,
        original_block: torch.Tensor,
        quantized_block: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        Stage 2: Optimize quantized weights for a single block.
        
        Args:
            original_block: Original weight block
            quantized_block: Initial quantized block
            scale: Quantization scale factor
            
        Returns:
            Optimized quantized block
        """
        # Create a copy for optimization
        optimized = quantized_block.clone().float().requires_grad_(True)
        
        # Define optimizer
        optimizer = torch.optim.Adam([optimized], lr=self.learning_rate)
        
        for _ in range(self.optimization_steps):
            optimizer.zero_grad()
            
            # Reconstruction loss
            reconstructed = optimized * scale
            loss = nn.MSELoss()(reconstructed, original_block)
            
            # Add regularization to maintain ternary structure
            reg_loss = (optimized.abs() - 1).abs().mean()
            total_loss = loss + 0.1 * reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Project back to ternary values
            with torch.no_grad():
                optimized.data = torch.sign(optimized.data)
                # Re-apply threshold for zeros
                threshold = torch.quantile(optimized.abs(), self.threshold_percentile / 100)
                optimized.data[optimized.abs() < threshold] = 0
        
        return optimized.detach()
    
    def quantize_layer(self, layer: nn.Linear) -> Dict[str, Any]:
        """
        Quantize a single linear layer to ternary weights.
        
        Args:
            layer: Linear layer to quantize
            
        Returns:
            Dictionary containing quantized weights and metadata
        """
        weights = layer.weight.data.clone()
        
        # Stage 1: Transform distribution
        transformed = self.transform_distribution(weights)
        
        # Initial quantization
        quantized, scale = self.absmean_quantize(transformed)
        
        # Stage 2: Block-wise optimization
        if weights.numel() > self.block_size:
            # Flatten weights for block processing
            flat_weights = weights.flatten()
            flat_quantized = quantized.flatten()
            
            # Process blocks
            num_blocks = (flat_weights.numel() + self.block_size - 1) // self.block_size
            optimized_blocks = []
            
            for i in range(num_blocks):
                start_idx = i * self.block_size
                end_idx = min((i + 1) * self.block_size, flat_weights.numel())
                
                original_block = flat_weights[start_idx:end_idx]
                quantized_block = flat_quantized[start_idx:end_idx]
                
                optimized_block = self.optimize_block(
                    original_block,
                    quantized_block,
                    scale
                )
                optimized_blocks.append(optimized_block)
            
            # Concatenate and reshape
            quantized = torch.cat(optimized_blocks).reshape(weights.shape)
        
        # Calculate statistics
        stats = self.calculate_statistics(weights, quantized * scale)
        
        return {
            'quantized_weights': quantized,
            'scale': scale,
            'original_shape': weights.shape,
            'statistics': stats,
            'bias': layer.bias.data.clone() if layer.bias is not None else None
        }
    
    def calculate_statistics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate quantization statistics.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            
        Returns:
            Dictionary of statistics
        """
        mse = nn.MSELoss()(reconstructed, original).item()
        
        # Calculate percentage of each ternary value
        ternary = torch.sign(reconstructed)
        neg_ones = (ternary == -1).sum().item()
        zeros = (ternary == 0).sum().item()
        pos_ones = (ternary == 1).sum().item()
        total = ternary.numel()
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'percent_neg_one': 100 * neg_ones / total,
            'percent_zero': 100 * zeros / total,
            'percent_pos_one': 100 * pos_ones / total,
            'compression_ratio': 32 / 1.58  # 32-bit to 1.58-bit
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
                print(f"  Distribution: -1: {stats['percent_neg_one']:.1f}%, "
                      f"0: {stats['percent_zero']:.1f}%, "
                      f"+1: {stats['percent_pos_one']:.1f}%")
        
        return quantized_layers


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights efficiently using 2 bits per weight.
    
    Encoding:
    - -1 -> 00
    -  0 -> 01
    - +1 -> 10
    - Reserved -> 11
    
    Args:
        weights: Ternary weight tensor
        
    Returns:
        Packed weight tensor
    """
    # Flatten weights
    flat = weights.flatten()
    
    # Encode ternary values to 2-bit representation
    encoded = torch.zeros_like(flat, dtype=torch.uint8)
    encoded[flat == -1] = 0b00
    encoded[flat == 0] = 0b01
    encoded[flat == 1] = 0b10
    
    # Pack 4 weights into each byte
    num_bytes = (flat.numel() + 3) // 4
    packed = torch.zeros(num_bytes, dtype=torch.uint8)
    
    for i in range(0, flat.numel(), 4):
        byte_idx = i // 4
        for j in range(min(4, flat.numel() - i)):
            packed[byte_idx] |= encoded[i + j] << (j * 2)
    
    return packed


def unpack_ternary_weights(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Unpack ternary weights from 2-bit representation.
    
    Args:
        packed: Packed weight tensor
        shape: Original weight shape
        
    Returns:
        Unpacked ternary weight tensor
    """
    # Calculate total number of weights
    num_weights = shape.numel()
    
    # Unpack weights
    unpacked = torch.zeros(num_weights, dtype=torch.float32)
    
    for i in range(num_weights):
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        
        # Extract 2-bit value
        value = (packed[byte_idx] >> bit_offset) & 0b11
        
        # Decode to ternary
        if value == 0b00:
            unpacked[i] = -1
        elif value == 0b01:
            unpacked[i] = 0
        elif value == 0b10:
            unpacked[i] = 1
    
    return unpacked.reshape(shape)