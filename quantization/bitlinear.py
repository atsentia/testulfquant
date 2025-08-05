"""
BitLinear: Ternary Linear Layer Implementation

This module implements BitLinear layers that use ternary weights {-1, 0, +1}
for efficient inference without multiplication operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class BitLinear(nn.Module):
    """
    Linear layer with ternary weights for 1.58-bit quantization.
    
    Replaces standard matrix multiplication with efficient ternary operations.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_activations: bool = True,
        activation_bits: int = 8
    ):
        """
        Initialize BitLinear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
            quantize_activations: Whether to quantize activations
            activation_bits: Number of bits for activation quantization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_activations = quantize_activations
        self.activation_bits = activation_bits
        
        # Ternary weights stored as int8 for efficiency
        self.register_buffer('weight', torch.zeros(
            out_features, in_features, dtype=torch.int8
        ))
        
        # Weight scale factor
        self.register_buffer('weight_scale', torch.ones(1))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Activation quantization parameters
        if quantize_activations:
            self.register_buffer('activation_scale', torch.ones(1))
            self.register_buffer('activation_zero_point', torch.zeros(1))
    
    def set_ternary_weights(
        self,
        ternary_weights: torch.Tensor,
        scale: float
    ):
        """
        Set the ternary weights and scale.
        
        Args:
            ternary_weights: Tensor with values in {-1, 0, +1}
            scale: Weight scaling factor
        """
        self.weight.data = ternary_weights.to(torch.int8)
        self.weight_scale.data = torch.tensor([scale])
    
    def quantize_activations_absmax(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Quantize activations using absmax quantization.
        
        Args:
            x: Input activations
            
        Returns:
            Quantized activations and scale factor
        """
        # Calculate scale using absolute maximum
        scale = x.abs().max() / (2 ** (self.activation_bits - 1) - 1)
        
        # Avoid division by zero
        if scale == 0:
            scale = 1.0
        
        # Quantize
        x_int = torch.round(x / scale).clamp(
            -(2 ** (self.activation_bits - 1)),
            2 ** (self.activation_bits - 1) - 1
        )
        
        return x_int, scale
    
    def ternary_matmul(
        self,
        input: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient matrix multiplication with ternary weights.
        
        This replaces multiplication with addition/subtraction operations.
        
        Args:
            input: Input tensor
            weight: Ternary weight tensor
            
        Returns:
            Output tensor
        """
        # Separate weights by value
        weight_float = weight.float()
        pos_mask = (weight_float == 1)
        neg_mask = (weight_float == -1)
        
        # Compute output using only additions and subtractions
        output = torch.zeros(
            input.shape[0], weight.shape[0],
            device=input.device, dtype=input.dtype
        )
        
        # Add contributions from +1 weights
        if pos_mask.any():
            output += F.linear(input, pos_mask.float(), None)
        
        # Subtract contributions from -1 weights
        if neg_mask.any():
            output -= F.linear(input, neg_mask.float(), None)
        
        return output
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BitLinear layer.
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor
        """
        # Quantize activations if enabled
        if self.quantize_activations and not self.training:
            input_quantized, act_scale = self.quantize_activations_absmax(input)
            input_for_compute = input_quantized
        else:
            input_for_compute = input
            act_scale = 1.0
        
        # Perform ternary matrix multiplication
        output = self.ternary_matmul(input_for_compute, self.weight)
        
        # Apply scales
        output = output * self.weight_scale * act_scale
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def extra_repr(self) -> str:
        """String representation of layer."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'quantize_activations={self.quantize_activations}, '
                f'activation_bits={self.activation_bits}')


class BitLinearOptimized(BitLinear):
    """
    Optimized BitLinear layer with lookup table operations.
    
    This version uses precomputed lookup tables for even faster inference.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Precompute lookup tables for common patterns
        self._init_lookup_tables()
    
    def _init_lookup_tables(self):
        """Initialize lookup tables for fast ternary operations."""
        # Create lookup table for 4-weight patterns (81 combinations)
        # Each pattern maps to its contribution
        self.register_buffer('lut_4', torch.zeros(81, dtype=torch.float32))
        
        # Generate all possible 4-weight patterns
        values = [-1, 0, 1]
        idx = 0
        for w0 in values:
            for w1 in values:
                for w2 in values:
                    for w3 in values:
                        # Store the pattern's contribution
                        pattern = torch.tensor([w0, w1, w2, w3])
                        self.lut_4[idx] = pattern.sum()
                        idx += 1
    
    def ternary_matmul_optimized(
        self,
        input: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized ternary matrix multiplication using lookup tables.
        
        Args:
            input: Input tensor
            weight: Ternary weight tensor
            
        Returns:
            Output tensor
        """
        batch_size = input.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        
        # Process weights in groups of 4 for LUT efficiency
        group_size = 4
        num_groups = (in_features + group_size - 1) // group_size
        
        output = torch.zeros(
            batch_size, out_features,
            device=input.device, dtype=input.dtype
        )
        
        for out_idx in range(out_features):
            for group_idx in range(num_groups):
                start_idx = group_idx * group_size
                end_idx = min(start_idx + group_size, in_features)
                
                # Get weight group
                weight_group = weight[out_idx, start_idx:end_idx]
                
                # Get corresponding input group
                input_group = input[:, start_idx:end_idx]
                
                # Compute contribution using ternary operations
                if weight_group.shape[0] == group_size:
                    # Use lookup table for full groups
                    pattern_idx = self._encode_pattern(weight_group)
                    contribution = self.lut_4[pattern_idx] * input_group.mean(dim=1)
                else:
                    # Handle partial groups directly
                    pos_mask = (weight_group == 1)
                    neg_mask = (weight_group == -1)
                    contribution = (
                        input_group[:, pos_mask].sum(dim=1) -
                        input_group[:, neg_mask].sum(dim=1)
                    )
                
                output[:, out_idx] += contribution
        
        return output
    
    def _encode_pattern(self, pattern: torch.Tensor) -> int:
        """
        Encode a ternary pattern to lookup table index.
        
        Args:
            pattern: Ternary pattern tensor
            
        Returns:
            Index into lookup table
        """
        # Convert {-1, 0, 1} to {0, 1, 2}
        encoded = pattern + 1
        
        # Compute base-3 index
        idx = 0
        for i, val in enumerate(encoded):
            idx += val * (3 ** i)
        
        return int(idx)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using optimized operations.
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor
        """
        # Use optimized matmul if weights are properly sized
        if self.weight.shape[1] % 4 == 0:
            # Quantize activations if enabled
            if self.quantize_activations and not self.training:
                input_quantized, act_scale = self.quantize_activations_absmax(input)
                input_for_compute = input_quantized
            else:
                input_for_compute = input
                act_scale = 1.0
            
            # Perform optimized ternary matrix multiplication
            output = self.ternary_matmul_optimized(input_for_compute, self.weight)
            
            # Apply scales
            output = output * self.weight_scale * act_scale
            
            # Add bias if present
            if self.bias is not None:
                output = output + self.bias
            
            return output
        else:
            # Fall back to standard implementation
            return super().forward(input)


def replace_linear_with_bitlinear(
    model: nn.Module,
    skip_layers: Optional[list] = None
) -> nn.Module:
    """
    Replace all Linear layers in a model with BitLinear layers.
    
    Args:
        model: Model to modify
        skip_layers: List of layer names to skip
        
    Returns:
        Modified model with BitLinear layers
    """
    if skip_layers is None:
        skip_layers = ['embed', 'lm_head']
    
    for name, child in model.named_children():
        # Skip specified layers
        if any(skip in name.lower() for skip in skip_layers):
            continue
        
        if isinstance(child, nn.Linear):
            # Create BitLinear layer with same dimensions
            bitlinear = BitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None
            )
            
            # Copy bias if present
            if child.bias is not None:
                bitlinear.bias.data = child.bias.data.clone()
            
            # Replace the layer
            setattr(model, name, bitlinear)
        else:
            # Recursively replace in child modules
            replace_linear_with_bitlinear(child, skip_layers)
    
    return model