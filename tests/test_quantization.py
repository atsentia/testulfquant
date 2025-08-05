"""
Unit tests for PT-BitNet quantization.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from quantization import (
    PTBitNetQuantizer,
    pack_ternary_weights,
    unpack_ternary_weights,
    BitLinear
)


class TestPTBitNetQuantizer:
    """Test suite for PTBitNetQuantizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = PTBitNetQuantizer()
        torch.manual_seed(42)
    
    def test_absmean_quantize(self):
        """Test absmean quantization to ternary values."""
        # Create test weights
        weights = torch.randn(10, 10)
        
        # Quantize
        quantized, scale = self.quantizer.absmean_quantize(weights)
        
        # Check that all values are ternary
        unique_values = torch.unique(quantized)
        assert all(v in [-1, 0, 1] for v in unique_values)
        
        # Check scale is positive
        assert scale > 0
        
        # Check that reconstruction is reasonable
        reconstructed = quantized * scale
        mse = nn.MSELoss()(reconstructed, weights)
        assert mse < weights.var()  # Should be better than zero approximation
    
    def test_transform_distribution(self):
        """Test weight distribution transformation."""
        # Create weights with outliers
        weights = torch.randn(100, 100)
        weights[0, 0] = 100  # Add outlier
        
        # Transform
        transformed = self.quantizer.transform_distribution(weights)
        
        # Check that outliers are handled
        assert transformed.max() < 10
        assert transformed.min() > -10
        
        # Check normalization
        assert abs(transformed.std() - 1.0) < 0.1
    
    def test_quantize_layer(self):
        """Test quantization of a linear layer."""
        # Create test layer
        layer = nn.Linear(64, 32)
        
        # Quantize
        result = self.quantizer.quantize_layer(layer)
        
        # Check output structure
        assert 'quantized_weights' in result
        assert 'scale' in result
        assert 'statistics' in result
        
        # Check quantized weights are ternary
        quantized = result['quantized_weights']
        unique_values = torch.unique(quantized)
        assert all(v in [-1, 0, 1] for v in unique_values)
        
        # Check statistics
        stats = result['statistics']
        assert 'mse' in stats
        assert 'percent_zero' in stats
        assert stats['percent_neg_one'] + stats['percent_zero'] + stats['percent_pos_one'] == pytest.approx(100)
    
    def test_zero_weights(self):
        """Test handling of zero weights."""
        # Create zero weights
        weights = torch.zeros(10, 10)
        
        # Quantize
        quantized, scale = self.quantizer.absmean_quantize(weights)
        
        # Should handle gracefully
        assert torch.all(quantized == 0)
        assert scale == 1.0
    
    def test_block_optimization(self):
        """Test block-wise optimization."""
        # Create test weights
        original = torch.randn(16)
        quantized = torch.sign(original)
        scale = original.abs().mean()
        
        # Optimize
        optimized = self.quantizer.optimize_block(original, quantized, scale)
        
        # Check that output is still ternary
        unique_values = torch.unique(optimized)
        assert all(v in [-1, 0, 1] for v in unique_values)
        
        # Check that optimization improves reconstruction
        original_mse = nn.MSELoss()(quantized * scale, original)
        optimized_mse = nn.MSELoss()(optimized * scale, original)
        assert optimized_mse <= original_mse + 0.01  # Allow small numerical difference


class TestPackingUnpacking:
    """Test suite for weight packing/unpacking."""
    
    def test_pack_unpack_identity(self):
        """Test that pack-unpack is identity operation."""
        # Create ternary weights
        weights = torch.tensor([[-1, 0, 1, -1],
                                [0, 1, -1, 0],
                                [1, 1, 0, -1]], dtype=torch.float32)
        
        # Pack and unpack
        packed = pack_ternary_weights(weights)
        unpacked = unpack_ternary_weights(packed, weights.shape)
        
        # Check identity
        assert torch.all(weights == unpacked)
    
    def test_packing_efficiency(self):
        """Test that packing reduces memory usage."""
        # Create large ternary tensor
        size = 1000
        weights = torch.randint(-1, 2, (size, size), dtype=torch.float32)
        
        # Pack weights
        packed = pack_ternary_weights(weights)
        
        # Check compression ratio
        original_bytes = weights.numel() * 4  # 32-bit float
        packed_bytes = packed.numel()
        compression_ratio = original_bytes / packed_bytes
        
        # Should achieve roughly 16x compression (32 bits to 2 bits)
        assert compression_ratio > 10
    
    def test_edge_cases(self):
        """Test packing with various sizes."""
        for size in [1, 3, 7, 15, 16, 17, 100]:
            weights = torch.randint(-1, 2, (size,), dtype=torch.float32)
            packed = pack_ternary_weights(weights)
            unpacked = unpack_ternary_weights(packed, weights.shape)
            assert torch.all(weights == unpacked)


class TestBitLinear:
    """Test suite for BitLinear layer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_bitlinear_creation(self):
        """Test BitLinear layer creation."""
        layer = BitLinear(64, 32, bias=True)
        
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.bias is not None
        assert layer.weight.shape == (32, 64)
    
    def test_set_ternary_weights(self):
        """Test setting ternary weights."""
        layer = BitLinear(10, 5)
        
        # Create ternary weights
        weights = torch.randint(-1, 2, (5, 10), dtype=torch.float32)
        scale = 0.5
        
        # Set weights
        layer.set_ternary_weights(weights, scale)
        
        # Check weights are set correctly
        assert torch.all(layer.weight == weights.to(torch.int8))
        assert layer.weight_scale.item() == scale
    
    def test_forward_pass(self):
        """Test forward pass through BitLinear."""
        layer = BitLinear(16, 8)
        
        # Set ternary weights
        weights = torch.sign(torch.randn(8, 16))
        layer.set_ternary_weights(weights, 1.0)
        
        # Forward pass
        input_tensor = torch.randn(4, 16)
        output = layer(input_tensor)
        
        # Check output shape
        assert output.shape == (4, 8)
        
        # Check that output is not zero
        assert not torch.all(output == 0)
    
    def test_ternary_matmul(self):
        """Test ternary matrix multiplication."""
        layer = BitLinear(4, 2)
        
        # Simple test case
        input_tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        weight = torch.tensor([[1, -1, 0, 1],
                              [0, 1, -1, 0]], dtype=torch.float32)
        
        # Compute
        output = layer.ternary_matmul(input_tensor, weight)
        
        # Expected: 
        # Row 1: 1*1 + 2*(-1) + 3*0 + 4*1 = 1 - 2 + 0 + 4 = 3
        # Row 2: 1*0 + 2*1 + 3*(-1) + 4*0 = 0 + 2 - 3 + 0 = -1
        expected = torch.tensor([[3, -1]], dtype=torch.float32)
        
        assert torch.allclose(output, expected)
    
    def test_activation_quantization(self):
        """Test activation quantization."""
        layer = BitLinear(10, 5, quantize_activations=True, activation_bits=8)
        
        # Test input
        x = torch.randn(2, 10) * 10
        
        # Quantize
        x_quant, scale = layer.quantize_activations_absmax(x)
        
        # Check that values are within 8-bit range
        assert x_quant.min() >= -128
        assert x_quant.max() <= 127
        
        # Check reconstruction
        x_reconstructed = x_quant * scale
        relative_error = (x - x_reconstructed).abs() / (x.abs() + 1e-6)
        assert relative_error.mean() < 0.1  # Less than 10% average error
    
    def test_no_bias(self):
        """Test BitLinear without bias."""
        layer = BitLinear(10, 5, bias=False)
        
        assert layer.bias is None
        
        # Forward pass should work
        weights = torch.sign(torch.randn(5, 10))
        layer.set_ternary_weights(weights, 1.0)
        
        input_tensor = torch.randn(2, 10)
        output = layer(input_tensor)
        
        assert output.shape == (2, 5)


class TestIntegration:
    """Integration tests for complete quantization pipeline."""
    
    def test_small_model_quantization(self):
        """Test quantizing a small model."""
        # Create a small model
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Quantize
        quantizer = PTBitNetQuantizer()
        quantized_layers = quantizer.quantize_model(model)
        
        # Check that linear layers were quantized
        assert len(quantized_layers) == 3
        
        # Check compression ratio
        for name, quant_data in quantized_layers.items():
            stats = quant_data['statistics']
            assert stats['compression_ratio'] > 10
    
    def test_memory_reduction(self):
        """Test that quantization reduces memory usage."""
        # Create a larger layer
        layer = nn.Linear(1024, 512)
        
        # Calculate original size
        original_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        original_bytes = original_params * 4  # 32-bit floats
        
        # Quantize
        quantizer = PTBitNetQuantizer()
        quantized = quantizer.quantize_layer(layer)
        
        # Pack weights
        packed = pack_ternary_weights(quantized['quantized_weights'])
        
        # Calculate quantized size
        quantized_bytes = packed.numel() + 4  # weights + scale
        if quantized['bias'] is not None:
            quantized_bytes += quantized['bias'].numel() * 4
        
        # Check significant reduction
        reduction_ratio = original_bytes / quantized_bytes
        assert reduction_ratio > 10  # At least 10x reduction


if __name__ == '__main__':
    pytest.main([__file__, '-v'])