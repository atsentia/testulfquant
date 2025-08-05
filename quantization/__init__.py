"""Quantization module for 1.58-bit model compression."""

from .pt_bitnet import (
    PTBitNetQuantizer,
    pack_ternary_weights,
    unpack_ternary_weights
)
from .bitlinear import (
    BitLinear,
    BitLinearOptimized,
    replace_linear_with_bitlinear
)

__all__ = [
    'PTBitNetQuantizer',
    'pack_ternary_weights',
    'unpack_ternary_weights',
    'BitLinear',
    'BitLinearOptimized',
    'replace_linear_with_bitlinear'
]