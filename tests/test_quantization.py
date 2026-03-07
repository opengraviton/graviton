import pytest
import torch
import math

from graviton.quantization.linear import LinearQuantizer
from graviton.quantization.ternary import TernaryQuantizer
from graviton.quantization.base import QuantizedTensor

def test_linear_quantization_int8():
    quantizer = LinearQuantizer(bits=8, symmetric=True)
    
    # Create a dummy weight matrix
    torch.manual_seed(42)
    weights = torch.randn(64, 128)
    
    # Quantize
    qtensor = quantizer.quantize(weights)
    
    assert isinstance(qtensor, QuantizedTensor)
    assert qtensor.bits == 8
    assert qtensor.shape == (64, 128)
    
    # Dequantize
    reconstructed = quantizer.dequantize(qtensor)
    
    assert reconstructed.shape == weights.shape
    
    # Check MSE error is reasonably small for 8-bit
    error = torch.mean((weights - reconstructed) ** 2).item()
    assert error < 0.01

def test_ternary_quantization_packing():
    quantizer = TernaryQuantizer(alpha=0.5, group_size=32)
    
    torch.manual_seed(42)
    weights = torch.randn(32, 32)
    
    qtensor = quantizer.quantize(weights)
    
    assert isinstance(qtensor, QuantizedTensor)
    assert qtensor.bits == math.log2(3)
    
    # Check if the data is packed effectively (1024 params / 4 trits per byte = 256 bytes)
    assert qtensor.data.numel() == (32 * 32) // 4
    assert qtensor.data.dtype == torch.uint8
    
    reconstructed = quantizer.dequantize(qtensor)
    assert reconstructed.shape == weights.shape

def test_ternary_matmul_correctness():
    quantizer = TernaryQuantizer(alpha=0.5, group_size=128)
    
    torch.manual_seed(42)
    weights = torch.randn(64, 128)
    inputs = torch.randn(16, 128)
    
    qtensor = quantizer.quantize(weights)
    
    # 1. Custom Ternary Matmul
    result_ternary = quantizer.ternary_matmul(inputs, qtensor)
    
    # 2. Reconstructed Float Matmul
    reconstructed_weights = quantizer.dequantize(qtensor)
    result_float = torch.matmul(inputs.float(), reconstructed_weights.t())
    
    # The ternary optimized path and the standard float math on reconstructed
    # ternary weights should be identical within float precision.
    error = torch.mean((result_ternary - result_float) ** 2).item()
    assert error < 1e-4
