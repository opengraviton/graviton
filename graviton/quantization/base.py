"""
Base Quantizer Interface

Abstract base class that all quantization strategies must implement.
Provides a consistent interface for quantize/dequantize operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

import torch


@dataclass
class QuantizedTensor:
    """
    A quantized tensor representation.

    Stores quantized data along with metadata needed for dequantization.

    Attributes:
        data: The quantized integer data.
        scale: Per-group or per-channel scale factors.
        zero_point: Zero point for asymmetric quantization.
        bits: Number of bits used for quantization.
        shape: Original tensor shape.
        group_size: Number of elements per quantization group.
        dtype: Original tensor dtype.
    """

    data: torch.Tensor
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor] = None
    bits: float = 4.0
    shape: tuple = ()
    group_size: int = 128
    dtype: torch.dtype = torch.float16

    @property
    def compressed_size_bytes(self) -> int:
        """Size of the compressed representation in bytes."""
        data_bytes = (self.data.numel() * self.bits) / 8
        scale_bytes = self.scale.numel() * self.scale.element_size()
        zp_bytes = (
            self.zero_point.numel() * self.zero_point.element_size()
            if self.zero_point is not None
            else 0
        )
        return int(data_bytes + scale_bytes + zp_bytes)

    @property
    def original_size_bytes(self) -> int:
        """Size of the original tensor in bytes."""
        import math

        numel = math.prod(self.shape) if self.shape else 0
        dtype_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(self.dtype, 4)
        return numel * dtype_size

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        comp = self.compressed_size_bytes
        if comp == 0:
            return 0.0
        return self.original_size_bytes / comp

    def numel(self) -> int:
        """Number of elements in the original tensor."""
        import math

        return math.prod(self.shape) if self.shape else 0


class BaseQuantizer(ABC):
    """
    Abstract base class for quantizers.

    All quantization strategies (linear, ternary, mixed-precision)
    must implement this interface.
    """

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        """
        Quantize a floating-point tensor.

        Args:
            tensor: Input tensor in float16/float32.

        Returns:
            QuantizedTensor with compressed data and metadata.
        """
        ...

    @abstractmethod
    def dequantize(self, qtensor: QuantizedTensor) -> torch.Tensor:
        """
        Dequantize back to floating-point.

        Args:
            qtensor: Quantized tensor.

        Returns:
            Dequantized tensor (approximate reconstruction).
        """
        ...

    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize and immediately dequantize (simulate quantization error).

        Useful for evaluating quantization quality without storing
        the compressed representation.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor after quantization-dequantization roundtrip.
        """
        qtensor = self.quantize(tensor)
        return self.dequantize(qtensor)

    def compute_error(self, tensor: torch.Tensor) -> dict:
        """
        Compute quantization error metrics.

        Args:
            tensor: Original tensor.

        Returns:
            Dictionary with error metrics (MSE, SNR, max error).
        """
        reconstructed = self.quantize_dequantize(tensor)
        diff = tensor.float() - reconstructed.float()

        mse = (diff**2).mean().item()
        signal_power = (tensor.float() ** 2).mean().item()
        snr = 10 * torch.log10(
            torch.tensor(signal_power / max(mse, 1e-10))
        ).item()
        max_error = diff.abs().max().item()
        mean_abs_error = diff.abs().mean().item()

        qtensor = self.quantize(tensor)

        return {
            "mse": mse,
            "snr_db": snr,
            "max_error": max_error,
            "mean_abs_error": mean_abs_error,
            "compression_ratio": qtensor.compression_ratio,
            "bits": qtensor.bits,
        }

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the quantization strategy."""
        ...

    @property
    @abstractmethod
    def bits(self) -> float:
        """Target bits per parameter."""
        ...
