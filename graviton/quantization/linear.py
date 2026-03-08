"""
Linear Quantization Engine

Implements standard uniform (linear) quantization for INT8, INT4, and INT2.
Supports per-channel and per-group quantization with both symmetric
and asymmetric modes.

Key insight: By quantizing weights to low-precision integers, we dramatically
reduce both memory and computation. A 4-bit model is 4x smaller than FP16,
and integer operations are much faster than floating-point.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch

from graviton.quantization.base import BaseQuantizer, QuantizedTensor

logger = logging.getLogger(__name__)


class LinearQuantizer(BaseQuantizer):
    """
    Uniform linear quantizer supporting INT8, INT4, and INT2.

    Quantizes using the formula:
        q = round(clamp(x / scale + zero_point, qmin, qmax))

    Dequantizes using:
        x_hat = (q - zero_point) * scale

    Supports per-channel and per-group quantization for higher accuracy.

    Example:
        >>> quantizer = LinearQuantizer(bits=4, group_size=128)
        >>> qtensor = quantizer.quantize(weight_tensor)
        >>> print(f"Compression: {qtensor.compression_ratio:.1f}x")
        >>> reconstructed = quantizer.dequantize(qtensor)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
    ):
        """
        Initialize the linear quantizer.

        Args:
            bits: Target bit width (2, 4, or 8).
            group_size: Number of elements per quantization group.
                Smaller groups = higher accuracy, more overhead.
            symmetric: If True, uses symmetric quantization (zero_point=0).
                If False, uses asymmetric quantization.
        """
        assert bits in (2, 4, 8), f"Supported bit widths: 2, 4, 8. Got: {bits}"

        self._bits = bits
        self._group_size = group_size
        self._symmetric = symmetric

        # Compute quantization range
        if symmetric:
            self._qmin = -(2 ** (bits - 1))
            self._qmax = 2 ** (bits - 1) - 1
        else:
            self._qmin = 0
            self._qmax = 2**bits - 1

        logger.debug(
            f"LinearQuantizer: {bits}-bit, group_size={group_size}, "
            f"symmetric={symmetric}, range=[{self._qmin}, {self._qmax}]"
        )

    @property
    def name(self) -> str:
        return f"linear_int{self._bits}"

    @property
    def bits(self) -> float:
        return float(self._bits)

    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        """
        Quantize a tensor to the target bit width.

        Uses per-group quantization for accuracy. Each group of
        `group_size` elements gets its own scale factor.

        Args:
            tensor: Input float tensor of any shape.

        Returns:
            QuantizedTensor with compressed data.
        """
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # Flatten for group processing
        flat = tensor.float().reshape(-1)
        num_elements = flat.numel()

        # Pad to group size if needed
        group_size = min(self._group_size, num_elements)
        if num_elements % group_size != 0:
            padding = group_size - (num_elements % group_size)
            flat = torch.nn.functional.pad(flat, (0, padding))
        else:
            padding = 0

        # Reshape into groups
        num_groups = flat.numel() // group_size
        grouped = flat.reshape(num_groups, group_size)

        if self._symmetric:
            # Symmetric: scale = max(|x|) / qmax
            max_abs = grouped.abs().max(dim=1, keepdim=True).values
            max_abs = max_abs.clamp(min=1e-8)  # Avoid division by zero
            scale = max_abs / self._qmax
            zero_point = None

            # Quantize
            quantized = torch.clamp(
                torch.round(grouped / scale), self._qmin, self._qmax
            ).to(torch.int8)

        else:
            # Asymmetric: scale = (max - min) / (qmax - qmin)
            mins = grouped.min(dim=1, keepdim=True).values
            maxs = grouped.max(dim=1, keepdim=True).values
            scale = (maxs - mins).clamp(min=1e-8) / (self._qmax - self._qmin)
            zero_point = torch.clamp(
                torch.round(-mins / scale) + self._qmin, self._qmin, self._qmax
            ).to(torch.int8)

            # Quantize
            quantized = torch.clamp(
                torch.round(grouped / scale + zero_point.float()),
                self._qmin,
                self._qmax,
            ).to(torch.int8)

        # Pack into efficient storage
        packed_data = self._pack_integers(quantized.reshape(-1), self._bits)

        return QuantizedTensor(
            data=packed_data,
            scale=scale.squeeze(1).to(torch.float16),
            zero_point=(
                zero_point.squeeze(1).to(torch.int8) if zero_point is not None else None
            ),
            bits=float(self._bits),
            shape=original_shape,
            group_size=group_size,
            dtype=original_dtype,
        )

    def dequantize(self, qtensor: QuantizedTensor) -> torch.Tensor:
        """
        Dequantize back to floating-point.

        Args:
            qtensor: Quantized tensor.

        Returns:
            Reconstructed float tensor.
        """
        # Unpack integers
        unpacked = self._unpack_integers(qtensor.data, int(qtensor.bits))

        # Reshape into groups
        group_size = qtensor.group_size
        num_elements_padded = unpacked.numel()
        num_groups = num_elements_padded // group_size
        grouped = unpacked[:num_groups * group_size].reshape(num_groups, group_size).float()

        scale = qtensor.scale.float().unsqueeze(1)

        if qtensor.zero_point is not None:
            # Asymmetric
            zero_point = qtensor.zero_point.float().unsqueeze(1)
            dequantized = (grouped - zero_point) * scale
        else:
            # Symmetric
            dequantized = grouped * scale

        # Remove padding and reshape
        flat = dequantized.reshape(-1)
        num_elements = math.prod(qtensor.shape)
        flat = flat[:num_elements]

        return flat.reshape(qtensor.shape).to(qtensor.dtype)

    def _pack_integers(self, data: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Pack low-bit integers into bytes for efficient storage.

        For 4-bit: pack 2 values per byte
        For 2-bit: pack 4 values per byte
        For 8-bit: no packing needed

        Args:
            data: Integer tensor with values in valid range.
            bits: Bit width.

        Returns:
            Packed byte tensor.
        """
        if bits == 8:
            return data.to(torch.int8)

        # Convert to unsigned for packing
        if self._symmetric:
            # Shift symmetric range to unsigned
            offset = 2 ** (bits - 1)
            unsigned = (data.int() + offset).clamp(0, 2**bits - 1).to(torch.uint8)
        else:
            unsigned = data.clamp(0, 2**bits - 1).to(torch.uint8)

        elements_per_byte = 8 // bits

        # Pad to multiple of elements_per_byte
        if unsigned.numel() % elements_per_byte != 0:
            pad = elements_per_byte - (unsigned.numel() % elements_per_byte)
            unsigned = torch.nn.functional.pad(unsigned, (0, pad))

        # Pack
        packed = torch.zeros(
            unsigned.numel() // elements_per_byte, dtype=torch.uint8,
            device=data.device,
        )

        for i in range(elements_per_byte):
            packed |= unsigned[i::elements_per_byte] << (i * bits)

        return packed

    def _unpack_integers(self, packed: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Unpack bytes back to integer tensor.

        Args:
            packed: Packed byte tensor.
            bits: Original bit width.

        Returns:
            Unpacked integer tensor.
        """
        if bits == 8:
            return packed.float()

        elements_per_byte = 8 // bits
        mask = (1 << bits) - 1

        device = packed.device
        unpacked = torch.zeros(
            packed.numel() * elements_per_byte, dtype=torch.float32, device=device
        )

        for i in range(elements_per_byte):
            unpacked[i::elements_per_byte] = ((packed >> (i * bits)) & mask).float()

        # Convert back to signed if symmetric
        if self._symmetric:
            offset = 2 ** (bits - 1)
            unpacked = unpacked - offset

        return unpacked
