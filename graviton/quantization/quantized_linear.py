"""
Quantized Linear Layer

Drop-in replacement for nn.Linear that stores weights in quantized format.
Supports INT4, INT8, and 1.58-bit ternary quantization.

Memory is reduced during storage; weights are dequantized on the first
forward pass and cached for subsequent calls.  For ternary mode the
cached representation uses two half-precision masks (positive/negative)
so that matmul reduces to pure addition and subtraction.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from graviton.quantization.base import QuantizedTensor

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """
    Linear layer backed by quantized weights.

    After construction the original float weights are discarded.  On the
    first forward call the packed data is decompressed into a compute-
    friendly representation and cached so that subsequent calls are fast.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set by from_linear()
        self._quantizer = None
        self._is_ternary = False

        # Register a None buffer so .to(device) picks it up if set later
        self.register_buffer("_bias", None)

        # Cached compute-ready weights (populated on first forward)
        self._cached_weight: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantizer) -> "QuantizedLinear":
        """
        Quantize an existing ``nn.Linear`` and wrap it.

        The original float weight is quantized, packed, and stored as
        registered buffers so that ``.to(device)`` works transparently.
        """
        from graviton.quantization.ternary import TernaryQuantizer

        mod = cls(linear.in_features, linear.out_features)
        mod._quantizer = quantizer
        mod._is_ternary = isinstance(quantizer, TernaryQuantizer)

        qw = quantizer.quantize(linear.weight.data)

        mod.register_buffer("_packed_data", qw.data)
        mod.register_buffer("_scale", qw.scale)
        if qw.zero_point is not None:
            mod.register_buffer("_zero_point", qw.zero_point)
        else:
            mod._zero_point = None

        mod._bits = qw.bits
        mod._group_size = qw.group_size
        mod._orig_dtype = qw.dtype

        if linear.bias is not None:
            mod._bias = linear.bias.data.clone()

        return mod

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_weight is None:
            self._materialize_weight(x.dtype, x.device)

        out = F.linear(x, self._cached_weight, self._bias)
        return out

    def _materialize_weight(self, dtype: torch.dtype, device: torch.device):
        """Dequantize packed weights once and cache the result."""
        qtensor = QuantizedTensor(
            data=self._packed_data,
            scale=self._scale,
            zero_point=self._zero_point,
            bits=self._bits,
            shape=(self.out_features, self.in_features),
            group_size=self._group_size,
            dtype=self._orig_dtype,
        )
        weight = self._quantizer.dequantize(qtensor)
        self._cached_weight = weight.to(dtype=dtype, device=device)

        packed_bytes = self._packed_data.numel() * self._packed_data.element_size()
        float_bytes = self._cached_weight.numel() * self._cached_weight.element_size()
        logger.debug(
            f"QuantizedLinear [{self.out_features}x{self.in_features}]: "
            f"packed {packed_bytes / 1024:.0f} KB -> cached {float_bytes / 1024:.0f} KB"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def packed_size_bytes(self) -> int:
        """Size of the quantized (packed) weight in bytes."""
        return self._packed_data.numel() * self._packed_data.element_size()

    @property
    def is_ternary(self) -> bool:
        return self._is_ternary

    def extra_repr(self) -> str:
        mode = "ternary" if self._is_ternary else f"{self._bits:.0f}bit"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={mode}"
        )
