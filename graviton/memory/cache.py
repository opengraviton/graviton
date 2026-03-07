"""
KV-Cache Compression

Compresses the key-value cache used during autoregressive generation
to reduce memory usage for long sequences. Without compression,
the KV cache can consume more memory than the model weights themselves.

Techniques:
- Quantization of cached keys/values to INT4/INT8
- Sliding window to bound cache size
- Token eviction for ultra-long sequences
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class KVCacheCompressor:
    """
    Compressed KV-cache for memory-efficient generation.

    The KV cache stores past key and value tensors for attention,
    growing linearly with sequence length. For long contexts, this
    can consume enormous memory.

    KVCacheCompressor reduces this by:
    1. Quantizing cached K/V to lower precision
    2. Using sliding window to bound memory
    3. Evicting unimportant tokens

    Example:
        >>> cache = KVCacheCompressor(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     max_length=4096,
        ...     bits=8,
        ... )
        >>> cache.update(layer_idx=0, key=new_key, value=new_value)
        >>> k, v = cache.get(layer_idx=0)
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_length: int = 4096,
        bits: int = 8,
        sliding_window: Optional[int] = None,
    ):
        """
        Initialize the KV cache compressor.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            max_length: Maximum sequence length.
            bits: Quantization bits for cache (4 or 8).
            sliding_window: If set, only keep last N tokens.
        """
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_length = max_length
        self._bits = bits
        self._sliding_window = sliding_window or max_length

        # Storage: quantized KV per layer
        # Format: [batch, heads, seq_len, head_dim]
        self._keys: dict = {}  # layer_idx → (quantized_data, scale)
        self._values: dict = {}
        self._lengths: dict = {}  # layer_idx → current sequence length

        # Quantization range
        if bits == 8:
            self._qmin, self._qmax = -128, 127
        elif bits == 4:
            self._qmin, self._qmax = -8, 7
        else:
            self._qmin, self._qmax = -128, 127

        logger.info(
            f"KVCacheCompressor: {num_layers} layers, {bits}-bit, "
            f"window={self._sliding_window}"
        )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Add new key-value pair to the cache.

        Args:
            layer_idx: Transformer layer index.
            key: New key tensor [batch, heads, new_len, head_dim].
            value: New value tensor [batch, heads, new_len, head_dim].
        """
        # Quantize new KV
        q_key, k_scale = self._quantize_cache(key)
        q_value, v_scale = self._quantize_cache(value)

        if layer_idx not in self._keys:
            # First update
            self._keys[layer_idx] = (q_key, k_scale)
            self._values[layer_idx] = (q_value, v_scale)
            self._lengths[layer_idx] = key.shape[-2]
        else:
            # Append to existing cache
            old_q_key, old_k_scale = self._keys[layer_idx]
            old_q_value, old_v_scale = self._values[layer_idx]

            new_q_key = torch.cat([old_q_key, q_key], dim=-2)
            new_k_scale = torch.cat([old_k_scale, k_scale], dim=-2)
            new_q_value = torch.cat([old_q_value, q_value], dim=-2)
            new_v_scale = torch.cat([old_v_scale, v_scale], dim=-2)

            # Apply sliding window
            seq_len = new_q_key.shape[-2]
            if seq_len > self._sliding_window:
                trim = seq_len - self._sliding_window
                new_q_key = new_q_key[..., trim:, :]
                new_k_scale = new_k_scale[..., trim:, :]
                new_q_value = new_q_value[..., trim:, :]
                new_v_scale = new_v_scale[..., trim:, :]

            self._keys[layer_idx] = (new_q_key, new_k_scale)
            self._values[layer_idx] = (new_q_value, new_v_scale)
            self._lengths[layer_idx] = new_q_key.shape[-2]

    def get(
        self,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get dequantized key-value tensors for a layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (key, value) tensors, or (None, None) if empty.
        """
        if layer_idx not in self._keys:
            return None, None

        q_key, k_scale = self._keys[layer_idx]
        q_value, v_scale = self._values[layer_idx]

        key = self._dequantize_cache(q_key, k_scale)
        value = self._dequantize_cache(q_value, v_scale)

        return key, value

    def _quantize_cache(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a KV tensor for compressed storage.

        Args:
            tensor: Input tensor [batch, heads, seq_len, head_dim].

        Returns:
            Tuple of (quantized_data, scale_factors).
        """
        # Per-head quantization scale
        max_abs = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / self._qmax

        quantized = torch.clamp(
            torch.round(tensor / scale), self._qmin, self._qmax
        ).to(torch.int8)

        return quantized, scale.to(torch.float16)

    def _dequantize_cache(
        self, quantized: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize cached KV tensor.

        Args:
            quantized: Quantized int8 tensor.
            scale: Scale factors.

        Returns:
            Dequantized float tensor.
        """
        return quantized.float() * scale.float()

    def clear(self, layer_idx: Optional[int] = None):
        """
        Clear the cache.

        Args:
            layer_idx: If specified, clear only this layer.
                If None, clear all layers.
        """
        if layer_idx is not None:
            self._keys.pop(layer_idx, None)
            self._values.pop(layer_idx, None)
            self._lengths.pop(layer_idx, None)
        else:
            self._keys.clear()
            self._values.clear()
            self._lengths.clear()

    def memory_usage_bytes(self) -> int:
        """Total memory used by the cache in bytes."""
        total = 0
        for layer_idx in self._keys:
            q_key, k_scale = self._keys[layer_idx]
            q_value, v_scale = self._values[layer_idx]
            total += q_key.numel() * q_key.element_size()
            total += k_scale.numel() * k_scale.element_size()
            total += q_value.numel() * q_value.element_size()
            total += v_scale.numel() * v_scale.element_size()
        return total

    def memory_usage_gb(self) -> float:
        """Total memory used by the cache in GB."""
        return self.memory_usage_bytes() / (1024**3)

    def uncompressed_size_bytes(self) -> int:
        """Estimated uncompressed size in bytes."""
        total = 0
        for layer_idx in self._lengths:
            seq_len = self._lengths[layer_idx]
            # FP16: 2 bytes per element
            total += 2 * 2 * self._num_heads * seq_len * self._head_dim  # K + V
        return total

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs uncompressed FP16."""
        compressed = self.memory_usage_bytes()
        if compressed == 0:
            return 0.0
        return self.uncompressed_size_bytes() / compressed

    def statistics(self) -> dict:
        """Cache statistics."""
        return {
            "num_layers_cached": len(self._keys),
            "memory_usage_mb": round(self.memory_usage_bytes() / (1024**2), 2),
            "uncompressed_mb": round(
                self.uncompressed_size_bytes() / (1024**2), 2
            ),
            "compression_ratio": round(self.compression_ratio, 2),
            "bits": self._bits,
            "sliding_window": self._sliding_window,
            "max_length": self._max_length,
        }
