"""
KV-Cache for Autoregressive Generation

Supports two modes:
  - **Fast mode** (default): Pre-allocated float buffers with zero-copy
    slice views. No quantization overhead — maximum decode speed.
  - **Compressed mode**: INT8 quantized storage for memory-constrained
    long-context scenarios.

The fast path eliminates per-token quantize/dequantize and torch.cat
allocations, which are the primary bottleneck on MPS / CUDA.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class KVCacheCompressor:
    """
    KV-cache with optional INT8 compression.

    By default operates in **fast (uncompressed)** mode: pre-allocated
    float16/float32 buffers with in-place writes and zero-copy reads.
    Set ``compress=True`` to enable INT8 quantization for long-context
    memory savings at the cost of decode speed.

    Example:
        >>> cache = KVCacheCompressor(num_layers=32, num_heads=32,
        ...                           head_dim=128, max_length=4096)
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
        compress: bool = False,
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_length = max_length
        self._bits = bits
        self._sliding_window = sliding_window or max_length
        self._compress = compress

        self._keys: dict = {}
        self._values: dict = {}
        self._lengths: dict = {}

        if bits == 8:
            self._qmin, self._qmax = -128, 127
        elif bits == 4:
            self._qmin, self._qmax = -8, 7
        else:
            self._qmin, self._qmax = -128, 127

        mode_str = f"{bits}-bit compressed" if compress else "uncompressed (fast)"
        logger.info(
            f"KVCache: {num_layers} layers, {mode_str}, "
            f"window={self._sliding_window}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Append new K/V to the cache for *layer_idx*."""
        if self._compress:
            self._update_compressed(layer_idx, key, value)
        else:
            self._update_fast(layer_idx, key, value)

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return cached (key, value) for *layer_idx*, or (None, None)."""
        if self._compress:
            return self._get_compressed(layer_idx)
        return self._get_fast(layer_idx)

    # ------------------------------------------------------------------
    # Fast (uncompressed) path — pre-allocated buffers, zero-copy views
    # ------------------------------------------------------------------

    def _update_fast(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        batch, heads, new_len, dim = key.shape

        if layer_idx not in self._keys:
            buf_k = torch.zeros(
                batch, heads, self._max_length, dim,
                dtype=key.dtype, device=key.device,
            )
            buf_v = torch.zeros(
                batch, heads, self._max_length, dim,
                dtype=value.dtype, device=value.device,
            )
            self._keys[layer_idx] = buf_k
            self._values[layer_idx] = buf_v
            self._lengths[layer_idx] = 0

        cur = self._lengths[layer_idx]
        end = cur + new_len

        if end > self._sliding_window:
            trim = end - self._sliding_window
            self._keys[layer_idx][:, :, :self._sliding_window - new_len, :] = (
                self._keys[layer_idx][:, :, trim:cur, :].clone()
            )
            self._values[layer_idx][:, :, :self._sliding_window - new_len, :] = (
                self._values[layer_idx][:, :, trim:cur, :].clone()
            )
            cur = self._sliding_window - new_len
            end = self._sliding_window

        self._keys[layer_idx][:, :, cur:end, :] = key
        self._values[layer_idx][:, :, cur:end, :] = value
        self._lengths[layer_idx] = end

    def _get_fast(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if layer_idx not in self._keys:
            return None, None
        length = self._lengths[layer_idx]
        return (
            self._keys[layer_idx][:, :, :length, :],
            self._values[layer_idx][:, :, :length, :],
        )

    # ------------------------------------------------------------------
    # Compressed (INT8) path — original implementation
    # ------------------------------------------------------------------

    def _update_compressed(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        q_key, k_scale = self._quantize_cache(key)
        q_value, v_scale = self._quantize_cache(value)

        if layer_idx not in self._keys:
            self._keys[layer_idx] = (q_key, k_scale)
            self._values[layer_idx] = (q_value, v_scale)
            self._lengths[layer_idx] = key.shape[-2]
        else:
            old_q_key, old_k_scale = self._keys[layer_idx]
            old_q_value, old_v_scale = self._values[layer_idx]

            new_q_key = torch.cat([old_q_key, q_key], dim=-2)
            new_k_scale = torch.cat([old_k_scale, k_scale], dim=-2)
            new_q_value = torch.cat([old_q_value, q_value], dim=-2)
            new_v_scale = torch.cat([old_v_scale, v_scale], dim=-2)

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

    def _get_compressed(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if layer_idx not in self._keys:
            return None, None
        q_key, k_scale = self._keys[layer_idx]
        q_value, v_scale = self._values[layer_idx]
        return self._dequantize_cache(q_key, k_scale), self._dequantize_cache(q_value, v_scale)

    def _quantize_cache(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_abs = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / self._qmax
        quantized = torch.clamp(
            torch.round(tensor / scale), self._qmin, self._qmax
        ).to(torch.int8)
        return quantized, scale.to(torch.float16)

    def _dequantize_cache(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return quantized.float() * scale.float()

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def clear(self, layer_idx: Optional[int] = None):
        if layer_idx is not None:
            self._keys.pop(layer_idx, None)
            self._values.pop(layer_idx, None)
            self._lengths.pop(layer_idx, None)
        else:
            self._keys.clear()
            self._values.clear()
            self._lengths.clear()

    def get_positions(self) -> dict:
        """Return a snapshot of current sequence lengths per layer."""
        return dict(self._lengths)

    def truncate_to(self, positions: dict):
        """Truncate each layer's cache to the given sequence length."""
        for layer_idx, target_len in positions.items():
            if layer_idx not in self._keys:
                continue
            current_len = self._lengths.get(layer_idx, 0)
            if target_len >= current_len:
                continue

            if self._compress:
                q_key, k_scale = self._keys[layer_idx]
                q_value, v_scale = self._values[layer_idx]
                self._keys[layer_idx] = (q_key[..., :target_len, :], k_scale[..., :target_len, :])
                self._values[layer_idx] = (q_value[..., :target_len, :], v_scale[..., :target_len, :])
            # For fast mode the buffer stays allocated; we just move the length pointer.
            self._lengths[layer_idx] = target_len

    def memory_usage_bytes(self) -> int:
        total = 0
        if self._compress:
            for layer_idx in self._keys:
                q_key, k_scale = self._keys[layer_idx]
                q_value, v_scale = self._values[layer_idx]
                total += q_key.numel() * q_key.element_size()
                total += k_scale.numel() * k_scale.element_size()
                total += q_value.numel() * q_value.element_size()
                total += v_scale.numel() * v_scale.element_size()
        else:
            for layer_idx in self._keys:
                total += self._keys[layer_idx].numel() * self._keys[layer_idx].element_size()
                total += self._values[layer_idx].numel() * self._values[layer_idx].element_size()
        return total

    def memory_usage_gb(self) -> float:
        return self.memory_usage_bytes() / (1024**3)

    def uncompressed_size_bytes(self) -> int:
        total = 0
        for layer_idx in self._lengths:
            seq_len = self._lengths[layer_idx]
            total += 2 * 2 * self._num_heads * seq_len * self._head_dim
        return total

    @property
    def compression_ratio(self) -> float:
        compressed = self.memory_usage_bytes()
        if compressed == 0:
            return 0.0
        return self.uncompressed_size_bytes() / compressed

    def statistics(self) -> dict:
        return {
            "num_layers_cached": len(self._keys),
            "memory_usage_mb": round(self.memory_usage_bytes() / (1024**2), 2),
            "uncompressed_mb": round(self.uncompressed_size_bytes() / (1024**2), 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "bits": self._bits,
            "compressed": self._compress,
            "sliding_window": self._sliding_window,
            "max_length": self._max_length,
        }
