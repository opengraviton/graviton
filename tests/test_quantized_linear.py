"""Tests for QuantizedLinear, KV cache snapshot/truncation, and device-aware quantizers."""

import torch
import torch.nn as nn

from graviton.quantization.linear import LinearQuantizer
from graviton.quantization.ternary import TernaryQuantizer
from graviton.quantization.quantized_linear import QuantizedLinear
from graviton.memory.cache import KVCacheCompressor


# ── QuantizedLinear ─────────────────────────────────────────────────


def test_quantized_linear_int4_roundtrip():
    """QuantizedLinear INT4 output should be close to the original nn.Linear."""
    torch.manual_seed(42)
    linear = nn.Linear(128, 64, bias=False)

    quantizer = LinearQuantizer(bits=4, group_size=64, symmetric=True)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    x = torch.randn(2, 128)
    expected = linear(x)
    actual = ql(x)

    assert actual.shape == expected.shape
    error = (expected - actual).abs().mean().item()
    assert error < 0.5, f"INT4 error too large: {error}"


def test_quantized_linear_int8_roundtrip():
    """INT8 should have much less error than INT4."""
    torch.manual_seed(42)
    linear = nn.Linear(128, 64, bias=False)

    quantizer = LinearQuantizer(bits=8, group_size=128, symmetric=True)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    x = torch.randn(4, 128)
    expected = linear(x)
    actual = ql(x)

    assert actual.shape == expected.shape
    error = (expected - actual).abs().mean().item()
    assert error < 0.05, f"INT8 error too large: {error}"


def test_quantized_linear_ternary_roundtrip():
    """QuantizedLinear with ternary quantization should work end-to-end."""
    torch.manual_seed(42)
    linear = nn.Linear(128, 64, bias=False)

    quantizer = TernaryQuantizer(alpha=0.7, group_size=64)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    x = torch.randn(2, 128)
    actual = ql(x)
    assert actual.shape == (2, 64)
    assert not torch.isnan(actual).any()


def test_quantized_linear_3d_input():
    """QuantizedLinear should handle [batch, seq, features] inputs."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=True)

    quantizer = LinearQuantizer(bits=8, group_size=64)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    x = torch.randn(1, 10, 64)
    actual = ql(x)
    assert actual.shape == (1, 10, 32)


def test_quantized_linear_with_bias():
    """Bias should be preserved and applied correctly."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=True)
    original_bias = linear.bias.data.clone()

    quantizer = LinearQuantizer(bits=8, group_size=64)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    assert ql._bias is not None
    assert torch.allclose(ql._bias, original_bias)

    x = torch.randn(2, 64)
    out = ql(x)
    assert out.shape == (2, 32)


def test_quantized_linear_without_bias():
    """No-bias layers should have _bias == None."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=False)

    quantizer = LinearQuantizer(bits=4, group_size=32)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    assert ql._bias is None


def test_quantized_linear_device_transfer():
    """QuantizedLinear buffers should move with .to()."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=False)
    quantizer = LinearQuantizer(bits=4, group_size=32)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    ql_cpu = ql.to("cpu")
    assert ql_cpu._packed_data.device.type == "cpu"
    assert ql_cpu._scale.device.type == "cpu"


def test_quantized_linear_packed_size_bytes():
    """packed_size_bytes should be much smaller than the original weight."""
    torch.manual_seed(42)
    linear = nn.Linear(256, 128, bias=False)
    original_bytes = linear.weight.numel() * linear.weight.element_size()

    quantizer = LinearQuantizer(bits=4, group_size=64)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    assert ql.packed_size_bytes < original_bytes
    assert ql.packed_size_bytes > 0


def test_quantized_linear_extra_repr():
    """extra_repr should report the quantization mode."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=False)

    q4 = QuantizedLinear.from_linear(linear, LinearQuantizer(bits=4))
    assert "4bit" in q4.extra_repr()
    assert "in_features=64" in q4.extra_repr()

    qt = QuantizedLinear.from_linear(linear, TernaryQuantizer())
    assert "ternary" in qt.extra_repr()


def test_quantized_linear_cached_weight_persists():
    """After the first forward call, the cached weight should be reused."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=False)
    quantizer = LinearQuantizer(bits=8, group_size=64)
    ql = QuantizedLinear.from_linear(linear, quantizer)

    assert ql._cached_weight is None

    x = torch.randn(2, 64)
    ql(x)
    assert ql._cached_weight is not None
    cached_ref = ql._cached_weight

    ql(x)
    assert ql._cached_weight is cached_ref  # same object, no re-dequantize


# ── KV Cache Snapshot & Truncation ──────────────────────────────────


def test_kv_cache_snapshot_and_truncate():
    """KV cache should support position snapshots and truncation."""
    cache = KVCacheCompressor(num_layers=4, num_heads=2, head_dim=8, max_length=64)

    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    cache.update(0, key, value)
    cache.update(1, key, value)

    snap = cache.get_positions()
    assert snap[0] == 5
    assert snap[1] == 5

    key2 = torch.randn(1, 2, 3, 8)
    value2 = torch.randn(1, 2, 3, 8)
    cache.update(0, key2, value2)
    assert cache.get_positions()[0] == 8

    cache.truncate_to({0: 5, 1: 5})
    assert cache.get_positions()[0] == 5
    assert cache.get_positions()[1] == 5

    k, v = cache.get(0)
    assert k.shape[-2] == 5


def test_kv_cache_empty_positions():
    """get_positions on a fresh cache should return an empty dict."""
    cache = KVCacheCompressor(num_layers=4, num_heads=2, head_dim=8, max_length=64)
    assert cache.get_positions() == {}


def test_kv_cache_truncate_noop_when_larger():
    """Truncating to a length >= current should be a no-op."""
    cache = KVCacheCompressor(num_layers=2, num_heads=1, head_dim=4, max_length=64)

    cache.update(0, torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
    cache.truncate_to({0: 10})  # 10 > 3, no effect

    k, v = cache.get(0)
    assert k.shape[-2] == 3


def test_kv_cache_truncate_to_zero():
    """Truncating to 0 should leave an empty-length entry."""
    cache = KVCacheCompressor(num_layers=2, num_heads=1, head_dim=4, max_length=64)

    cache.update(0, torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
    cache.truncate_to({0: 0})

    k, v = cache.get(0)
    assert k.shape[-2] == 0


# ── Device-aware quantizer operations ───────────────────────────────


def test_linear_quantizer_device_roundtrip():
    """Quantize/dequantize should keep tensors on the same device."""
    quantizer = LinearQuantizer(bits=4, group_size=32)
    w = torch.randn(32, 64)  # CPU

    qt = quantizer.quantize(w)
    assert qt.data.device.type == "cpu"
    assert qt.scale.device.type == "cpu"

    recon = quantizer.dequantize(qt)
    assert recon.device.type == "cpu"
    assert recon.shape == w.shape


def test_ternary_quantizer_device_roundtrip():
    """Ternary quantize/dequantize should stay on the same device."""
    quantizer = TernaryQuantizer(alpha=0.7, group_size=32)
    w = torch.randn(32, 64)

    qt = quantizer.quantize(w)
    assert qt.data.device.type == "cpu"

    recon = quantizer.dequantize(qt)
    assert recon.device.type == "cpu"
    assert recon.shape == w.shape


def test_ternary_matmul_device_consistency():
    """ternary_matmul should work when both inputs are on the same device."""
    quantizer = TernaryQuantizer(alpha=0.5, group_size=64)
    w = torch.randn(32, 64)
    x = torch.randn(4, 64)

    qt = quantizer.quantize(w)
    result = quantizer.ternary_matmul(x, qt)

    assert result.device.type == "cpu"
    assert result.shape == (4, 32)
