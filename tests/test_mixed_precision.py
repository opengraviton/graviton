"""Tests for MixedPrecisionQuantizer layer-bit selection and quantize/dequantize."""

import torch

from graviton.quantization.mixed_precision import (
    MixedPrecisionQuantizer,
    CRITICAL_PATTERNS,
    NON_CRITICAL_PATTERNS,
)
from graviton.quantization.base import QuantizedTensor


def _mq(critical=8, non_critical=4, default=4, group_size=32):
    return MixedPrecisionQuantizer(
        critical_bits=critical,
        non_critical_bits=non_critical,
        default_bits=default,
        group_size=group_size,
    )


# ── Layer bit selection ─────────────────────────────────────────────


def test_critical_layers_get_high_bits():
    mq = _mq(critical=8, non_critical=4)
    assert mq.get_layer_bits("layers.0.self_attn.q_proj") == 8
    assert mq.get_layer_bits("layers.5.self_attn.k_proj") == 8
    assert mq.get_layer_bits("embed_tokens") == 8
    assert mq.get_layer_bits("lm_head") == 8


def test_non_critical_layers_get_low_bits():
    mq = _mq(critical=8, non_critical=4)
    assert mq.get_layer_bits("layers.3.mlp.gate_proj") == 4
    assert mq.get_layer_bits("layers.3.mlp.up_proj") == 4
    assert mq.get_layer_bits("layers.3.mlp.down_proj") == 4


def test_default_bits_for_unknown_layers():
    mq = _mq(critical=8, non_critical=4, default=8)
    assert mq.get_layer_bits("some_custom_layer.weight") == 8


def test_explicit_override_takes_priority():
    mq = _mq(critical=8, non_critical=4)
    mq.set_layer_bits("layers.0.mlp.gate_proj", 2)
    assert mq.get_layer_bits("layers.0.mlp.gate_proj") == 2


def test_sensitivity_score_override():
    mq = _mq(critical=8, non_critical=4)
    mq._sensitivity_scores["custom.layer"] = 0.9
    assert mq.get_layer_bits("custom.layer") == 8  # high sensitivity -> critical

    mq._sensitivity_scores["another.layer"] = 0.05
    assert mq.get_layer_bits("another.layer") == 4  # low sensitivity -> non-critical


# ── Quantize/Dequantize roundtrip ──────────────────────────────────


def test_quantize_roundtrip_critical():
    mq = _mq(critical=8, non_critical=4)
    w = torch.randn(64, 128)
    qt = mq.quantize(w, layer_name="layers.0.self_attn.q_proj")
    assert isinstance(qt, QuantizedTensor)
    recon = mq.dequantize(qt)
    assert recon.shape == w.shape


def test_quantize_roundtrip_non_critical():
    mq = _mq(critical=8, non_critical=4)
    w = torch.randn(64, 128)
    qt = mq.quantize(w, layer_name="layers.3.mlp.gate_proj")
    assert isinstance(qt, QuantizedTensor)
    recon = mq.dequantize(qt)
    assert recon.shape == w.shape


def test_norm_layers_skipped():
    """Norm layers should be kept at 16-bit."""
    mq = _mq()
    w = torch.randn(64, 128)
    qt = mq.quantize(w, layer_name="layers.0.input_layernorm")
    assert qt.bits == 16.0


def test_small_tensor_skipped():
    """Very small tensors should be kept at 16-bit."""
    mq = _mq()
    w = torch.randn(4, 4)  # 16 elements, < 64 threshold
    qt = mq.quantize(w, layer_name="layers.0.mlp.gate_proj")
    assert qt.bits == 16.0


# ── Properties ──────────────────────────────────────────────────────


def test_name_property():
    mq = _mq(critical=8, non_critical=4)
    assert "8" in mq.name
    assert "4" in mq.name


def test_bits_property():
    mq = _mq(critical=8, non_critical=4, default=4)
    assert mq.bits == 4.0


# ── Compression report ──────────────────────────────────────────────


def test_compression_report():
    mq = _mq(critical=8, non_critical=4)
    weights = {
        "layers.0.self_attn.q_proj": torch.randn(64, 128),
        "layers.0.mlp.gate_proj": torch.randn(128, 64),
    }
    report = mq.get_compression_report(weights)
    assert report["total_original_gb"] > 0
    assert report["overall_ratio"] > 1.0
    assert len(report["layers"]) == 2
