"""Tests for GravitonConfig convenience constructors and flag propagation."""

from graviton.core.config import (
    GravitonConfig,
    QuantMode,
    DeviceType,
    QuantizationConfig,
    DecodingConfig,
)


# ── Convenience parameter propagation ───────────────────────────────


def test_quant_bits_sets_mode():
    cfg = GravitonConfig(quant_bits=8)
    assert cfg.quantization.mode == QuantMode.INT8

    cfg4 = GravitonConfig(quant_bits=4)
    assert cfg4.quantization.mode == QuantMode.INT4

    cfg_t = GravitonConfig(quant_bits=1.58)
    assert cfg_t.quantization.mode == QuantMode.TERNARY


def test_use_speculative_flag_propagation():
    cfg = GravitonConfig(use_speculative=True)
    assert cfg.decoding.use_speculative is True

    cfg2 = GravitonConfig(use_speculative=False)
    assert cfg2.decoding.use_speculative is False


def test_sparsity_ratio_propagation():
    cfg = GravitonConfig(sparsity_ratio=0.3)
    assert cfg.sparsity.k_ratio == 0.3


def test_max_memory_propagation():
    cfg = GravitonConfig(max_memory_gb=16.0)
    assert cfg.memory.max_memory_gb == 16.0


def test_mmap_propagation():
    cfg = GravitonConfig(use_mmap=False)
    assert cfg.memory.use_mmap is False


# ── Preset constructors ────────────────────────────────────────────


def test_for_mac_mini():
    cfg = GravitonConfig.for_mac_mini(memory_gb=16)
    assert cfg.device == DeviceType.MPS
    assert cfg.quantization.mode == QuantMode.INT4
    assert cfg.decoding.use_speculative is True
    assert cfg.memory.max_memory_gb < 16.0


def test_for_extreme_compression():
    cfg = GravitonConfig.for_extreme_compression()
    assert cfg.quantization.mode == QuantMode.TERNARY
    assert cfg.sparsity.k_ratio <= 0.5
    assert cfg.decoding.use_speculative is True


def test_for_quality():
    cfg = GravitonConfig.for_quality()
    assert cfg.quantization.mode == QuantMode.INT8
    assert cfg.decoding.use_speculative is False


# ── QuantMode properties ────────────────────────────────────────────


def test_quant_mode_bits():
    assert QuantMode.NONE.bits == 16.0
    assert QuantMode.INT8.bits == 8.0
    assert QuantMode.INT4.bits == 4.0
    assert QuantMode.INT2.bits == 2.0
    assert 1.5 < QuantMode.TERNARY.bits < 1.6


# ── QuantizationConfig helpers ──────────────────────────────────────


def test_compression_ratio():
    qcfg = QuantizationConfig(mode=QuantMode.INT4)
    assert qcfg.estimated_compression_ratio() == 4.0


def test_effective_bits():
    qcfg = QuantizationConfig(mode=QuantMode.INT8)
    assert qcfg.effective_bits == 8.0


# ── Memory estimation ──────────────────────────────────────────────


def test_estimate_memory_usage():
    cfg = GravitonConfig(quant_bits=4)
    est = cfg.estimate_memory_usage(num_params=1_000_000_000)
    assert est["weights_gb"] > 0
    assert est["total_gb"] > est["weights_gb"]
    assert est["compression_ratio"] == 4.0


# ── Summary ─────────────────────────────────────────────────────────


def test_summary_returns_string():
    cfg = GravitonConfig(quant_bits=4, sparsity_ratio=0.5)
    s = cfg.summary()
    assert "Graviton" in s
    assert "int4" in s
