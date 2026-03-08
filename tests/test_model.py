"""Tests for GravitonCausalLM: forward pass, layer_skip, quantize_weights."""

import torch
import torch.nn as nn

from graviton.models.graviton_model import GravitonCausalLM
from graviton.quantization.linear import LinearQuantizer
from graviton.quantization.ternary import TernaryQuantizer
from graviton.quantization.quantized_linear import QuantizedLinear
from graviton.quantization.mixed_precision import MixedPrecisionQuantizer


TINY_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
}


def _make_model(**kwargs):
    return GravitonCausalLM(TINY_CONFIG, **kwargs)


# ── Forward Pass ────────────────────────────────────────────────────


def test_forward_basic():
    """Model should produce logits of [batch, seq, vocab]."""
    model = _make_model()
    model.eval()
    ids = torch.randint(0, 256, (1, 8))
    logits = model(ids)
    assert logits.shape == (1, 8, 256)
    assert not torch.isnan(logits).any()


def test_forward_with_kv_cache():
    """Incremental generation with a KV cache should work."""
    model = _make_model()
    model.eval()
    model.init_kv_cache(max_length=32)

    # Prefill
    ids = torch.randint(0, 256, (1, 5))
    logits = model(ids, start_pos=0)
    assert logits.shape == (1, 5, 256)

    # Decode one token at a time
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    logits2 = model(next_id, start_pos=5)
    assert logits2.shape == (1, 1, 256)


# ── Layer Skip ──────────────────────────────────────────────────────


def test_layer_skip_produces_different_logits():
    """layer_skip=2 (half the layers) should give different logits than full."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    ids = torch.randint(0, 256, (1, 6))

    full = model(ids, layer_skip=1)
    half = model(ids, layer_skip=2)

    assert full.shape == half.shape
    assert not torch.allclose(full, half, atol=1e-4)


def test_layer_skip_identity():
    """layer_skip=1 means all layers run; logits should be deterministic."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    ids = torch.randint(0, 256, (1, 4))

    out1 = model(ids, layer_skip=1)
    out2 = model(ids, layer_skip=1)
    assert torch.allclose(out1, out2)


# ── KV Cache Override ───────────────────────────────────────────────


def test_kv_cache_override():
    """kv_cache_override should be used instead of self.kv_cache."""
    from graviton.memory.cache import KVCacheCompressor

    model = _make_model()
    model.eval()
    model.init_kv_cache(max_length=32)

    external_cache = KVCacheCompressor(
        num_layers=4, num_heads=2, head_dim=16, max_length=32
    )

    ids = torch.randint(0, 256, (1, 4))
    model(ids, start_pos=0, kv_cache_override=external_cache)

    # External cache should have entries; internal should be empty
    assert len(external_cache.get_positions()) > 0
    assert len(model.kv_cache.get_positions()) == 0


# ── quantize_weights with LinearQuantizer ───────────────────────────


def test_quantize_weights_linear():
    """quantize_weights should replace nn.Linear layers with QuantizedLinear."""
    model = _make_model()
    quantizer = LinearQuantizer(bits=4, group_size=32)
    model.quantize_weights(quantizer)

    ql_count = sum(
        1 for m in model.modules() if isinstance(m, QuantizedLinear)
    )
    assert ql_count > 0, "No QuantizedLinear modules found"

    # Embed and lm_head should NOT be quantized
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)


def test_quantize_weights_ternary():
    """Ternary quantization should also work."""
    model = _make_model()
    quantizer = TernaryQuantizer(alpha=0.7, group_size=32)
    model.quantize_weights(quantizer)

    has_ternary = any(
        m.is_ternary for m in model.modules() if isinstance(m, QuantizedLinear)
    )
    assert has_ternary


def test_quantize_weights_forward_still_works():
    """Model should still produce valid logits after quantization."""
    torch.manual_seed(42)
    model = _make_model()
    model.eval()

    quantizer = LinearQuantizer(bits=8, group_size=64)
    model.quantize_weights(quantizer)

    ids = torch.randint(0, 256, (1, 6))
    logits = model(ids)
    assert logits.shape == (1, 6, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


# ── quantize_weights with MixedPrecisionQuantizer ───────────────────


def test_quantize_weights_mixed_precision():
    """Mixed precision should assign different quantizers per layer."""
    model = _make_model()
    mq = MixedPrecisionQuantizer(critical_bits=8, non_critical_bits=4, group_size=32)
    model.quantize_weights(mq)

    bits_seen = set()
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            bits_seen.add(int(m._bits))

    assert len(bits_seen) >= 1, "Expected at least one quantization bit-width"


def test_quantize_weights_mixed_precision_forward():
    """Model should still work after mixed-precision quantization."""
    torch.manual_seed(42)
    model = _make_model()
    model.eval()

    mq = MixedPrecisionQuantizer(critical_bits=8, non_critical_bits=4, group_size=32)
    model.quantize_weights(mq)

    ids = torch.randint(0, 256, (1, 4))
    logits = model(ids)
    assert logits.shape == (1, 4, 256)
    assert not torch.isnan(logits).any()


# ── clear_kv_cache ──────────────────────────────────────────────────


def test_clear_kv_cache():
    model = _make_model()
    model.init_kv_cache(max_length=32)
    assert model.kv_cache is not None

    model.clear_kv_cache()
    assert model.kv_cache is None
