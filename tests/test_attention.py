"""Tests for attention mask correctness (especially multi-token decode with KV cache)."""

import torch

from graviton.models.attention import (
    EfficientAttention,
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)
from graviton.memory.cache import KVCacheCompressor


def _make_attn(hidden=64, heads=4, kv_heads=2):
    return EfficientAttention(
        hidden_size=hidden, num_heads=heads, num_kv_heads=kv_heads
    )


def test_single_token_decode_no_mask():
    """Single-token decode (q_len=1) should NOT apply a causal mask."""
    attn = _make_attn()
    cache = KVCacheCompressor(num_layers=1, num_heads=2, head_dim=16, max_length=64)

    # Prefill: 5 tokens
    x_prefill = torch.randn(1, 5, 64)
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)
    pos = rope(torch.arange(5).unsqueeze(0))
    attn(x_prefill, kv_cache=cache, layer_idx=0, position_embeddings=pos)

    # Single-token decode at position 5
    x_decode = torch.randn(1, 1, 64)
    pos_dec = rope(torch.tensor([[5]]))
    out = attn(x_decode, kv_cache=cache, layer_idx=0, position_embeddings=pos_dec)

    assert out.shape == (1, 1, 64)
    assert not torch.isnan(out).any()


def test_multi_token_decode_causal_mask():
    """
    Multi-token decode (q_len > 1, kv_len > q_len) must use an explicit
    causal mask, NOT is_causal=True (which assumes q_len == kv_len).
    """
    attn = _make_attn()
    cache = KVCacheCompressor(num_layers=1, num_heads=2, head_dim=16, max_length=64)
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)

    # Prefill 4 tokens
    x_pre = torch.randn(1, 4, 64)
    pos_pre = rope(torch.arange(4).unsqueeze(0))
    attn(x_pre, kv_cache=cache, layer_idx=0, position_embeddings=pos_pre)

    # Multi-token decode: 3 new tokens at positions 4,5,6
    # KV length = 4 + 3 = 7, Q length = 3 → is_causal=True would be WRONG
    x_verify = torch.randn(1, 3, 64)
    pos_verify = rope(torch.arange(4, 7).unsqueeze(0))
    out = attn(x_verify, kv_cache=cache, layer_idx=0, position_embeddings=pos_verify)

    assert out.shape == (1, 3, 64)
    assert not torch.isnan(out).any()


def test_multi_token_prefill_uses_is_causal():
    """When q_len == kv_len (fresh prefill), is_causal=True should be used."""
    attn = _make_attn()
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)

    x = torch.randn(1, 6, 64)
    pos = rope(torch.arange(6).unsqueeze(0))
    out = attn(x, position_embeddings=pos)

    assert out.shape == (1, 6, 64)
    assert not torch.isnan(out).any()


def test_causal_mask_prevents_future_leakage():
    """
    In multi-token decode, token 0 of the new chunk must NOT attend to
    token 1 or 2 of the same chunk.

    We verify this by checking that the output for position 0 is the same
    whether we send 1 token or 3 tokens (positions 1,2 shouldn't affect 0).
    """
    torch.manual_seed(42)
    attn = _make_attn()
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)

    # Prefill 3 tokens
    x_pre = torch.randn(1, 3, 64)
    pos_pre = rope(torch.arange(3).unsqueeze(0))

    # Path A: send 1 token at position 3
    cache_a = KVCacheCompressor(num_layers=1, num_heads=2, head_dim=16, max_length=64)
    attn(x_pre, kv_cache=cache_a, layer_idx=0, position_embeddings=pos_pre)

    tok = torch.randn(1, 1, 64)
    pos_one = rope(torch.tensor([[3]]))
    out_single = attn(tok, kv_cache=cache_a, layer_idx=0, position_embeddings=pos_one)

    # Path B: send 3 tokens at positions 3,4,5
    cache_b = KVCacheCompressor(num_layers=1, num_heads=2, head_dim=16, max_length=64)
    attn(x_pre, kv_cache=cache_b, layer_idx=0, position_embeddings=pos_pre)

    extra = torch.randn(1, 2, 64)
    multi_tok = torch.cat([tok, extra], dim=1)
    pos_multi = rope(torch.arange(3, 6).unsqueeze(0))
    out_multi = attn(multi_tok, kv_cache=cache_b, layer_idx=0, position_embeddings=pos_multi)

    # Position 0 of multi-token output should match the single-token output
    # because future tokens (pos 4,5) must be masked away from pos 3.
    diff = (out_single[:, 0, :] - out_multi[:, 0, :]).abs().max().item()
    assert diff < 1e-4, f"Future token leakage detected, max diff={diff}"


def test_rope_produces_valid_cos_sin():
    """RotaryPositionEmbedding should return finite cos/sin of correct shape."""
    rope = RotaryPositionEmbedding(dim=32, max_position_embeddings=128)
    pos_ids = torch.arange(0, 10).unsqueeze(0)
    cos, sin = rope(pos_ids)

    assert cos.shape == (1, 1, 10, 32)
    assert sin.shape == (1, 1, 10, 32)
    assert not torch.isnan(cos).any()
    assert not torch.isnan(sin).any()
    assert (cos.abs() <= 1.0).all()
    assert (sin.abs() <= 1.0).all()


def test_apply_rotary_pos_emb_shapes():
    """apply_rotary_pos_emb should preserve tensor shapes."""
    q = torch.randn(1, 4, 8, 32)
    k = torch.randn(1, 2, 8, 32)
    cos = torch.randn(1, 1, 8, 32)
    sin = torch.randn(1, 1, 8, 32)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
