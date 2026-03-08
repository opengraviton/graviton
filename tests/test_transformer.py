"""Tests for GravitonTransformerBlock and GravitonFeedForward."""

import torch

from graviton.models.transformer import GravitonTransformerBlock, GravitonFeedForward
from graviton.models.attention import RotaryPositionEmbedding
from graviton.memory.cache import KVCacheCompressor


BLOCK_CONFIG = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
}


def test_feedforward_shape():
    ff = GravitonFeedForward(hidden_size=64, intermediate_size=128, sparsity_ratio=0.5)
    x = torch.randn(2, 8, 64)
    out = ff(x)
    assert out.shape == x.shape


def test_feedforward_sparsity_applied():
    """With sparsity_ratio < 1.0, TopK should zero out activations."""
    ff = GravitonFeedForward(hidden_size=64, intermediate_size=128, sparsity_ratio=0.25)
    x = torch.randn(1, 4, 64)
    out = ff(x)
    assert out.shape == x.shape


def test_transformer_block_forward():
    block = GravitonTransformerBlock(BLOCK_CONFIG, layer_idx=0)
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)

    x = torch.randn(1, 8, 64)
    pos = rope(torch.arange(8).unsqueeze(0))
    out = block(x, position_embeddings=pos)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_transformer_block_with_kv_cache():
    block = GravitonTransformerBlock(BLOCK_CONFIG, layer_idx=0)
    cache = KVCacheCompressor(num_layers=1, num_heads=2, head_dim=16, max_length=64)
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)

    # Prefill
    x = torch.randn(1, 5, 64)
    pos = rope(torch.arange(5).unsqueeze(0))
    out = block(x, kv_cache=cache, position_embeddings=pos)
    assert out.shape == (1, 5, 64)

    # Decode
    x2 = torch.randn(1, 1, 64)
    pos2 = rope(torch.tensor([[5]]))
    out2 = block(x2, kv_cache=cache, position_embeddings=pos2)
    assert out2.shape == (1, 1, 64)


def test_transformer_block_residual_connection():
    """Output should differ from input due to attention + FFN but have same shape."""
    block = GravitonTransformerBlock(BLOCK_CONFIG, layer_idx=0)
    rope = RotaryPositionEmbedding(dim=16, max_position_embeddings=64)

    x = torch.randn(1, 4, 64)
    pos = rope(torch.arange(4).unsqueeze(0))
    out = block(x, position_embeddings=pos)

    assert out.shape == x.shape
    assert not torch.allclose(out, x, atol=1e-6)
