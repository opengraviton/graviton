"""
Efficient Attention Mechanisms

Implements optimized attention variations:
- Flash Attention (if available)
- Grouped Query Attention (GQA)
- Multi-Query Attention (MQA)
- Support for KV-cache compression
- Rotary Position Embeddings (RoPE)
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graviton.memory.cache import KVCacheCompressor

logger = logging.getLogger(__name__)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes absolute position information into query and key vectors
    using rotation matrices, enabling relative position awareness.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos/sin embeddings for given positions.

        Args:
            position_ids: [batch, seq_len] position indices.

        Returns:
            (cos, sin) each of shape [1, 1, seq_len, dim].
        """
        freqs = torch.outer(position_ids[0].float(), self.inv_freq.to(position_ids.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EfficientAttention(nn.Module):
    """
    Optimized multi-head attention that supports Grouped Query Attention
    and compressed KV caching.
    
    This replaces standard attention to dramatically reduce memory during
    generation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize efficient attention.
        
        Args:
            hidden_size: Total hidden dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads for GQA/MQA.
                If None, uses standard MHA (num_kv_heads = num_heads).
            dropout: Attention dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        
        # GQA: Each KV head is shared by multiple Query heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = dropout

        # Check for PyTorch 2.0+ Flash Attention
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCacheCompressor] = None,
        layer_idx: Optional[int] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional KV caching and RoPE.

        Args:
            hidden_states: Input features [batch, seq_len, hidden_size]
            attention_mask: Mask for padding/causality
            kv_cache: Optional compressed KV cache
            layer_idx: Required if using kv_cache
            position_embeddings: Optional (cos, sin) from RotaryPositionEmbedding

        Returns:
            Attention output tensor.
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Manage KV Cache
        if kv_cache is not None and layer_idx is not None:
            kv_cache.update(layer_idx, key_states, value_states)
            cached_k, cached_v = kv_cache.get(layer_idx)
            if cached_k is not None and cached_v is not None:
                key_states = cached_k.to(query_states.dtype)
                value_states = cached_v.to(query_states.dtype)

        # Repeat KV heads for Grouped Query Attention
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
            value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)
            
        # Compute Attention
        if self._use_flash and not hidden_states.requires_grad:
            # Fast path: PyTorch 2.0+ Flash Attention
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True if attention_mask is None and seq_length > 1 else False,
            )
        else:
            # Slow path: Standard Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            # Upcast softmax to full precision for stability
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            if self.dropout > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
                
            attn_output = torch.matmul(attn_weights, value_states)
            
        # Reshape output: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Final projection
        output = self.o_proj(attn_output)
        
        return output
