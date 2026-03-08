"""
Graviton-Optimized Transformer

A highly efficient transformer block implementation designed to take
full advantage of Graviton's quantization and sparsity engines.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from graviton.models.attention import EfficientAttention
from graviton.sparsity.topk import TopKActivation

logger = logging.getLogger(__name__)

try:
    _RMSNorm = nn.RMSNorm
except AttributeError:

    class _RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight * x).to(x.dtype)


class GravitonFeedForward(nn.Module):
    """
    Sparsity-aware Feed-Forward Network.
    
    Replaces standard dense activations with Top-K sparse activations,
    reducing computation significantly.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, sparsity_ratio: float = 0.5):
        super().__init__()
        # Standard LLaMA-style SwiGLU FFN consists of gate, up, and down projections
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Graviton sparse activation
        self.act_fn = TopKActivation(k_ratio=sparsity_ratio)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # Top-K forces many of these values to strictly zero
        gate = self.act_fn(self.gate_proj(x))
        
        # If gate is highly sparse, the element-wise multiply with up_proj
        # will also be highly sparse. An optimized engine would skip the
        # corresponding compute for down_proj entirely.
        return self.down_proj(gate * self.up_proj(x))


class GravitonTransformerBlock(nn.Module):
    """
    A single layer of a transformer model, optimized for Graviton.
    
    Integrates efficient attention, sparse FFN, and supports mixed
    precision quantization seamlessly.
    """
    
    def __init__(self, config: dict, layer_idx: int, engine_config=None):
        """
        Initialize the transformer block.
        
        Args:
            config: Model architecture config (hidden_size, etc.)
            layer_idx: The index of this layer (used for KV caching)
            engine_config: Graviton engine configuration
        """
        super().__init__()
        self.hidden_size = config.get("hidden_size", 4096)
        self.layer_idx = layer_idx
        
        # Extract sparsity config
        sparsity_ratio = 1.0
        if engine_config is not None:
            sparsity_ratio = engine_config.sparsity.k_ratio
            
        # Attention
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        self.self_attn = EfficientAttention(
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        
        # FFN
        intermediate_size = config.get("intermediate_size", 11008)
        self.mlp = GravitonFeedForward(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            sparsity_ratio=sparsity_ratio,
        )
        
        self.input_layernorm = _RMSNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = _RMSNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        position_embeddings=None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        """
        # 1. Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # 2. MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
