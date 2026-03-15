"""
Omega Causal LM — Graviton inference support.

Ultra-sparse MoE + BitNet. k=1 routing, ternary weights.
Loads Graviton-Native Omega checkpoints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from graviton.models.bitnet_causal_lm import BitLinear
from graviton.memory.cache import KVCacheCompressor


class Top1Router(nn.Module):
    """k=1 router — each token uses 1 expert."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        expert_idx = logits.argmax(dim=-1)
        return logits, expert_idx


class OmegaExpert(nn.Module):
    """BitNet FFN expert — ternary, ReLU²."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.gate_proj(x)).pow(2) * self.up_proj(x))


class OmegaMoELayer(nn.Module):
    """k=1 MoE layer."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        num_experts = config.get("num_experts", 8)
        intermediate = config.get("intermediate_size", hidden * 4)
        ratio = config.get("expert_intermediate_ratio", 4)
        expert_dim = intermediate // ratio

        self.router = Top1Router(hidden, num_experts)
        self.experts = nn.ModuleList([
            OmegaExpert(hidden, expert_dim) for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, x):
        batch, seq, hidden = x.shape
        _, expert_idx = self.router(x)
        x_flat = x.view(-1, hidden)
        expert_idx_flat = expert_idx.view(-1)
        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (expert_idx_flat == e)
            if mask.any():
                output[mask] = self.experts[e](x_flat[mask])
        return output.view(batch, seq, hidden)


class OmegaBlock(nn.Module):
    """Transformer block: BitNet attention + Omega MoE."""

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv = config.get("num_key_value_heads", num_heads)
        head_dim = hidden // num_heads

        self.q_proj = BitLinear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = BitLinear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = BitLinear(num_heads * head_dim, hidden, bias=False)
        self.moe = OmegaMoELayer(config)
        self.input_layernorm = nn.RMSNorm(hidden, eps=config.get("rms_norm_eps", 1e-5))
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=config.get("rms_norm_eps", 1e-5))
        self.num_heads = num_heads
        self.num_kv_heads = num_kv
        self.head_dim = head_dim

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, kv_cache=None, position_embeddings=None):
        b, s, _ = x.shape
        residual = x
        x = self.input_layernorm(x)
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v)
            cached_k, cached_v = kv_cache.get(self.layer_idx)
            if cached_k is not None:
                k = cached_k.to(q.dtype) if cached_k.dtype != q.dtype else cached_k
                v = cached_v.to(q.dtype) if cached_v.dtype != q.dtype else cached_v
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if s > 1:
            mask = torch.triu(torch.ones(s, k.size(2), device=x.device), diagonal=1).bool()
            attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        h = torch.matmul(attn, v)
        h = h.transpose(1, 2).contiguous().view(b, s, -1)
        x = residual + self.o_proj(h)
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.moe(x)
        return x


class OmegaCausalLM(nn.Module):
    """Omega causal LM — Graviton inference."""

    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        hidden = model_config["hidden_size"]
        num_layers = model_config["num_hidden_layers"]
        vocab = model_config["vocab_size"]

        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([OmegaBlock(model_config, i) for i in range(num_layers)])
        self.norm = nn.RMSNorm(hidden, eps=model_config.get("rms_norm_eps", 1e-5))
        self.lm_head = BitLinear(hidden, vocab, bias=False)
        self.kv_cache: Optional[KVCacheCompressor] = None

        rope_theta = model_config.get("rope_theta", 10000.0)
        head_dim = hidden // model_config["num_attention_heads"]
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def init_kv_cache(self, max_length=None):
        cfg = self.model_config
        num_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        max_len = max_length or cfg.get("max_position_embeddings", 2048)
        self.kv_cache = KVCacheCompressor(
            num_layers=cfg["num_hidden_layers"],
            num_heads=num_kv,
            head_dim=head_dim,
            max_length=max_len,
        )

    def clear_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.clear()
        self.kv_cache = None

    def forward(self, input_ids, start_pos=0, layer_skip=1, kv_cache_override=None):
        _batch, seq_len = input_ids.shape
        cache = kv_cache_override if kv_cache_override is not None else self.kv_cache
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device).float()
        freqs = torch.outer(position_ids, self.inv_freq.to(input_ids.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            if layer_skip > 1 and i % layer_skip != 0:
                continue
            h = layer(h, kv_cache=cache, position_embeddings=(cos, sin))
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_pretrained_dir(cls, model_dir: Path, dtype: torch.dtype = torch.float16) -> "OmegaCausalLM":
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {model_dir}")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(config)
        bin_path = model_dir / "pytorch_model.bin"
        if bin_path.exists():
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            new_state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
            model.load_state_dict(new_state, strict=False)
        model = model.to(dtype)
        model.eval()
        return model
