"""
MoE (Mixture of Experts) Causal LM — Graviton inference support.

Loads Graviton-Native MoE checkpoints.
500B total params, ~10B active per token.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from graviton.memory.cache import KVCacheCompressor


class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        probs = F.softmax(top_k_logits.float(), dim=-1).to(logits.dtype)
        return logits, top_k_indices, probs


class MoEExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        num_experts = config.get("num_experts", 8)
        top_k = config.get("top_k", 2)
        intermediate = config.get("intermediate_size", hidden * 4)

        self.router = TopKRouter(hidden, num_experts, top_k)
        self.experts = nn.ModuleList([
            MoEExpert(hidden, intermediate) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        batch, seq, hidden = x.shape
        _, top_k_indices, top_k_probs = self.router(x)
        x_flat = x.view(-1, hidden)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices_flat[:, k]
            expert_weight = top_k_probs_flat[:, k:k + 1]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    output[mask] = output[mask] + expert_weight[mask] * self.experts[e](x_flat[mask])
        return output.view(batch, seq, hidden)


class MoEBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv = config.get("num_key_value_heads", num_heads)
        head_dim = hidden // num_heads

        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.moe = MoELayer(config)
        self.input_layernorm = nn.RMSNorm(hidden, eps=config.get("rms_norm_eps", 1e-5))
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=config.get("rms_norm_eps", 1e-5))
        self.num_heads = num_heads
        self.num_kv_heads = num_kv
        self.head_dim = head_dim

    def forward(self, x, kv_cache=None, position_embeddings=None):
        residual = x
        x = self.input_layernorm(x)
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v)
            cached_k, cached_v = kv_cache.get(self.layer_idx)
            if cached_k is not None:
                k = cached_k.to(q.dtype) if cached_k.dtype != q.dtype else cached_k
                v = cached_v.to(q.dtype) if cached_v.dtype != q.dtype else cached_v
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


class MoECausalLM(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        hidden = model_config["hidden_size"]
        num_heads = model_config["num_attention_heads"]
        num_layers = model_config["num_hidden_layers"]
        vocab = model_config["vocab_size"]

        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([MoEBlock(model_config, i) for i in range(num_layers)])
        self.norm = nn.RMSNorm(hidden, eps=model_config.get("rms_norm_eps", 1e-5))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.kv_cache: Optional[KVCacheCompressor] = None

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
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            if layer_skip > 1 and i % layer_skip != 0:
                continue
            h = layer(h, kv_cache=cache, position_embeddings=None)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_pretrained_dir(cls, model_dir: Path, dtype: torch.dtype = torch.float16) -> "MoECausalLM":
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
