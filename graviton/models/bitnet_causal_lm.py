"""
BitNet b1.58 Causal LM — Graviton inference support.

Loads Graviton-Native checkpoints and Microsoft BitNet-style configs.
Uses BitLinear (ternary weights) for efficient inference.
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

logger = __import__("logging").getLogger(__name__)

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
        def forward(self, x):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            return (self.weight * x * torch.rsqrt(variance + self.eps)).to(x.dtype)


class BitLinear(nn.Module):
    """Ternary weights {-1, 0, +1} — add/subtract only matmul."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, alpha: float = 0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize_weight(self):
        w = self.weight
        absmean = w.abs().mean(dim=1, keepdim=True)
        threshold = self.alpha * absmean
        signs = w.sign()
        mask = (w.abs() > threshold).float()
        ternary = (signs * mask).to(w.dtype)
        scale = absmean.squeeze(1)
        return ternary, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary, scale = self._quantize_weight()
        w_ternary = w_ternary.to(x.device)
        scale = scale.to(x.device)
        pos_mask = (w_ternary == 1).float()
        neg_mask = (w_ternary == -1).float()
        out = (F.linear(x, pos_mask, None) - F.linear(x, neg_mask, None)) * scale.unsqueeze(0)
        if self.bias is not None:
            out = out + self.bias
        return out


class BitNetBlock(nn.Module):
    """BitNet transformer block with KV cache support."""

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv = config.get("num_key_value_heads", num_heads)
        intermediate = config.get("intermediate_size", hidden * 4)
        head_dim = hidden // num_heads

        self.q_proj = BitLinear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = BitLinear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = BitLinear(num_heads * head_dim, hidden, bias=False)
        self.gate_proj = BitLinear(hidden, intermediate, bias=False)
        self.up_proj = BitLinear(hidden, intermediate, bias=False)
        self.down_proj = BitLinear(intermediate, hidden, bias=False)

        self.input_layernorm = _RMSNorm(hidden, eps=config.get("rms_norm_eps", 1e-5))
        self.post_attention_layernorm = _RMSNorm(hidden, eps=config.get("rms_norm_eps", 1e-5))

        self.num_heads = num_heads
        self.num_kv_heads = num_kv
        self.head_dim = head_dim
        self.hidden_size = hidden

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _relu2(self, x):
        return F.relu(x).pow(2)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCacheCompressor] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        b, s, _ = x.shape
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v)
            cached_k, cached_v = kv_cache.get(self.layer_idx)
            if cached_k is not None:
                k = cached_k.to(q.dtype) if cached_k.dtype != q.dtype else cached_k
                v = cached_v.to(q.dtype) if cached_v.dtype != q.dtype else cached_v

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

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
        gate = self._relu2(self.gate_proj(x))
        x = residual + self.down_proj(gate * self.up_proj(x))
        return x


class BitNetCausalLM(nn.Module):
    """BitNet causal LM — Graviton engine compatible."""

    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        hidden = model_config["hidden_size"]
        num_heads = model_config["num_attention_heads"]
        num_layers = model_config["num_hidden_layers"]
        vocab = model_config["vocab_size"]

        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([BitNetBlock(model_config, i) for i in range(num_layers)])
        self.norm = _RMSNorm(hidden, eps=model_config.get("rms_norm_eps", 1e-5))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        head_dim = hidden // num_heads
        inv_freq = 1.0 / (model_config.get("rope_theta", 10000.0) ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq)
        self.kv_cache: Optional[KVCacheCompressor] = None

    def init_kv_cache(self, max_length: Optional[int] = None):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        layer_skip: int = 1,
        kv_cache_override: Optional[KVCacheCompressor] = None,
    ) -> torch.Tensor:
        _batch, seq_len = input_ids.shape
        device = input_ids.device
        cache = kv_cache_override if kv_cache_override is not None else self.kv_cache

        h = self.embed_tokens(input_ids)
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.long).unsqueeze(0)
        freqs = torch.outer(position_ids[0].float(), self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            if layer_skip > 1 and i % layer_skip != 0:
                continue
            h = layer(h, kv_cache=cache, position_embeddings=(cos, sin))

        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_pretrained_dir(cls, model_dir: Path, dtype: torch.dtype = torch.float16) -> "BitNetCausalLM":
        """Load BitNet from Graviton-Native or BitNet-style checkpoint."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {model_dir}")

        with open(config_path) as f:
            config = json.load(f)

        # Normalize config keys (BitNet vs Graviton-Native)
        if "model_type" in config and config["model_type"] == "bitnet":
            pass  # Microsoft format
        # Ensure required keys
        for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]:
            if key not in config:
                raise ValueError(f"config.json missing {key}")

        model = cls(config)

        # Load weights
        bin_path = model_dir / "pytorch_model.bin"
        if not bin_path.exists():
            bin_path = model_dir / "model.safetensors"
        if bin_path.exists():
            if bin_path.suffix == ".bin":
                state = torch.load(bin_path, map_location="cpu", weights_only=True)
            else:
                from safetensors import safe_open
                state = {}
                with safe_open(str(bin_path), framework="pt") as sf:
                    for k in sf.keys():
                        state[k] = sf.get_tensor(k)
            # Strip "model." prefix if present
            new_state = {}
            for k, v in state.items():
                new_k = k[len("model."):] if k.startswith("model.") else k
                new_state[new_k] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            if missing:
                logger.warning(f"BitNet load missing: {missing[:5]}...")
        else:
            logger.warning(f"No weights found in {model_dir}, using random init")

        model = model.to(dtype)
        model.eval()
        return model
