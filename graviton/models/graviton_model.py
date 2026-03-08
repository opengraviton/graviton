"""
Graviton Causal Language Model

Full model assembly: embedding -> transformer layers -> norm -> lm_head.
Supports LLaMA-family architectures (LLaMA, TinyLlama, Mistral, etc.).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn

from graviton.models.transformer import GravitonTransformerBlock
from graviton.models.attention import RotaryPositionEmbedding
from graviton.memory.cache import KVCacheCompressor

logger = logging.getLogger(__name__)


# nn.RMSNorm was added in PyTorch 2.4; provide a fallback for older versions
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


class GravitonCausalLM(nn.Module):
    """
    Complete causal language model for autoregressive text generation.

    Assembles embedding, transformer layers, final norm, and the language
    model head into a single nn.Module that can be loaded with pretrained
    HuggingFace weights and used for token-by-token generation.
    """

    SUPPORTED_ARCHITECTURES = [
        "LlamaForCausalLM",
        "MistralForCausalLM",
    ]

    def __init__(self, model_config: dict, engine_config=None):
        super().__init__()
        self.model_config = model_config

        vocab_size = model_config["vocab_size"]
        hidden_size = model_config["hidden_size"]
        num_layers = model_config["num_hidden_layers"]
        num_heads = model_config["num_attention_heads"]
        head_dim = hidden_size // num_heads

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            GravitonTransformerBlock(model_config, layer_idx=i, engine_config=engine_config)
            for i in range(num_layers)
        ])

        self.norm = _RMSNorm(hidden_size, eps=model_config.get("rms_norm_eps", 1e-5))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_position_embeddings=model_config.get("max_position_embeddings", 2048),
            base=model_config.get("rope_theta", 10000.0),
        )

        self.kv_cache: Optional[KVCacheCompressor] = None

    def init_kv_cache(self, max_length: Optional[int] = None):
        """Allocate a fresh KV cache for generation."""
        cfg = self.model_config
        num_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        max_len = max_length or cfg.get("max_position_embeddings", 2048)

        self.kv_cache = KVCacheCompressor(
            num_layers=cfg["num_hidden_layers"],
            num_heads=num_kv_heads,
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
        """
        Forward pass through the full model.

        Args:
            input_ids: Token IDs [batch, seq_len].
            start_pos: Position offset for RoPE (0 for prefill, increments during decode).
            layer_skip: Process every Nth layer (1 = all layers, 2 = half, etc.).
                        Used for speculative decoding draft mode.
            kv_cache_override: Explicit KV cache to use instead of ``self.kv_cache``.

        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        _batch, seq_len = input_ids.shape
        device = input_ids.device
        cache = kv_cache_override if kv_cache_override is not None else self.kv_cache

        h = self.embed_tokens(input_ids)

        position_ids = torch.arange(
            start_pos, start_pos + seq_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        position_embeddings = self.rope(position_ids)

        for i, layer in enumerate(self.layers):
            if layer_skip > 1 and i % layer_skip != 0:
                continue
            h = layer(h, kv_cache=cache, position_embeddings=position_embeddings)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    # ------------------------------------------------------------------
    # Post-load quantization
    # ------------------------------------------------------------------

    def quantize_weights(self, quantizer):
        """
        Replace all eligible nn.Linear layers with QuantizedLinear.

        Embedding, layer norms, and lm_head are kept at full precision
        to preserve model quality.  For ``MixedPrecisionQuantizer`` the
        per-layer bit width is determined by the layer name.
        """
        from graviton.quantization.quantized_linear import QuantizedLinear
        from graviton.quantization.mixed_precision import MixedPrecisionQuantizer

        skip_patterns = ["embed", "norm", "lm_head"]
        is_mixed = isinstance(quantizer, MixedPrecisionQuantizer)
        count = 0
        saved_bytes = 0

        for name, module in list(self.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if any(p in name for p in skip_patterns):
                continue

            if is_mixed:
                bits = quantizer.get_layer_bits(name)
                layer_quantizer = quantizer._get_quantizer(bits)
            else:
                layer_quantizer = quantizer

            orig_bytes = module.weight.numel() * module.weight.element_size()
            qlinear = QuantizedLinear.from_linear(module, layer_quantizer)
            packed_bytes = qlinear.packed_size_bytes
            saved_bytes += orig_bytes - packed_bytes

            parts = name.split(".")
            parent = self
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], qlinear)
            count += 1

        logger.info(
            f"Quantized {count} linear layers, "
            f"saved {saved_bytes / (1024**2):.0f} MB in packed storage"
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_dir(
        cls,
        model_dir: Path,
        engine_config=None,
        dtype: torch.dtype = torch.float32,
    ) -> "GravitonCausalLM":
        """
        Build a model from a local HuggingFace-format directory.

        Reads config.json, constructs the architecture, and loads the
        safetensors / pytorch-bin weights with automatic name remapping.
        """
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {model_dir}")

        with open(config_path) as f:
            model_config = json.load(f)

        logger.info(
            f"Building model: {model_config.get('architectures', ['unknown'])[0]}, "
            f"{model_config['num_hidden_layers']} layers, "
            f"hidden={model_config['hidden_size']}"
        )

        model = cls(model_config, engine_config)

        raw_weights = cls._load_raw_weights(model_dir)
        state_dict = cls._remap_weight_names(raw_weights, model_config)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading weights: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.debug(f"Unexpected keys (ignored): {unexpected[:5]}")

        model = model.to(dtype)
        return model

    @staticmethod
    def _load_raw_weights(model_dir: Path) -> Dict[str, torch.Tensor]:
        """Load all weight tensors from safetensors or pytorch bin files."""
        weights: Dict[str, torch.Tensor] = {}

        safetensors_files = sorted(model_dir.glob("*.safetensors"))
        if safetensors_files:
            from safetensors import safe_open

            for f in safetensors_files:
                with safe_open(str(f), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        weights[key] = sf.get_tensor(key)
            return weights

        bin_files = sorted(model_dir.glob("*.bin"))
        for f in bin_files:
            state = torch.load(str(f), map_location="cpu", weights_only=True)
            weights.update(state)
        return weights

    @staticmethod
    def _remap_weight_names(
        weights: Dict[str, torch.Tensor],
        model_config: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Remap HuggingFace weight names to our model's state_dict keys.

        HF LLaMA uses 'model.layers.X...' while our model uses 'layers.X...'.
        """
        mapped: Dict[str, torch.Tensor] = {}
        tie_word_embeddings = model_config.get("tie_word_embeddings", False)

        for key, tensor in weights.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            mapped[new_key] = tensor

        if tie_word_embeddings and "lm_head.weight" not in mapped:
            if "embed_tokens.weight" in mapped:
                mapped["lm_head.weight"] = mapped["embed_tokens.weight"]

        return mapped
