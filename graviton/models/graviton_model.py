"""
Graviton Causal Language Model

Full model assembly: embedding -> transformer layers -> norm -> lm_head.
Supports LLaMA-family architectures (LLaMA, TinyLlama, Mistral, etc.).
"""

from __future__ import annotations

import gc
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Callable

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

    # ------------------------------------------------------------------
    # Streaming weight loading (for large models)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_dir_streaming(
        cls,
        model_dir: Path,
        engine_config=None,
        dtype: torch.dtype = torch.float32,
        quantizer=None,
        target_device: torch.device = torch.device("cpu"),
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> "GravitonCausalLM":
        """
        Stream-load a large model layer by layer with on-the-fly quantization.

        Peak memory ≈ 1 FP16 layer + all previously quantized layers,
        enabling models that far exceed available RAM in their FP16 form.
        """
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {model_dir}")

        with open(config_path) as f:
            model_config = json.load(f)

        num_layers = model_config["num_hidden_layers"]

        def _progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        _progress("Building model skeleton...")

        weight_index = cls._build_weight_index(model_dir, model_config)

        with torch.device("meta"):
            model = cls(model_config, engine_config)

        # RoPE uses register_buffer; recreate on CPU to avoid meta tensors
        head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
        model.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_position_embeddings=model_config.get("max_position_embeddings", 2048),
            base=model_config.get("rope_theta", 10000.0),
        )

        # --- Non-layer weights (embeddings, final norm, lm_head) ---
        _progress("Loading embeddings & head...")

        non_layer_keys = {
            k: v for k, v in weight_index.items()
            if not k.startswith("layers.")
        }

        model.embed_tokens = model.embed_tokens.to_empty(device="cpu")
        model.norm = model.norm.to_empty(device="cpu")
        model.lm_head = model.lm_head.to_empty(device="cpu")

        non_layer_state = cls._batch_load_tensors(non_layer_keys)
        model.load_state_dict(non_layer_state, strict=False)
        del non_layer_state

        model.embed_tokens = model.embed_tokens.to(dtype).to(target_device)
        model.norm = model.norm.to(dtype).to(target_device)
        model.lm_head = model.lm_head.to(dtype).to(target_device)
        model.rope = model.rope.to(target_device)

        # --- Transformer layers: load → quantize → move, one at a time ---
        for i in range(num_layers):
            _progress(f"Loading layer {i + 1}/{num_layers}...")

            layer_prefix = f"layers.{i}."
            layer_keys = {
                k: v for k, v in weight_index.items()
                if k.startswith(layer_prefix)
            }

            model.layers[i] = model.layers[i].to_empty(device="cpu")

            layer_state = {}
            shards = defaultdict(list)
            for param_name, (shard_file, orig_key) in layer_keys.items():
                shards[shard_file].append((param_name, orig_key))

            for shard_file, keys in shards.items():
                tensors = cls._load_tensors_from_shard(shard_file, [k for _, k in keys])
                for param_name, orig_key in keys:
                    local_name = param_name[len(layer_prefix):]
                    layer_state[local_name] = tensors[orig_key].to(dtype)

            model.layers[i].load_state_dict(layer_state, strict=False)
            del layer_state

            model.layers[i] = model.layers[i].to(target_device)

            if quantizer:
                cls._quantize_single_layer(model.layers[i], quantizer, f"layers.{i}")

            gc.collect()
            if target_device.type == "mps":
                torch.mps.empty_cache()

        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_gb = (param_bytes + buf_bytes) / (1024 ** 3)
        _progress(f"Model ready: {param_count / 1e9:.1f}B params, {total_gb:.1f} GB")

        return model

    @classmethod
    def _build_weight_index(
        cls, model_dir: Path, model_config: dict,
    ) -> Dict[str, tuple]:
        """
        Map every remapped weight name to ``(shard_file_path, original_key)``.

        Supports sharded safetensors (with index.json), single safetensors,
        and legacy pytorch .bin files.
        """
        index: Dict[str, tuple] = {}

        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                shard_data = json.load(f)
            for orig_key, shard_name in shard_data.get("weight_map", {}).items():
                remapped = orig_key
                if remapped.startswith("model."):
                    remapped = remapped[len("model."):]
                index[remapped] = (str(model_dir / shard_name), orig_key)
        else:
            sf_files = sorted(model_dir.glob("*.safetensors"))
            if sf_files:
                from safetensors import safe_open

                for sf_path in sf_files:
                    with safe_open(str(sf_path), framework="pt") as sf:
                        for orig_key in sf.keys():
                            remapped = orig_key
                            if remapped.startswith("model."):
                                remapped = remapped[len("model."):]
                            index[remapped] = (str(sf_path), orig_key)
            else:
                for bin_path in sorted(model_dir.glob("*.bin")):
                    state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
                    for orig_key in state.keys():
                        remapped = orig_key
                        if remapped.startswith("model."):
                            remapped = remapped[len("model."):]
                        index[remapped] = (str(bin_path), orig_key)
                    del state

        if model_config.get("tie_word_embeddings", False):
            if "lm_head.weight" not in index and "embed_tokens.weight" in index:
                index["lm_head.weight"] = index["embed_tokens.weight"]

        logger.info(f"Weight index: {len(index)} tensors mapped")
        return index

    @staticmethod
    def _load_tensors_from_shard(
        shard_path: str, keys: list[str],
    ) -> Dict[str, torch.Tensor]:
        """Load specific tensors from a single shard file (safetensors or bin)."""
        tensors: Dict[str, torch.Tensor] = {}
        if shard_path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for key in keys:
                    tensors[key] = sf.get_tensor(key)
        else:
            state = torch.load(shard_path, map_location="cpu", weights_only=True)
            for key in keys:
                tensors[key] = state[key]
        return tensors

    @classmethod
    def _batch_load_tensors(
        cls, key_map: Dict[str, tuple],
    ) -> Dict[str, torch.Tensor]:
        """Load a set of tensors, batching reads by shard file."""
        shards: Dict[str, list] = defaultdict(list)
        for param_name, (shard_file, orig_key) in key_map.items():
            shards[shard_file].append((param_name, orig_key))

        result: Dict[str, torch.Tensor] = {}
        for shard_file, keys in shards.items():
            tensors = cls._load_tensors_from_shard(shard_file, [k for _, k in keys])
            for param_name, orig_key in keys:
                result[param_name] = tensors[orig_key]
        return result

    @staticmethod
    def _quantize_single_layer(layer: nn.Module, quantizer, layer_name_prefix: str):
        """Quantize all nn.Linear modules in a single transformer layer in-place."""
        from graviton.quantization.quantized_linear import QuantizedLinear
        from graviton.quantization.mixed_precision import MixedPrecisionQuantizer

        is_mixed = isinstance(quantizer, MixedPrecisionQuantizer)
        skip_patterns = ["norm"]

        for name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if any(p in name for p in skip_patterns):
                continue

            full_name = f"{layer_name_prefix}.{name}"
            if is_mixed:
                bits = quantizer.get_layer_bits(full_name)
                lq = quantizer._get_quantizer(bits)
            else:
                lq = quantizer

            qlinear = QuantizedLinear.from_linear(module, lq)

            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], qlinear)
