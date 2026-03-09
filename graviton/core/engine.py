"""
Graviton Inference Engine

The main orchestrator that combines quantization, sparsity,
memory management, and decoding into a unified inference pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Generator, Union

import torch
import numpy as np

from graviton.core.config import GravitonConfig, QuantMode, DeviceType
from graviton.core.hardware import HardwareProfile, detect_hardware, recommend_config
from graviton.quantization.linear import LinearQuantizer
from graviton.quantization.ternary import TernaryQuantizer
from graviton.quantization.mixed_precision import MixedPrecisionQuantizer
from graviton.memory.manager import MemoryManager
from graviton.memory.streaming import LayerStreamer
from graviton.sparsity.topk import TopKActivation
from graviton.sparsity.pruning import DynamicPruner
from graviton.decoding.speculative import SpeculativeDecoder
from graviton.decoding.sampling import Sampler

logger = logging.getLogger(__name__)


class GravitonEngine:
    """
    Ultra-efficient AI inference engine.

    Orchestrates quantization, sparsity, memory management, and
    decoding to run massive models on minimal hardware.

    Example:
        >>> from graviton import GravitonEngine, GravitonConfig
        >>> config = GravitonConfig(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_bits=4)
        >>> engine = GravitonEngine(config=config)
        >>> engine.load_model()
        >>> print(engine.generate("Hello!", max_tokens=50))
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[GravitonConfig] = None,
    ):
        self.config = config or GravitonConfig()
        self.model_path = model_path or self.config.model_path

        self.hardware = detect_hardware()
        self._auto_configure()

        self.quantizer = self._init_quantizer()
        self.memory_manager = MemoryManager(self.config.memory, self.hardware)
        self.sparsity_engine = self._init_sparsity()
        self.sampler = Sampler(self.config.decoding)

        # Model state
        self._model_loaded = False
        self._model_weights = {}
        self._layer_count = 0

        # Inference model and tokenizer (populated by load_model)
        self._model = None
        self._tokenizer = None
        self._model_config = None

        # Optional callback for reporting loading progress (e.g. to UI)
        self.progress_callback = None

        logger.info("Graviton Engine initialized")
        if self.config.verbose:
            print(self.hardware.summary())
            print()
            print(self.config.summary())

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _auto_configure(self):
        """Auto-configure based on detected hardware."""
        if self.config.device == DeviceType.AUTO:
            if self.hardware.has_mps:
                self.config.device = DeviceType.MPS
            elif self.hardware.has_cuda:
                self.config.device = DeviceType.CUDA
            else:
                self.config.device = DeviceType.CPU

        if self.config.memory.max_memory_gb <= 0:
            self.config.memory.max_memory_gb = self.hardware.available_memory_gb * 0.8
            logger.info(
                f"Auto-set memory budget: {self.config.memory.max_memory_gb:.1f} GB"
            )

    def _init_quantizer(self):
        """Initialize the appropriate quantizer."""
        mode = self.config.quantization.mode
        if self.config.quantization.use_mixed_precision:
            return MixedPrecisionQuantizer(self.config.quantization)
        if mode == QuantMode.TERNARY:
            return TernaryQuantizer()
        elif mode in (QuantMode.INT8, QuantMode.INT4, QuantMode.INT2):
            return LinearQuantizer(
                bits=int(mode.bits),
                group_size=self.config.quantization.group_size,
                symmetric=self.config.quantization.symmetric,
            )
        return None

    def _init_sparsity(self):
        """Initialize the sparsity engine."""
        from graviton.core.config import SparsityMode

        mode = self.config.sparsity.mode
        if mode == SparsityMode.TOPK:
            return TopKActivation(k_ratio=self.config.sparsity.k_ratio)
        elif mode == SparsityMode.MAGNITUDE:
            return DynamicPruner(threshold=self.config.sparsity.pruning_threshold)
        return None

    @property
    def device(self) -> torch.device:
        """Get the target torch device."""
        device_map = {
            DeviceType.CPU: "cpu",
            DeviceType.MPS: "mps",
            DeviceType.CUDA: "cuda",
            DeviceType.AUTO: "cpu",
        }
        return torch.device(device_map[self.config.device])

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype based on device."""
        if self.config.device in (DeviceType.MPS, DeviceType.CUDA):
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: Optional[str] = None):
        """
        Load a model for inference.

        Downloads from HuggingFace if the path is not a local directory,
        builds the full transformer model, loads weights, and prepares
        the tokenizer.
        """
        path = model_path or self.model_path
        if path is None:
            raise ValueError("No model path specified")

        logger.info(f"Loading model from: {path}")
        start_time = time.time()

        self._report_progress("Downloading model files...")
        model_dir = self._resolve_model_dir(path)
        self._build_inference_model(model_dir)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")
        self._model_loaded = True

    def _resolve_model_dir(self, path: str) -> Path:
        """
        Ensure the model exists locally, downloading from HuggingFace
        Hub if necessary. Returns the local directory Path.
        """
        local = Path(path)
        if local.is_dir() and (
            list(local.glob("*.safetensors")) or list(local.glob("*.bin"))
        ):
            return local

        try:
            from huggingface_hub import snapshot_download, HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install with: pip install graviton-ai[huggingface]"
            )

        short_name = path.split("/")[-1]
        self._report_progress(f"Downloading {short_name}...")

        # Determine download size and choose the right file patterns
        allow_patterns = ["*.json", "*.model", "tokenizer*"]
        total_bytes = 0

        try:
            api = HfApi()
            info = api.model_info(path, files_metadata=True)
            safetensor_files = [
                s for s in info.siblings
                if s.rfilename.endswith(".safetensors") and s.size
            ]

            has_sharded = any(
                "model-" in s.rfilename and "-of-" in s.rfilename
                for s in safetensor_files
            )
            has_consolidated = any(
                s.rfilename == "consolidated.safetensors"
                for s in safetensor_files
            )

            if has_sharded:
                allow_patterns.append("model-*.safetensors")
                allow_patterns.append("model.safetensors.index.json")
                total_bytes = sum(
                    s.size for s in safetensor_files
                    if "model-" in s.rfilename and "-of-" in s.rfilename
                )
            elif has_consolidated:
                allow_patterns.append("consolidated.safetensors")
                total_bytes = sum(
                    s.size for s in safetensor_files
                    if s.rfilename == "consolidated.safetensors"
                )
            else:
                allow_patterns.append("*.safetensors")
                total_bytes = sum(s.size for s in safetensor_files)

            if total_bytes > 0:
                gb = total_bytes / (1024 ** 3)
                self._report_progress(
                    f"Downloading {short_name} ({gb:.1f} GB)..."
                )
        except Exception:
            allow_patterns.append("*.safetensors")

        # Start a background monitor to track download progress via cache
        import threading
        download_done = threading.Event()
        cache_dir = None

        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = Path(HF_HUB_CACHE)
        except Exception:
            pass

        def _monitor_download():
            if not cache_dir or total_bytes <= 0:
                return
            model_cache = cache_dir / f"models--{path.replace('/', '--')}"
            blobs_dir = model_cache / "blobs"
            prev_bytes = 0
            import time as _time
            prev_time = _time.monotonic()
            speed_mbps = 0.0
            while not download_done.is_set():
                try:
                    if blobs_dir.exists():
                        all_files = [
                            f for f in blobs_dir.iterdir() if f.is_file()
                        ]
                        incomplete = [
                            f for f in all_files
                            if f.name.endswith(".incomplete")
                        ]
                        if not incomplete:
                            self._report_progress(
                                f"Preparing {short_name}..."
                            )
                        else:
                            downloaded = sum(
                                f.stat().st_size for f in all_files
                            )
                            now = _time.monotonic()
                            dt = now - prev_time
                            if dt > 0.5 and prev_bytes > 0:
                                delta = downloaded - prev_bytes
                                speed_mbps = max(
                                    0, (delta / (1024 ** 2)) / dt
                                )
                            prev_bytes = downloaded
                            prev_time = now
                            capped = min(downloaded, total_bytes)
                            gb_done = capped / (1024 ** 3)
                            gb_total = total_bytes / (1024 ** 3)
                            speed_str = ""
                            if speed_mbps > 1:
                                speed_str = f" @ {speed_mbps:.0f} MB/s"
                            self._report_progress(
                                f"Downloading {short_name} "
                                f"({gb_done:.2f} / {gb_total:.2f} GB"
                                f"{speed_str})"
                            )
                except Exception:
                    pass
                download_done.wait(timeout=1)

        # Fast path: if model is already cached, skip network entirely
        try:
            local_dir = snapshot_download(
                repo_id=path,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                local_files_only=True,
            )
            self._report_progress(f"Found {short_name} in cache")
            return Path(local_dir)
        except Exception:
            pass

        monitor = threading.Thread(target=_monitor_download, daemon=True)
        monitor.start()

        try:
            local_dir = snapshot_download(
                repo_id=path,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            return Path(local_dir)
        except Exception as exc:
            msg = str(exc)
            if "gated" in msg.lower() or "access" in msg.lower():
                raise RuntimeError(
                    f"Access denied for '{path}'. This model requires you to "
                    f"accept its license at https://huggingface.co/{path} "
                    f"and set a HuggingFace token."
                ) from exc
            if "401" in msg or "unauthorized" in msg.lower():
                raise RuntimeError(
                    f"Authentication failed for '{path}'. "
                    f"Check your HuggingFace token."
                ) from exc
            if "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Model '{path}' not found on HuggingFace."
                ) from exc
            raise
        finally:
            download_done.set()
            monitor.join(timeout=5)

    def _report_progress(self, message: str):
        """Forward a progress message to the optional callback."""
        logger.info(message)
        if self.progress_callback:
            self.progress_callback(message)

    def _estimate_fp16_gb(self, model_dir: Path) -> float:
        """Estimate FP16 model size in GB from config.json."""
        import json as _json

        config_path = model_dir / "config.json"
        with open(config_path) as f:
            mc = _json.load(f)

        hidden = mc.get("hidden_size", 4096)
        num_layers = mc.get("num_hidden_layers", 32)
        vocab = mc.get("vocab_size", 32000)
        intermediate = mc.get("intermediate_size", hidden * 4)
        num_kv_heads = mc.get("num_key_value_heads", mc.get("num_attention_heads", 32))
        num_heads = mc.get("num_attention_heads", 32)
        head_dim = hidden // num_heads

        attn_params = hidden * (num_heads + 2 * num_kv_heads + num_heads) * head_dim
        mlp_params = 3 * hidden * intermediate
        per_layer = attn_params + mlp_params
        total = per_layer * num_layers + vocab * hidden * 2
        return (total * 2) / (1024 ** 3)

    def _is_bitnet_model(self, model_dir: Path) -> bool:
        """Detect BitNet / Graviton-Native checkpoint."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        return (
            cfg.get("model_type") == "bitnet"
            or cfg.get("use_ternary_weights") is True
            or (cfg.get("architectures") or [""])[0] == "BitNetForCausalLM"
        )

    def _is_moe_model(self, model_dir: Path) -> bool:
        """Detect MoE / Graviton-Native checkpoint."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("num_experts", 0) > 0

    def _build_inference_model(self, model_dir: Path):
        """
        Build the full inference model from a local directory.

        Automatically chooses between direct loading (small models that
        fit in RAM as FP16) and streaming layer-by-layer loading with
        on-the-fly quantization (large models).
        BitNet / Graviton-Native models use native ternary inference.
        """
        from graviton.models.graviton_model import GravitonCausalLM
        from graviton.models.bitnet_causal_lm import BitNetCausalLM
        from graviton.core.config import QuantMode

        if self._is_bitnet_model(model_dir):
            self._report_progress("Loading BitNet model (native ternary)...")
            self._model = BitNetCausalLM.from_pretrained_dir(model_dir, dtype=self.dtype)
            self._model.to(self.device)
            self._model.eval()
            self._model_config = self._model.model_config
            self._load_tokenizer(model_dir)
            return

        if self._is_moe_model(model_dir):
            from graviton.models.moe_causal_lm import MoECausalLM
            self._report_progress("Loading MoE model (Mixture of Experts)...")
            self._model = MoECausalLM.from_pretrained_dir(model_dir, dtype=self.dtype)
            self._model.to(self.device)
            self._model.eval()
            self._model_config = self._model.model_config
            self._load_tokenizer(model_dir)
            return

        fp16_gb = self._estimate_fp16_gb(model_dir)
        use_streaming = fp16_gb > self.hardware.available_memory_gb * 0.7

        if use_streaming:
            self._report_progress(
                f"Large model (~{fp16_gb:.0f} GB FP16) — streaming layer-by-layer..."
            )

            quantizer = None
            if (
                self.quantizer is not None
                and self.config.quantization.mode != QuantMode.NONE
            ):
                quantizer = self.quantizer

            self._model = GravitonCausalLM.from_pretrained_dir_streaming(
                model_dir,
                engine_config=self.config,
                dtype=self.dtype,
                quantizer=quantizer,
                target_device=self.device,
                progress_callback=self._report_progress,
            )
            self._model_config = self._model.model_config
        else:
            self._report_progress("Building inference model...")
            self._model = GravitonCausalLM.from_pretrained_dir(
                model_dir,
                engine_config=self.config,
                dtype=self.dtype,
            )
            self._report_progress("Moving model to device...")
            self._model.to(self.device)
            self._model.eval()
            self._model_config = self._model.model_config

            if (
                self.quantizer is not None
                and self.config.quantization.mode != QuantMode.NONE
            ):
                self._report_progress(f"Applying {self.config.quantization.mode.value} quantization...")
                self._model.quantize_weights(self.quantizer)

            param_count = sum(p.numel() for p in self._model.parameters())
            buf_bytes = sum(b.numel() * b.element_size() for b in self._model.buffers())
            param_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters())
            total_gb = (param_bytes + buf_bytes) / (1024 ** 3)
            self._report_progress(f"Model ready: {param_count / 1e9:.2f}B params, {total_gb:.2f} GB on {self.device}")

        self._load_tokenizer(model_dir)

    def _load_tokenizer(self, model_dir: Path):
        """Load the tokenizer from the model directory."""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), trust_remote_code=False
            )
            logger.info(f"Tokenizer loaded: vocab_size={len(self._tokenizer)}")
        except Exception as e:
            if self._is_bitnet_model(model_dir) or self._is_moe_model(model_dir):
                logger.warning(f"Tokenizer load failed ({e}), using TinyLlama fallback")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=False
                )
            else:
                raise

    def format_chat_prompt(
        self,
        system: str,
        history: list,
        message: str,
    ) -> Optional[str]:
        """
        Format system + history + message using the tokenizer's chat template.
        Returns None if the tokenizer has no chat_template (caller should fall back).
        """
        if self._tokenizer is None:
            return None
        tokenizer = self._tokenizer
        if not getattr(tokenizer, "chat_template", None):
            return None
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(prompt, str):
                logger.debug("Using tokenizer chat template for prompt")
                return prompt
            if isinstance(prompt, (list, tuple)) and prompt:
                first = prompt[0]
                if isinstance(first, (list, int)):
                    ids = first if isinstance(first, list) else prompt
                    out = tokenizer.decode(ids, skip_special_tokens=False)
                    logger.debug("Using tokenizer chat template (decoded from ids)")
                    return out
            return None
        except Exception as e:
            logger.warning(
                "apply_chat_template failed: %s — fallback format will be used",
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Top-p (nucleus) sampling.
            top_k: Top-k sampling.
            stream: If True, returns a generator yielding text chunks.

        Returns:
            Generated text string, or generator if stream=True.
        """
        max_tokens = max_tokens or self.config.decoding.max_tokens
        temperature = temperature if temperature is not None else self.config.decoding.temperature
        top_p = top_p if top_p is not None else self.config.decoding.top_p
        top_k = top_k if top_k is not None else self.config.decoding.top_k

        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call engine.load_model() first.")

        if stream:
            return self._generate_stream(prompt, max_tokens, temperature, top_p, top_k)
        else:
            return self._generate_batch(prompt, max_tokens, temperature, top_p, top_k)

    def _generate_batch(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        """Generate text in batch mode (returns complete string)."""
        chunks = list(self._generate_stream(prompt, max_tokens, temperature, top_p, top_k))
        return "".join(chunks)

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Generator[str, None, None]:
        """
        Generate text token-by-token with streaming output.

        Automatically switches to speculative decoding when enabled
        in the configuration.
        """
        if self.config.decoding.use_speculative:
            yield from self._generate_stream_speculative(
                prompt, max_tokens, temperature, top_p, top_k,
            )
        else:
            yield from self._generate_stream_standard(
                prompt, max_tokens, temperature, top_p, top_k,
            )

    def _generate_stream_standard(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Generator[str, None, None]:
        """
        Standard autoregressive generation with KV cache.

        Pipeline:
            1. Tokenize the prompt
            2. Prefill: forward pass on full prompt, populate KV cache
            3. Decode loop: generate one token per step using cached KV
            4. Detokenize and yield text chunks
        """
        logger.info("Starting standard token generation...")

        device = self.device
        model = self._model
        tokenizer = self._tokenizer

        self.sampler.temperature = temperature
        self.sampler.top_p = top_p
        self.sampler.top_k = top_k

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        model.init_kv_cache()

        generated_ids: list[int] = []
        prev_text = ""

        # Pre-build the prompt id list once for repetition penalty
        prompt_id_list = input_ids[0].tolist()
        # Re-usable single-token input tensor (avoid allocation per step)
        token_buf = torch.zeros(1, 1, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids, start_pos=0)
            next_logits = logits[:, -1, :].float()

            next_token = self.sampler(next_logits)
            next_token_id = next_token.item()

            if next_token_id == tokenizer.eos_token_id:
                model.clear_kv_cache()
                return

            generated_ids.append(next_token_id)
            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = current_text[len(prev_text):]
            prev_text = current_text
            if new_text:
                yield new_text

            current_pos = prompt_len
            for _step in range(max_tokens - 1):
                token_buf[0, 0] = next_token_id
                logits = model(token_buf, start_pos=current_pos)
                next_logits = logits[:, -1, :].float()

                if self.sampler.repetition_penalty != 1.0:
                    prev_tokens = torch.tensor(
                        [prompt_id_list + generated_ids], device=device
                    )
                    next_token = self.sampler(next_logits, previous_tokens=prev_tokens)
                else:
                    next_token = self.sampler(next_logits)

                next_token_id = next_token.item()

                if next_token_id == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)
                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                new_text = current_text[len(prev_text):]
                prev_text = current_text
                if new_text:
                    yield new_text

                current_pos += 1

        model.clear_kv_cache()

    # ------------------------------------------------------------------
    # Speculative decoding generation
    # ------------------------------------------------------------------

    def _generate_stream_speculative(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Generator[str, None, None]:
        """
        Speculative decoding with layer-skip draft model.

        Algorithm per step:
            1. Draft: run gamma tokens through the model with layer_skip=3
               (3x fewer layers -> ~3x faster draft).
            2. Rollback the KV cache to pre-draft state.
            3. Target: verify all gamma draft tokens in one forward pass
               through the full model.
            4. Accept/reject using standard speculative rejection sampling.
            5. Yield accepted text and continue.
        """
        gamma = self.config.decoding.num_speculative_tokens
        draft_layer_skip = 2
        logger.info(
            f"Starting speculative generation (gamma={gamma}, "
            f"draft_layer_skip={draft_layer_skip})..."
        )

        device = self.device
        model = self._model
        tokenizer = self._tokenizer

        self.sampler.temperature = temperature
        self.sampler.top_p = top_p
        self.sampler.top_k = top_k

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        model.init_kv_cache()

        generated_ids: list[int] = []
        prev_text = ""
        tokens_generated = 0

        # Stats
        accepted_total = 0
        speculated_total = 0

        with torch.no_grad():
            # --- Prefill ---
            logits = model(input_ids, start_pos=0)
            next_logits = logits[:, -1, :].float()

            next_token = self.sampler(next_logits)
            next_token_id = next_token.item()

            if next_token_id == tokenizer.eos_token_id:
                model.clear_kv_cache()
                return

            generated_ids.append(next_token_id)
            tokens_generated += 1
            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = current_text[len(prev_text):]
            prev_text = current_text
            if new_text:
                yield new_text

            current_pos = prompt_len

            # --- Speculative decode loop ---
            while tokens_generated < max_tokens:
                # 1) Save KV cache positions before drafting
                cache_snapshot = model.kv_cache.get_positions()

                # 2) Draft phase: generate gamma tokens with layer_skip
                draft_token_ids = []
                draft_probs = []
                draft_id = next_token_id

                for _ in range(gamma):
                    token_input = torch.tensor([[draft_id]], device=device)
                    draft_logits = model(
                        token_input,
                        start_pos=current_pos + len(draft_token_ids),
                        layer_skip=draft_layer_skip,
                    )
                    draft_logits_last = draft_logits[:, -1, :].float()
                    draft_p = torch.softmax(draft_logits_last, dim=-1)
                    draft_id = torch.argmax(draft_p, dim=-1).item()
                    draft_token_ids.append(draft_id)
                    draft_probs.append(draft_p)

                # 3) Rollback KV cache to pre-draft state
                model.kv_cache.truncate_to(cache_snapshot)

                # 4) Target verification: run all draft tokens in one forward pass
                verify_input = torch.tensor(
                    [[next_token_id] + draft_token_ids], device=device
                )
                target_logits = model(
                    verify_input,
                    start_pos=current_pos,
                    layer_skip=1,
                )

                # 5) Accept/reject via rejection sampling
                #    n_verified = purely accepted tokens (KV cache valid)
                #    corrected_id = resampled token at first rejection point
                n_verified = 0
                corrected_id = None
                for t in range(gamma):
                    target_p = torch.softmax(
                        target_logits[:, t, :].float(), dim=-1
                    )
                    proposed_id = draft_token_ids[t]
                    p_target = target_p[0, proposed_id].item()
                    p_draft = draft_probs[t][0, proposed_id].item()

                    r = torch.rand(1).item()
                    if p_draft > 0 and r < (p_target / max(p_draft, 1e-10)):
                        n_verified += 1
                    else:
                        corrected = self.sampler(target_logits[:, t, :].float())
                        corrected_id = corrected.item()
                        break

                accepted_total += n_verified
                speculated_total += gamma

                # 6) KV cache management
                # The target verification populated positions for all gamma+1 tokens.
                # KV entries for positions current_pos .. current_pos+n_verified are
                # valid (verify_input[0] + n_verified accepted drafts).
                # Truncate everything beyond that.
                keep_len = {
                    layer: pos + 1 + n_verified
                    for layer, pos in cache_snapshot.items()
                }
                model.kv_cache.truncate_to(keep_len)

                # Bonus token: if all gamma tokens accepted
                bonus_id = None
                if n_verified == gamma:
                    bonus_logits = target_logits[:, -1, :].float()
                    bonus_token = self.sampler(bonus_logits)
                    bonus_id = bonus_token.item()

                # Yield verified tokens
                output_ids = draft_token_ids[:n_verified]
                if corrected_id is not None:
                    output_ids.append(corrected_id)

                hit_eos = False
                for tid in output_ids:
                    if tid == tokenizer.eos_token_id:
                        hit_eos = True
                        break
                    generated_ids.append(tid)
                    tokens_generated += 1
                    current_text = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    new_text = current_text[len(prev_text):]
                    prev_text = current_text
                    if new_text:
                        yield new_text

                if hit_eos:
                    break

                # Yield bonus token
                if bonus_id is not None and not hit_eos:
                    if bonus_id == tokenizer.eos_token_id:
                        break
                    generated_ids.append(bonus_id)
                    tokens_generated += 1
                    current_text = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    new_text = current_text[len(prev_text):]
                    prev_text = current_text
                    if new_text:
                        yield new_text

                # Advance position: the target verification added KV entries for
                # next_token_id (1) + n_verified accepted draft tokens.
                # The corrected/bonus token does NOT have a KV entry yet;
                # it becomes next_token_id and gets its KV computed as
                # verify_input[0] in the next speculative step.
                current_pos += 1 + n_verified
                if bonus_id is not None:
                    next_token_id = bonus_id
                elif corrected_id is not None:
                    next_token_id = corrected_id
                elif output_ids:
                    next_token_id = output_ids[-1]

        acceptance_rate = accepted_total / max(speculated_total, 1)
        logger.info(
            f"Speculative decoding done: {tokens_generated} tokens, "
            f"acceptance rate {acceptance_rate:.1%} "
            f"({accepted_total}/{speculated_total})"
        )
        model.clear_kv_cache()

    # ------------------------------------------------------------------
    # Utility methods (quantization, sparsity, benchmarking)
    # ------------------------------------------------------------------

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a single tensor using the configured quantizer."""
        if self.quantizer is None:
            return tensor
        return self.quantizer.quantize(tensor)

    def apply_sparsity(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply sparsity to activations."""
        if self.sparsity_engine is None:
            return tensor
        return self.sparsity_engine(tensor)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._model_loaded or self._model is None:
            return {"loaded": False}

        total_params = sum(p.numel() for p in self._model.parameters())
        total_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters())

        return {
            "loaded": True,
            "num_layers": self._model_config.get("num_hidden_layers", 0),
            "total_parameters": total_params,
            "total_parameters_billions": total_params / 1e9,
            "memory_usage_gb": total_bytes / (1024**3),
            "quantization": self.config.quantization.mode.value,
            "device": str(self.device),
        }

    def benchmark(self, num_tokens: int = 100) -> dict:
        """Run a simple benchmark on quantization and sparsity speed."""
        logger.info(f"Running benchmark with {num_tokens} tokens...")

        test_tensor = torch.randn(4096, 4096)

        start = time.time()
        if self.quantizer:
            quantized = self.quantizer.quantize(test_tensor)
            dequantized = self.quantizer.dequantize(quantized)
        quant_time = time.time() - start

        start = time.time()
        if self.sparsity_engine:
            sparse = self.sparsity_engine(test_tensor)
        sparse_time = time.time() - start

        import psutil

        mem = psutil.virtual_memory()

        return {
            "quantization_time_ms": round(quant_time * 1000, 2),
            "sparsity_time_ms": round(sparse_time * 1000, 2),
            "matrix_size": "4096x4096",
            "quantization_mode": self.config.quantization.mode.value,
            "sparsity_mode": self.config.sparsity.mode.value,
            "memory_used_gb": round((mem.total - mem.available) / (1024**3), 2),
            "memory_available_gb": round(mem.available / (1024**3), 2),
            "device": self.config.device.value,
        }

    def __repr__(self) -> str:
        return (
            f"GravitonEngine("
            f"quant={self.config.quantization.mode.value}, "
            f"sparsity={self.config.sparsity.k_ratio:.0%}, "
            f"memory={self.config.memory.max_memory_gb:.1f}GB, "
            f"device={self.config.device.value})"
        )
