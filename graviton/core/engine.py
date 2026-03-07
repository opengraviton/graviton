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
        >>> config = GravitonConfig(quant_bits=4, max_memory_gb=16)
        >>> engine = GravitonEngine(model_path="path/to/model", config=config)
        >>> output = engine.generate("Hello, world!", max_tokens=100)
        >>> print(output)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[GravitonConfig] = None,
    ):
        """
        Initialize the Graviton Engine.

        Args:
            model_path: Path to the model (local path or HuggingFace model ID).
            config: Engine configuration. If None, auto-detects optimal config.
        """
        self.config = config or GravitonConfig()
        self.model_path = model_path or self.config.model_path

        # Detect hardware and adjust config if needed
        self.hardware = detect_hardware()
        self._auto_configure()

        # Initialize components
        self.quantizer = self._init_quantizer()
        self.memory_manager = MemoryManager(self.config.memory, self.hardware)
        self.sparsity_engine = self._init_sparsity()
        self.sampler = Sampler(self.config.decoding)

        # Model state
        self._model_loaded = False
        self._model_weights = {}
        self._layer_count = 0

        logger.info("Graviton Engine initialized")
        if self.config.verbose:
            print(self.hardware.summary())
            print()
            print(self.config.summary())

    def _auto_configure(self):
        """Auto-configure based on detected hardware."""
        # Set device if auto
        if self.config.device == DeviceType.AUTO:
            if self.hardware.has_mps:
                self.config.device = DeviceType.MPS
            elif self.hardware.has_cuda:
                self.config.device = DeviceType.CUDA
            else:
                self.config.device = DeviceType.CPU

        # Auto-set memory budget
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
        else:
            return None  # No quantization

    def _init_sparsity(self):
        """Initialize the sparsity engine."""
        from graviton.core.config import SparsityMode

        mode = self.config.sparsity.mode

        if mode == SparsityMode.TOPK:
            return TopKActivation(k_ratio=self.config.sparsity.k_ratio)
        elif mode == SparsityMode.MAGNITUDE:
            return DynamicPruner(threshold=self.config.sparsity.pruning_threshold)
        else:
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

    def load_model(self, model_path: Optional[str] = None):
        """
        Load and optimize a model for inference.

        Args:
            model_path: Path to model. Uses constructor path if not specified.
        """
        path = model_path or self.model_path
        if path is None:
            raise ValueError("No model path specified")

        logger.info(f"Loading model from: {path}")
        start_time = time.time()

        # Check if path exists locally
        model_path_obj = Path(path)

        if model_path_obj.exists():
            self._load_local_model(model_path_obj)
        else:
            # Attempt to load from HuggingFace
            self._load_huggingface_model(path)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")
        self._model_loaded = True

    def _load_local_model(self, path: Path):
        """Load a model from local disk."""
        # Check for different formats
        if path.suffix == ".gguf":
            self._load_gguf(path)
        elif path.suffix == ".safetensors" or (path / "model.safetensors").exists():
            self._load_safetensors(path)
        elif path.suffix in (".bin", ".pt", ".pth"):
            self._load_pytorch(path)
        elif path.is_dir():
            # Directory — look for model files
            safetensors = list(path.glob("*.safetensors"))
            bins = list(path.glob("*.bin"))
            if safetensors:
                self._load_safetensors(path)
            elif bins:
                self._load_pytorch(path)
            else:
                raise ValueError(f"No recognized model files in {path}")
        else:
            raise ValueError(f"Unrecognized model format: {path}")

    def _load_safetensors(self, path: Path):
        """Load model from SafeTensors format."""
        try:
            from safetensors import safe_open

            if path.is_dir():
                files = sorted(path.glob("*.safetensors"))
            else:
                files = [path]

            for f in files:
                with safe_open(str(f), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        tensor = sf.get_tensor(key)
                        # Apply quantization
                        if self.quantizer is not None and self._should_quantize(key):
                            tensor = self.quantizer.quantize(tensor)
                        self._model_weights[key] = tensor

        except ImportError:
            raise ImportError(
                "safetensors package required. Install with: pip install safetensors"
            )

    def _load_pytorch(self, path: Path):
        """Load model from PyTorch format."""
        if path.is_dir():
            files = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        else:
            files = [path]

        for f in files:
            state_dict = torch.load(str(f), map_location="cpu", weights_only=True)
            for key, tensor in state_dict.items():
                if self.quantizer is not None and self._should_quantize(key):
                    tensor = self.quantizer.quantize(tensor)
                self._model_weights[key] = tensor

    def _load_gguf(self, path: Path):
        """Load model from GGUF format (llama.cpp compatible)."""
        logger.warning("GGUF loading is experimental")
        # GGUF loader will be implemented in a future version
        raise NotImplementedError("GGUF loading coming soon")

    def _load_huggingface_model(self, model_id: str):
        """Load a model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading model from HuggingFace: {model_id}")
            local_dir = snapshot_download(model_id)
            self._load_local_model(Path(local_dir))

        except ImportError:
            raise ImportError(
                "huggingface_hub required. Install with: "
                "pip install graviton-ai[huggingface]"
            )

    def _should_quantize(self, layer_name: str) -> bool:
        """Determine if a layer should be quantized."""
        # Skip embeddings and layer norms
        skip_patterns = ["embed", "norm", "lm_head", "wte", "wpe"]
        return not any(pattern in layer_name.lower() for pattern in skip_patterns)

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a single tensor using the configured quantizer.

        Args:
            tensor: Input tensor to quantize.

        Returns:
            Quantized tensor.
        """
        if self.quantizer is None:
            return tensor
        return self.quantizer.quantize(tensor)

    def apply_sparsity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply sparsity to activations.

        Args:
            tensor: Activation tensor.

        Returns:
            Sparse activation tensor.
        """
        if self.sparsity_engine is None:
            return tensor
        return self.sparsity_engine(tensor)

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
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling.
            top_k: Top-k sampling.
            stream: If True, returns a generator yielding tokens.

        Returns:
            Generated text string, or generator if stream=True.
        """
        max_tokens = max_tokens or self.config.decoding.max_tokens
        temperature = temperature or self.config.decoding.temperature
        top_p = top_p or self.config.decoding.top_p
        top_k = top_k or self.config.decoding.top_k

        logger.info(f"Generating with max_tokens={max_tokens}, temp={temperature}")

        if not self._model_loaded:
            raise RuntimeError(
                "No model loaded. Call engine.load_model() first."
            )

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
        """Generate text in batch mode."""
        tokens = []
        for token in self._generate_stream(
            prompt, max_tokens, temperature, top_p, top_k
        ):
            tokens.append(token)
        return "".join(tokens)

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Generator[str, None, None]:
        """Generate text token by token (streaming)."""
        # This is a simplified generation loop
        # Full implementation requires tokenizer and model forward pass
        logger.info("Starting token generation...")

        # Placeholder — actual implementation depends on model architecture
        # This demonstrates the pipeline flow
        for i in range(max_tokens):
            # In full implementation:
            # 1. Tokenize input
            # 2. Forward pass through transformer layers
            #    - Apply sparsity to activations
            #    - Use quantized weights
            #    - Stream layers if needed
            # 3. Sample next token
            # 4. Detokenize and yield
            yield ""

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"loaded": False}

        total_params = sum(
            t.numel() if isinstance(t, torch.Tensor) else 0
            for t in self._model_weights.values()
        )

        total_bytes = sum(
            t.element_size() * t.numel() if isinstance(t, torch.Tensor) else 0
            for t in self._model_weights.values()
        )

        return {
            "loaded": True,
            "num_layers": self._layer_count,
            "total_parameters": total_params,
            "total_parameters_billions": total_params / 1e9,
            "memory_usage_gb": total_bytes / (1024**3),
            "num_tensors": len(self._model_weights),
            "quantization": self.config.quantization.mode.value,
        }

    def benchmark(self, num_tokens: int = 100) -> dict:
        """
        Run a simple benchmark.

        Args:
            num_tokens: Number of tokens to generate for benchmarking.

        Returns:
            Dictionary with benchmark results.
        """
        logger.info(f"Running benchmark with {num_tokens} tokens...")

        # Benchmark quantization speed
        test_tensor = torch.randn(4096, 4096)

        # Quantization benchmark
        start = time.time()
        if self.quantizer:
            quantized = self.quantizer.quantize(test_tensor)
            dequantized = self.quantizer.dequantize(quantized)
        quant_time = time.time() - start

        # Sparsity benchmark
        start = time.time()
        if self.sparsity_engine:
            sparse = self.sparsity_engine(test_tensor)
        sparse_time = time.time() - start

        # Memory info
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
        """String representation."""
        return (
            f"GravitonEngine("
            f"quant={self.config.quantization.mode.value}, "
            f"sparsity={self.config.sparsity.k_ratio:.0%}, "
            f"memory={self.config.memory.max_memory_gb:.1f}GB, "
            f"device={self.config.device.value})"
        )
