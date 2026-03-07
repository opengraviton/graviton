"""
Graviton Configuration System

Provides a comprehensive configuration dataclass for controlling
all aspects of the Graviton inference engine, from quantization
parameters to memory management and decoding strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuantMode(Enum):
    """Quantization mode enumeration."""

    NONE = "none"  # No quantization (FP16/FP32)
    INT8 = "int8"  # 8-bit integer quantization
    INT4 = "int4"  # 4-bit integer quantization
    INT2 = "int2"  # 2-bit integer quantization
    TERNARY = "ternary"  # 1.58-bit ternary {-1, 0, +1}

    @property
    def bits(self) -> float:
        """Return the effective bits per parameter."""
        return {
            QuantMode.NONE: 16.0,
            QuantMode.INT8: 8.0,
            QuantMode.INT4: 4.0,
            QuantMode.INT2: 2.0,
            QuantMode.TERNARY: math.log2(3),  # ~1.585 bits
        }[self]


class DeviceType(Enum):
    """Target device for computation."""

    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"  # Apple Metal Performance Shaders
    CUDA = "cuda"


class SparsityMode(Enum):
    """Sparsity strategy enumeration."""

    NONE = "none"
    TOPK = "topk"  # Top-K neuron activation
    MAGNITUDE = "magnitude"  # Magnitude-based pruning
    MOE = "moe"  # Mixture of Experts routing


@dataclass
class QuantizationConfig:
    """Configuration for the quantization engine."""

    mode: QuantMode = QuantMode.INT4
    group_size: int = 128  # Number of elements per quantization group
    symmetric: bool = True  # Symmetric vs asymmetric quantization
    calibration_samples: int = 128  # Number of calibration samples
    sensitivity_threshold: float = 0.01  # Layer sensitivity threshold for mixed precision

    # Mixed precision settings
    use_mixed_precision: bool = True
    critical_layer_bits: float = 8.0  # Higher precision for critical layers
    non_critical_layer_bits: float = 4.0  # Lower precision for non-critical layers

    @property
    def effective_bits(self) -> float:
        """Return effective bits per parameter."""
        return self.mode.bits

    def estimated_compression_ratio(self) -> float:
        """Return the estimated compression ratio vs FP16."""
        return 16.0 / self.effective_bits


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    max_memory_gb: float = 0.0  # 0 = auto-detect
    use_mmap: bool = True  # Use memory-mapped file loading
    use_layer_streaming: bool = True  # Stream layers from disk
    prefetch_layers: int = 2  # Number of layers to prefetch
    layer_cache_size: int = 8  # Max layers to keep in memory (LRU)

    # KV-cache settings
    kv_cache_bits: int = 8  # Quantization bits for KV cache
    max_context_length: int = 4096  # Maximum context length
    sliding_window_size: int = 2048  # Sliding window for bounded memory

    def get_memory_budget_bytes(self) -> int:
        """Return memory budget in bytes."""
        if self.max_memory_gb > 0:
            return int(self.max_memory_gb * (1024**3))
        return 0  # Will be set by hardware detection


@dataclass
class SparsityConfig:
    """Configuration for sparsity and dynamic computation reduction."""

    mode: SparsityMode = SparsityMode.TOPK
    k_ratio: float = 0.5  # Fraction of neurons to keep active
    pruning_threshold: float = 0.01  # Magnitude threshold for pruning

    # MoE settings
    num_experts: int = 8  # Total number of experts
    active_experts: int = 2  # Number of experts to activate per token
    load_balance_factor: float = 0.01  # Load balancing loss coefficient


@dataclass
class DecodingConfig:
    """Configuration for text generation and decoding."""

    use_speculative: bool = False  # Enable speculative decoding
    num_speculative_tokens: int = 5  # Tokens to speculate per step

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_tokens: int = 512

    # Batch settings
    batch_size: int = 1
    stream: bool = True  # Enable token streaming


@dataclass
class GravitonConfig:
    """
    Master configuration for the Graviton inference engine.

    Combines all sub-configurations into a single, unified config.
    Provides convenient constructor arguments for common settings.

    Example:
        >>> config = GravitonConfig(
        ...     quant_bits=4,
        ...     sparsity_ratio=0.5,
        ...     max_memory_gb=16,
        ... )
    """

    # Convenience parameters (override sub-configs)
    quant_bits: Optional[float] = None
    sparsity_ratio: Optional[float] = None
    max_memory_gb: Optional[float] = None
    use_mmap: Optional[bool] = None
    use_speculative: Optional[bool] = None
    device: DeviceType = DeviceType.AUTO

    # Sub-configurations
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    # Model settings
    model_path: Optional[str] = None
    dtype: str = "float16"  # Base dtype before quantization

    # Logging
    verbose: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Apply convenience parameters to sub-configs."""
        if self.quant_bits is not None:
            self._set_quant_mode(self.quant_bits)

        if self.sparsity_ratio is not None:
            self.sparsity.k_ratio = self.sparsity_ratio

        if self.max_memory_gb is not None:
            self.memory.max_memory_gb = self.max_memory_gb

        if self.use_mmap is not None:
            self.memory.use_mmap = self.use_mmap

        if self.use_speculative is not None:
            self.decoding.use_speculative = self.use_speculative

    def _set_quant_mode(self, bits: float):
        """Set quantization mode from bits value."""
        if bits >= 16:
            self.quantization.mode = QuantMode.NONE
        elif bits >= 8:
            self.quantization.mode = QuantMode.INT8
        elif bits >= 4:
            self.quantization.mode = QuantMode.INT4
        elif bits >= 2:
            self.quantization.mode = QuantMode.INT2
        else:
            self.quantization.mode = QuantMode.TERNARY

    def estimate_memory_usage(self, num_params: int) -> dict:
        """
        Estimate memory usage for a model with the given number of parameters.

        Args:
            num_params: Number of model parameters.

        Returns:
            Dictionary with memory estimates in GB.
        """
        bits = self.quantization.effective_bits
        weight_memory_gb = (num_params * bits / 8) / (1024**3)

        # KV cache estimate (approximate)
        kv_memory_gb = (
            self.memory.max_context_length
            * 2  # keys + values
            * 128  # typical hidden dim per head
            * 32  # typical num heads
            * (self.memory.kv_cache_bits / 8)
        ) / (1024**3)

        # Activation memory (approximate)
        activation_gb = weight_memory_gb * 0.1  # ~10% of weights

        total_gb = weight_memory_gb + kv_memory_gb + activation_gb

        return {
            "weights_gb": round(weight_memory_gb, 2),
            "kv_cache_gb": round(kv_memory_gb, 2),
            "activations_gb": round(activation_gb, 2),
            "total_gb": round(total_gb, 2),
            "compression_ratio": round(16.0 / bits, 1),
            "original_fp16_gb": round((num_params * 2) / (1024**3), 2),
        }

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║       Graviton Configuration         ║",
            "╠══════════════════════════════════════╣",
            f"║ Quantization: {self.quantization.mode.value:>20} ║",
            f"║   Effective bits: {self.quantization.effective_bits:>17.2f} ║",
            f"║   Compression: {self.quantization.estimated_compression_ratio():>18.1f}x ║",
            f"║ Sparsity: {self.sparsity.mode.value:>24} ║",
            f"║   K ratio: {self.sparsity.k_ratio:>23.1%} ║",
            f"║ Memory budget: {self.memory.max_memory_gb:>17.1f} GB ║",
            f"║   mmap: {str(self.memory.use_mmap):>26} ║",
            f"║   Layer streaming: {str(self.memory.use_layer_streaming):>15} ║",
            f"║ Speculative decoding: {str(self.decoding.use_speculative):>12} ║",
            f"║ Device: {self.device.value:>26} ║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    @classmethod
    def for_mac_mini(cls, memory_gb: float = 16.0) -> "GravitonConfig":
        """Create an optimized config for Mac Mini."""
        return cls(
            quant_bits=4,
            sparsity_ratio=0.5,
            max_memory_gb=memory_gb * 0.8,  # Leave 20% for OS
            use_mmap=True,
            use_speculative=True,
            device=DeviceType.MPS,
        )

    @classmethod
    def for_extreme_compression(cls, memory_gb: float = 8.0) -> "GravitonConfig":
        """Create a config for extreme compression (max model size)."""
        config = cls(
            quant_bits=1.58,
            sparsity_ratio=0.3,
            max_memory_gb=memory_gb * 0.8,
            use_mmap=True,
            use_speculative=True,
        )
        config.memory.use_layer_streaming = True
        config.memory.layer_cache_size = 4
        return config

    @classmethod
    def for_quality(cls, memory_gb: float = 64.0) -> "GravitonConfig":
        """Create a config prioritizing output quality."""
        return cls(
            quant_bits=8,
            sparsity_ratio=0.8,
            max_memory_gb=memory_gb * 0.8,
            use_mmap=True,
            use_speculative=False,
        )
