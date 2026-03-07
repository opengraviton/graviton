"""
Mixed-Precision Quantization

Applies different quantization bit-widths to different layers based
on their sensitivity to quantization error. Critical layers (attention,
embeddings, first/last layers) use higher precision, while less
sensitive layers (FFN middle layers) use aggressive compression.

This approach preserves model quality while maximizing compression
on non-critical pathways.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List

import torch

from graviton.quantization.base import BaseQuantizer, QuantizedTensor
from graviton.quantization.linear import LinearQuantizer
from graviton.quantization.ternary import TernaryQuantizer

logger = logging.getLogger(__name__)


# Default sensitivity patterns for transformer models
CRITICAL_PATTERNS = [
    "embed",          # Embedding layers — high impact
    "lm_head",        # Output head — high impact
    "norm",           # Layer norms — must stay high precision
    "wte", "wpe",     # Position/token embeddings
    ".0.",            # First transformer layer
    "q_proj",         # Query projection (attention)
    "k_proj",         # Key projection (attention)
]

NON_CRITICAL_PATTERNS = [
    "gate_proj",      # FFN gate
    "up_proj",        # FFN up projection
    "down_proj",      # FFN down projection
    "mlp",            # MLP layers generally
    "fc1", "fc2",     # Feedforward layers
    "dense",          # Dense/linear layers
]


class MixedPrecisionQuantizer(BaseQuantizer):
    """
    Layer-aware mixed-precision quantizer.

    Automatically determines which layers are critical (need higher
    precision) and which can be aggressively compressed. Uses a
    combination of:
        1. Pattern matching on layer names
        2. Sensitivity analysis (optional calibration)
        3. Configurable per-layer overrides

    Example:
        >>> quantizer = MixedPrecisionQuantizer(
        ...     config,
        ...     critical_bits=8,
        ...     non_critical_bits=2,
        ... )
        >>> # Automatically chooses precision per layer:
        >>> qtensor = quantizer.quantize(weight, layer_name="model.layers.5.mlp.gate_proj")
    """

    def __init__(
        self,
        config=None,
        critical_bits: int = 8,
        non_critical_bits: int = 4,
        default_bits: int = 4,
        group_size: int = 128,
    ):
        """
        Initialize the mixed-precision quantizer.

        Args:
            config: QuantizationConfig (optional).
            critical_bits: Bit width for critical layers.
            non_critical_bits: Bit width for non-critical layers.
            default_bits: Bit width for layers not matching any pattern.
            group_size: Group size for linear quantization.
        """
        if config is not None:
            critical_bits = int(config.critical_layer_bits)
            non_critical_bits = int(config.non_critical_layer_bits)

        self._critical_bits = critical_bits
        self._non_critical_bits = non_critical_bits
        self._default_bits = default_bits
        self._group_size = group_size

        # Create quantizers for each bit width
        self._quantizers: Dict[int, BaseQuantizer] = {}
        for b in set([critical_bits, non_critical_bits, default_bits]):
            if b <= 2:
                # Use ternary for ≤2 bits
                self._quantizers[b] = TernaryQuantizer()
            else:
                self._quantizers[b] = LinearQuantizer(
                    bits=min(b, 8), group_size=group_size
                )

        # Layer sensitivity scores (higher = more sensitive)
        self._sensitivity_scores: Dict[str, float] = {}

        # Custom per-layer overrides
        self._layer_overrides: Dict[str, int] = {}

        logger.info(
            f"MixedPrecisionQuantizer: critical={critical_bits}b, "
            f"non-critical={non_critical_bits}b, default={default_bits}b"
        )

    @property
    def name(self) -> str:
        return f"mixed_precision_{self._critical_bits}_{self._non_critical_bits}"

    @property
    def bits(self) -> float:
        # Return average bits (approximate)
        return float(self._default_bits)

    def set_layer_bits(self, layer_name: str, bits: int):
        """
        Override the bit width for a specific layer.

        Args:
            layer_name: Full layer name.
            bits: Target bit width.
        """
        self._layer_overrides[layer_name] = bits
        logger.debug(f"Layer override: {layer_name} → {bits}-bit")

    def get_layer_bits(self, layer_name: str) -> int:
        """
        Determine the appropriate bit width for a layer.

        Priority:
            1. Explicit overrides
            2. Sensitivity scores (from calibration)
            3. Pattern matching on layer name
            4. Default bit width

        Args:
            layer_name: Full layer name.

        Returns:
            Target bit width for this layer.
        """
        # Check explicit overrides first
        if layer_name in self._layer_overrides:
            return self._layer_overrides[layer_name]

        # Check sensitivity scores
        if layer_name in self._sensitivity_scores:
            sensitivity = self._sensitivity_scores[layer_name]
            if sensitivity > 0.5:
                return self._critical_bits
            elif sensitivity < 0.1:
                return self._non_critical_bits

        # Pattern matching
        layer_lower = layer_name.lower()

        for pattern in CRITICAL_PATTERNS:
            if pattern in layer_lower:
                return self._critical_bits

        for pattern in NON_CRITICAL_PATTERNS:
            if pattern in layer_lower:
                return self._non_critical_bits

        return self._default_bits

    def quantize(
        self,
        tensor: torch.Tensor,
        layer_name: str = "",
    ) -> QuantizedTensor:
        """
        Quantize a tensor with layer-appropriate precision.

        Args:
            tensor: Input tensor.
            layer_name: Layer name for precision selection.

        Returns:
            QuantizedTensor with appropriate precision.
        """
        bits = self.get_layer_bits(layer_name)

        # Skip quantization for norms and very small tensors
        if "norm" in layer_name.lower() or tensor.numel() < 64:
            # Return as-is in a QuantizedTensor wrapper
            return QuantizedTensor(
                data=tensor.to(torch.float16),
                scale=torch.ones(1, dtype=torch.float16),
                bits=16.0,
                shape=tensor.shape,
                group_size=tensor.numel(),
                dtype=tensor.dtype,
            )

        quantizer = self._get_quantizer(bits)
        qtensor = quantizer.quantize(tensor)

        logger.debug(
            f"Mixed precision: {layer_name} → {bits}-bit "
            f"(compression: {qtensor.compression_ratio:.1f}x)"
        )

        return qtensor

    def dequantize(self, qtensor: QuantizedTensor) -> torch.Tensor:
        """
        Dequantize a tensor.

        Args:
            qtensor: Quantized tensor.

        Returns:
            Dequantized tensor.
        """
        if qtensor.bits >= 16:
            return qtensor.data.to(qtensor.dtype)

        bits = int(qtensor.bits) if qtensor.bits == int(qtensor.bits) else 2
        quantizer = self._get_quantizer(bits)
        return quantizer.dequantize(qtensor)

    def _get_quantizer(self, bits: int) -> BaseQuantizer:
        """Get or create a quantizer for the given bit width."""
        if bits not in self._quantizers:
            if bits <= 2:
                self._quantizers[bits] = TernaryQuantizer()
            else:
                self._quantizers[bits] = LinearQuantizer(
                    bits=min(bits, 8), group_size=self._group_size
                )
        return self._quantizers[bits]

    def analyze_sensitivity(
        self,
        model_weights: Dict[str, torch.Tensor],
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Analyze layer sensitivity to quantization.

        Computes a sensitivity score for each layer based on:
        1. Weight distribution (kurtosis, outliers)
        2. Quantization error (MSE after quantize-dequantize roundtrip)

        Args:
            model_weights: Dictionary of layer_name → weight_tensor.
            calibration_data: Optional calibration inputs.

        Returns:
            Dictionary of layer_name → sensitivity_score (0-1).
        """
        logger.info(f"Analyzing sensitivity of {len(model_weights)} layers...")

        for name, weight in model_weights.items():
            if weight.ndim < 2:
                continue

            # Compute weight statistics
            abs_values = weight.float().abs()
            mean_abs = abs_values.mean().item()
            max_abs = abs_values.max().item()

            # Outlier ratio: fraction of values > 3*mean
            outlier_ratio = (abs_values > 3 * mean_abs).float().mean().item()

            # Kurtosis (higher = more outliers = more sensitive)
            std = weight.float().std().item()
            if std > 0:
                kurtosis = ((weight.float() - weight.float().mean()) ** 4).mean().item() / (
                    std**4
                )
            else:
                kurtosis = 0.0

            # Quantization error at lowest bit width
            low_quantizer = self._get_quantizer(self._non_critical_bits)
            error = low_quantizer.compute_error(weight)

            # Combine into sensitivity score
            sensitivity = min(1.0, (
                outlier_ratio * 5         # Outliers heavily penalize
                + (kurtosis / 10) * 0.3   # High kurtosis is sensitive
                + error["mse"] * 100      # High MSE is sensitive
            ))

            self._sensitivity_scores[name] = sensitivity

            logger.debug(
                f"  {name}: sensitivity={sensitivity:.3f} "
                f"(outliers={outlier_ratio:.3f}, kurtosis={kurtosis:.1f})"
            )

        return self._sensitivity_scores

    def get_compression_report(
        self, model_weights: Dict[str, torch.Tensor]
    ) -> dict:
        """
        Generate a compression report for a model.

        Args:
            model_weights: Dictionary of layer_name → weight_tensor.

        Returns:
            Report with per-layer and total compression statistics.
        """
        total_original = 0
        total_compressed = 0
        layer_reports = []

        for name, weight in model_weights.items():
            bits = self.get_layer_bits(name)
            original_bytes = weight.numel() * weight.element_size()
            compressed_bytes = (weight.numel() * bits) / 8

            total_original += original_bytes
            total_compressed += compressed_bytes

            layer_reports.append({
                "name": name,
                "bits": bits,
                "original_mb": original_bytes / (1024**2),
                "compressed_mb": compressed_bytes / (1024**2),
                "ratio": original_bytes / max(compressed_bytes, 1),
            })

        return {
            "total_original_gb": total_original / (1024**3),
            "total_compressed_gb": total_compressed / (1024**3),
            "overall_ratio": total_original / max(total_compressed, 1),
            "layers": layer_reports,
        }
