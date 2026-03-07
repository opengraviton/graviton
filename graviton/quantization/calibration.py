"""
Calibration Data Collection

Collects calibration data for quantization-aware compression.
Calibration helps determine optimal scale factors and thresholds
by running representative inputs through the model.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Callable

import torch

logger = logging.getLogger(__name__)


class CalibrationCollector:
    """
    Collects activation statistics for calibration.

    Hooks into model layers to capture min/max/mean statistics
    that inform quantization parameters.

    Example:
        >>> collector = CalibrationCollector()
        >>> collector.attach(model)
        >>> for batch in calibration_data:
        ...     model(batch)
        >>> stats = collector.get_statistics()
        >>> collector.detach()
    """

    def __init__(self, num_samples: int = 128):
        """
        Initialize the calibration collector.

        Args:
            num_samples: Maximum number of samples to collect.
        """
        self._num_samples = num_samples
        self._hooks = []
        self._statistics: dict = {}
        self._sample_count = 0

    def attach(self, model: torch.nn.Module):
        """
        Attach calibration hooks to all linear layers.

        Args:
            model: PyTorch model to calibrate.
        """
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

        logger.info(f"Attached {len(self._hooks)} calibration hooks")

    def detach(self):
        """Remove all calibration hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("Detached calibration hooks")

    def _make_hook(self, layer_name: str):
        """Create a forward hook for a layer."""

        def hook_fn(module, input, output):
            if self._sample_count >= self._num_samples:
                return

            with torch.no_grad():
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input

                if layer_name not in self._statistics:
                    self._statistics[layer_name] = {
                        "min": float("inf"),
                        "max": float("-inf"),
                        "abs_mean_sum": 0.0,
                        "count": 0,
                    }

                stats = self._statistics[layer_name]
                stats["min"] = min(stats["min"], x.min().item())
                stats["max"] = max(stats["max"], x.max().item())
                stats["abs_mean_sum"] += x.abs().mean().item()
                stats["count"] += 1

            self._sample_count += 1

        return hook_fn

    def get_statistics(self) -> dict:
        """
        Get collected statistics.

        Returns:
            Dictionary of layer_name → statistics.
        """
        result = {}
        for name, stats in self._statistics.items():
            count = max(stats["count"], 1)
            result[name] = {
                "min": stats["min"],
                "max": stats["max"],
                "abs_mean": stats["abs_mean_sum"] / count,
                "range": stats["max"] - stats["min"],
                "samples": count,
            }
        return result

    def reset(self):
        """Reset collected statistics."""
        self._statistics.clear()
        self._sample_count = 0


def generate_calibration_data(
    tokenizer=None,
    texts: Optional[List[str]] = None,
    num_samples: int = 128,
    seq_length: int = 512,
) -> List[torch.Tensor]:
    """
    Generate calibration data for quantization.

    Uses either provided texts or default calibration prompts.

    Args:
        tokenizer: Tokenizer instance.
        texts: Optional list of calibration texts.
        num_samples: Number of calibration samples.
        seq_length: Sequence length per sample.

    Returns:
        List of input tensors for calibration.
    """
    if texts is None:
        texts = _default_calibration_texts()

    if tokenizer is None:
        # Without tokenizer, generate random data
        logger.warning("No tokenizer provided, using random calibration data")
        return [
            torch.randint(0, 32000, (1, seq_length))
            for _ in range(num_samples)
        ]

    calibration_inputs = []
    for text in texts[:num_samples]:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_length,
            truncation=True,
            padding="max_length",
        )
        calibration_inputs.append(tokens["input_ids"])

    return calibration_inputs


def _default_calibration_texts() -> List[str]:
    """Return default calibration texts covering diverse domains."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing. Then there was light.",
        "Machine learning is a subset of artificial intelligence.",
        "The capital of France is Paris, known for the Eiffel Tower.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Water is composed of two hydrogen atoms and one oxygen atom.",
        "The stock market experienced significant volatility today.",
        "Quantum computing leverages quantum mechanical phenomena.",
        "The recipe calls for two cups of flour and one egg.",
        "Climate change poses significant challenges to global ecosystems.",
        "Neural networks are inspired by the structure of the brain.",
        "The theory of relativity was proposed by Albert Einstein.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The Industrial Revolution began in the late 18th century.",
        "Blockchain technology provides decentralized record keeping.",
        "Music theory explains the structure and elements of music.",
    ]
