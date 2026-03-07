"""
Dynamic Inference-Time Pruning

Unlike traditional static pruning (which removes weights permanently
after training), dynamic pruning decides which weights to ignore
on-the-fly during inference.

This module provides mechanisms to prune:
1. Individual low-magnitude weights (unstructured)
2. Entire attention heads (structured)
3. FFN rows/columns (structured)
"""

from __future__ import annotations

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DynamicPruner:
    """
    Applies dynamic pruning thresholds during inference.

    Uses a configurable magnitude threshold to zero out weights
    before matrix multiplication. When combined with quantization,
    this creates highly compressible sparse representations.
    """

    def __init__(self, threshold: float = 0.01):
        """
        Initialize the pruner.

        Args:
            threshold: Magnitude threshold below which weights are pruned.
        """
        self.threshold = threshold
        self._pruning_stats: Dict[str, float] = {}

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Alias for prune()."""
        return self.prune(tensor)

    def prune(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """
        Prune low-magnitude weights.

        Args:
            tensor: Input weight tensor.
            name: Optional layer name for statistics tracking.

        Returns:
            Pruned tensor (zeros inserted below threshold).
        """
        if self.threshold <= 0:
            return tensor

        mask = tensor.abs() > self.threshold
        pruned = tensor * mask

        if name:
            sparsity = 1.0 - (mask.sum().item() / tensor.numel())
            self._pruning_stats[name] = sparsity

        return pruned

    def structured_prune_attention(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        o_proj: torch.Tensor,
        num_heads: int,
        keep_ratio: float = 0.5,
    ) -> tuple:
        """
        Dynamically prune entire attention heads based on importance.

        Importance is estimated by the L2 norm of the projection matrices
        for each head. Less important heads are dropped entirely.

        Args:
            q_proj: Query projection weights [embed_dim, embed_dim]
            k_proj: Key projection weights
            v_proj: Value projection weights
            o_proj: Output projection weights
            num_heads: Number of attention heads
            keep_ratio: Fraction of heads to keep

        Returns:
            Tuple of pruned (q, k, v, o) projection matrices.
        """
        embed_dim = q_proj.shape[0]
        head_dim = embed_dim // num_heads

        # Reshape to [num_heads, head_dim, embed_dim]
        # (Assuming weight matrices are [out_features, in_features])
        q_heads = q_proj.view(num_heads, head_dim, embed_dim)
        v_heads = v_proj.view(num_heads, head_dim, embed_dim)
        o_heads = o_proj.t().view(num_heads, head_dim, embed_dim)

        # Estimate head importance by L2 norm of Value and Output projections
        # V and O are more indicative of the information a head contributes
        v_norms = v_heads.norm(dim=(1, 2))
        o_norms = o_heads.norm(dim=(1, 2))
        importance = v_norms * o_norms

        # Find top K heads
        num_keep = max(1, int(num_heads * keep_ratio))
        _, keep_indices = torch.topk(importance, k=num_keep)

        # Create mask
        mask = torch.zeros(num_heads, device=q_proj.device, dtype=torch.bool)
        mask[keep_indices] = True

        # Expand mask
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)

        # Apply mask
        q_pruned = (q_heads * expanded_mask).view(embed_dim, embed_dim)
        k_pruned = k_proj.view(num_heads, -1, embed_dim) * expanded_mask.view(num_heads, 1, 1)
        k_pruned = k_pruned.view(-1, embed_dim)
        v_pruned = (v_heads * expanded_mask).view(embed_dim, embed_dim)
        o_pruned = (o_heads * expanded_mask).view(num_heads * head_dim, embed_dim).t()

        return q_pruned, k_pruned, v_pruned, o_pruned

    def get_statistics(self) -> dict:
        """Get pruning statistics per layer."""
        if not self._pruning_stats:
            return {"overall_sparsity": 0.0}

        overall = sum(self._pruning_stats.values()) / len(self._pruning_stats)
        return {
            "overall_sparsity": overall,
            "layers": self._pruning_stats,
        }
