"""
Mixture of Experts (MoE) Routing

Mixture of Experts divides the Feed-Forward Network into multiple
smaller sub-networks ("experts"). A routing network decides which
token goes to which expert.

In Graviton, we use MoE principles not just for model architecture,
but for memory management: we can swap experts in and out of active
memory based on routing probabilities, keeping memory usage bounded
even for massive MoE models like Mixtral 8x22B.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MixtureOfExpertsRouter(nn.Module):
    """
    Router for Sparse Mixture of Experts layer.

    Takes token hidden states and predicts which K experts should
    process each token.

    Attributes:
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        """
        Initialize the MoE router.

        Args:
            hidden_dim: Hidden dimension size.
            num_experts: Total number of experts (e.g., 8 for Mixtral).
            top_k: Number of experts to activate per token.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Gate linear layer: maps hidden states to expert logits
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim].

        Returns:
            Tuple containing:
            - routing_weights: Weights for the selected experts [batch, seq, top_k]
            - selected_experts: Indices of selected experts [batch, seq, top_k]
        """
        # Calculate router logits
        logits = self.gate(hidden_states)

        # Get top-k experts and their logits
        top_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over the top-k experts to get routing weights
        routing_weights = F.softmax(top_logits, dim=-1, dtype=torch.float32).to(
            hidden_states.dtype
        )

        return routing_weights, selected_experts

    def compute_expert_load(self, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Compute the load on each expert (for load balancing).

        Args:
            selected_experts: Tensor of expert indices [..., top_k].

        Returns:
            Load per expert (fraction of tokens routed to it).
        """
        # Flatten tokens
        flat_indices = selected_experts.view(-1)
        total_tokens = flat_indices.numel() / self.top_k

        # Count assignments per expert
        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts)

        # Calculate fraction
        return expert_counts.float() / max(total_tokens, 1.0)


def extract_moe_experts(
    model_state_dict: dict, layer_idx: int, num_experts: int = 8
) -> List[dict]:
    """
    Utility: Extract individual experts from a unified MoE state dict.

    Useful for lazy-loading experts individually from disk.

    Args:
        model_state_dict: Huge state dict containing all experts.
        layer_idx: Transformer layer index.
        num_experts: Total number of experts.

    Returns:
        List of state dictionaries, one per expert.
    """
    experts = [{} for _ in range(num_experts)]
    prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts."

    for i in range(num_experts):
        expert_prefix = f"{prefix}{i}."

        # Find keys that belong to this expert
        for key, tensor in model_state_dict.items():
            if key.startswith(expert_prefix):
                # Strip prefix for the individual expert dict
                short_key = key[len(expert_prefix) :]
                experts[i][short_key] = tensor

    return experts
