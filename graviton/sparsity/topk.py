"""
Top-K Neuron Activation (Dynamic Sparsity)

Modern LLMs have enormous Feed-Forward Networks (FFNs) that consume
roughly 2/3 of the model's parameters and computation time.
However, for any given token, only a small fraction of these neurons
are actually necessary for the prediction.

TopKActivation forces sparsity by only computing the top K% of
activations and zeroing out the rest. When combined with specialized
sparse matrix operations, this drastically reduces compute requirements.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TopKActivation(nn.Module):
    """
    Top-K sparse activation function.

    Replaces standard activation functions (ReLU, GELU, SiLU) by only
    keeping the largest K values and zeroing out the rest.

    Benefits:
    1. Acts as a natural regularizer
    2. Zeroes out 50-90% of activations
    3. Enables sparse matrix multiplication in the subsequent layer

    Example:
        >>> activation = TopKActivation(k_ratio=0.3)  # Keep top 30%
        >>> x = torch.randn(2, 4096)
        >>> out = activation(x)
        >>> print((out == 0).float().mean())  # ~0.70 (70% sparse)
    """

    def __init__(self, k_ratio: float = 0.5, dim: int = -1):
        """
        Initialize Top-K activation.

        Args:
            k_ratio: Fraction of top values to keep (0.0 to 1.0).
            dim: Dimension along which to calculate top-k. Default is -1
                 (the feature dimension).
        """
        super().__init__()
        assert 0.0 < k_ratio <= 1.0, "k_ratio must be in (0, 1]"

        self.k_ratio = k_ratio
        self.dim = dim

        # Base activation function (applied to the kept values)
        # Using SiLU/Swish as it's standard in modern LLMs (LLaMA, Mistral)
        self.base_activation = nn.SiLU()

        logger.debug(f"TopKActivation initialized: k_ratio={k_ratio}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-K sparsity.

        Args:
            x: Input tensor.

        Returns:
            Sparse tensor with only top K% values non-zero.
        """
        if self.k_ratio == 1.0:
            return self.base_activation(x)

        # Calculate absolute K based on tensor shape and ratio
        k = max(1, int(x.shape[self.dim] * self.k_ratio))

        # We first apply the base activation (e.g., SiLU)
        # SiLU can produce negative values, so we rank based on magnitude (abs)
        # Alternatively, we could just rank by the pre-activation value
        activated = self.base_activation(x)

        # Find top K values
        # Note: topk on GPU is relatively fast, but still has overhead
        values, indices = torch.topk(activated.abs(), k=k, dim=self.dim)

        # Create sparse output
        # Instead of a dense tensor with zeros, we should eventually
        # return a sparse tensor format when sparse matmul is implemented
        sparse_out = torch.zeros_like(x)

        # Scatter the actual values back into their original positions
        sparse_out.scatter_(self.dim, indices, activated.gather(self.dim, indices))

        return sparse_out

    def extra_repr(self) -> str:
        return f"k_ratio={self.k_ratio}, dim={self.dim}"

    @staticmethod
    def sparsify_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        k_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Fused sparse linear operation.

        If the input x is already sparse, we can skip multiplying by columns
        of the weight matrix corresponding to the zeros in x.

        This is a reference implementation. A true optimized implementation
        would use custom CUDA/Triton kernels or specialized CPU instructions.

        Args:
            x: Sparse input vector [..., in_features].
            weight: Dense weight matrix [out_features, in_features].
            bias: Optional bias vector [out_features].
            k_ratio: The known sparsity ratio of x.

        Returns:
            Output tensor.
        """
        # In a real optimized engine, we would:
        # 1. Get non-zero indices of x
        # 2. Gather corresponding columns from weight
        # 3. Perform dense matmul on the much smaller gathered matrices

        # Fallback to dense for this Python reference
        return torch.nn.functional.linear(x, weight, bias)
