"""
Token Sampling Strategies

Provides various strategies for autoregressive token generation:
- Temperature scaling
- Top-K sampling
- Top-P (Nucleus) sampling
- Repetition penalties
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Sampler:
    """
    Token sampler for text generation.

    Implements standard decoding heuristics to control the randomness
    and quality of generated text.

    Example:
        >>> sampler = Sampler(config)
        >>> next_token = sampler(logits)
    """

    def __init__(self, config=None):
        """
        Initialize the sampler.

        Args:
            config: DecodingConfig instance.
        """
        self.temperature = config.temperature if config else 1.0
        self.top_p = config.top_p if config else 0.9
        self.top_k = config.top_k if config else 50
        self.repetition_penalty = config.repetition_penalty if config else 1.0

    def __call__(
        self,
        logits: torch.Tensor,
        previous_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample the next token from logits.

        Args:
            logits: Unnormalized log probabilities [batch_size, vocab_size].
            previous_tokens: Previously generated tokens for repetition penalty.

        Returns:
            Sampled token IDs [batch_size, 1].
        """
        batch_size, vocab_size = logits.shape

        # 1. Apply repetition penalty
        if (
            self.repetition_penalty != 1.0
            and previous_tokens is not None
            and previous_tokens.numel() > 0
        ):
            # Gather logits for previously generated tokens
            # If logit > 0, divide by penalty. If < 0, multiply by penalty.
            score = torch.gather(logits, 1, previous_tokens)
            score = torch.where(
                score < 0,
                score * self.repetition_penalty,
                score / self.repetition_penalty,
            )
            logits.scatter_(1, previous_tokens, score)

        # 2. Apply temperature
        if self.temperature != 1.0 and self.temperature > 0.0:
            logits = logits / self.temperature

        # 3. Apply top-k filtering
        if self.top_k > 0 and self.top_k < vocab_size:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # 4. Apply top-p (nucleus) filtering
        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the filtered mask back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # 5. Sample
        probs = F.softmax(logits, dim=-1)

        if self.temperature == 0.0:
            # Greedy decoding
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # Multinomial sampling
            next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens
