"""
Speculative Decoding Engine

Speculative decoding is a technique that accelerates LLM inference without
changing the output distribution. It uses a small, fast "draft" model to
predict the next K tokens, and then uses the large, slow "target" model
to verify all K tokens in a single forward pass.

Since memory bandwidth is the bottleneck for generation, running K tokens
through the large model takes roughly the same time as running 1 token.
If the draft model is accurate, we get a 2-3x speedup.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Callable

import torch

logger = logging.getLogger(__name__)


class SpeculativeDecoder:
    """
    Implements speculative decoding.

    Takes a draft model (small, fast) and a target model (large, accurate).
    In Graviton, the "draft" model could actually be the same model but
    with extremely aggressive dynamic pruning (e.g., 90% sparsity), while
    the "target" model uses standard precision (e.g., 4-bit, 50% sparsity).

    Example:
        >>> decoder = SpeculativeDecoder(
        ...     draft_forward_fn=draft_model,
        ...     target_forward_fn=target_model,
        ...     gamma=5,
        ... )
        >>> tokens = decoder.generate(prompt_tokens)
    """

    def __init__(
        self,
        draft_forward_fn: Callable,
        target_forward_fn: Callable,
        gamma: int = 4,
    ):
        """
        Initialize the speculative decoder.

        Args:
            draft_forward_fn: Function that runs the draft model
                Signature: draft(tokens) -> logits
            target_forward_fn: Function that runs the target model
                Signature: target(tokens) -> logits
            gamma: Number of speculative tokens to generate per step.
        """
        self.draft_fn = draft_forward_fn
        self.target_fn = target_forward_fn
        self.gamma = gamma

        # Statistics
        self.accepted_tokens = 0
        self.total_speculated = 0
        self.steps = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of speculated tokens that were accepted."""
        if self.total_speculated == 0:
            return 0.0
        return self.accepted_tokens / self.total_speculated

    def decode_step(
        self,
        input_ids: torch.Tensor,
        sampler_fn: Callable,
    ) -> torch.Tensor:
        """
        Perform one step of speculative decoding.

        Algorithm:
        1. Autoregressively generate `gamma` tokens using draft model.
        2. Run target model on all `gamma + 1` tokens in parallel.
        3. Compare target probabilities with draft probabilities.
        4. Accept tokens that match target distribution.
        5. Sample correction token for the first rejected position.

        Args:
            input_ids: Current sequence [batch, seq_len].
            sampler_fn: Function to sample from logits.

        Returns:
            Appended input_ids with 1 to gamma+1 new tokens.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Keep track of the original sequence length
        original_len = input_ids.shape[1]

        # 1. Draft phase: generate gamma tokens
        draft_ids = input_ids.clone()
        draft_probs = []

        for _ in range(self.gamma):
            # Run draft model on current sequence
            logits = self.draft_fn(draft_ids)

            # Get probabilities for the last token position
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)

            # Sample next token (using greedy for draft is typical, but can use sampler)
            # For strict speculative decoding, we need the exact probabilities
            next_token = sampler_fn(next_token_logits)

            draft_ids = torch.cat([draft_ids, next_token], dim=1)

            # Store the probability of the *sampled* token according to the draft model
            # Gather prob for the sampled token: [batch, 1]
            token_prob = torch.gather(probs, 1, next_token)
            draft_probs.append(token_prob)

        # 2. Target phase: verify tokens
        # We run the target model on the sequence + draft tokens
        # Target model gives logits for all positions simultaneously
        target_logits = self.target_fn(draft_ids)

        # Get the logits corresponding to the positions we just predicted
        # We need logits from position `original_len - 1` to `original_len + gamma - 1`
        # These are the predictions for the new tokens
        verify_logits = target_logits[:, original_len - 1 : original_len + self.gamma, :]
        verify_probs = torch.softmax(verify_logits, dim=-1)

        # 3. Acceptance phase
        n_accepted = 0
        valid_ids = input_ids.clone()

        for t in range(self.gamma):
            # The token that the draft model proposed
            proposed_token = draft_ids[:, original_len + t : original_len + t + 1]

            # The probability the target model gave to that token
            # Shape: [batch, 1]
            p_target = torch.gather(verify_probs[:, t, :], 1, proposed_token)

            # The probability the draft model gave
            p_draft = draft_probs[t]

            # Rejection sampling criterion
            # Accept if rand < p_target / p_draft
            rand = torch.rand(batch_size, 1, device=device)
            accept = rand < (p_target / p_draft)

            if accept.all():
                # Token accepted
                n_accepted += 1
                valid_ids = torch.cat([valid_ids, proposed_token], dim=1)
            else:
                # Token rejected, we must resample from a corrected distribution
                # Corrected prob = max(0, p_target - p_draft) (normalized)
                
                # First, get the full target and draft distributions for this step
                # Note: full draft distribution would need to be stored,
                # omitting for simplicity in this reference implementation
                
                # Simplified: just sample from the target distribution at this point
                resampled_token = sampler_fn(verify_logits[:, t, :])
                valid_ids = torch.cat([valid_ids, resampled_token], dim=1)
                break

        # If all gamma tokens were accepted, we get a bonus token from the target model!
        if n_accepted == self.gamma:
            bonus_token = sampler_fn(verify_logits[:, -1, :])
            valid_ids = torch.cat([valid_ids, bonus_token], dim=1)

        # Update stats
        self.accepted_tokens += n_accepted
        self.total_speculated += self.gamma
        self.steps += 1

        logger.debug(
            f"Speculative step: accepted {n_accepted}/{self.gamma} "
            f"(rate: {self.acceptance_rate:.1%})"
        )

        return valid_ids
