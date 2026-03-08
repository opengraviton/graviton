"""Tests for Sampler, SpeculativeDecoder, and end-to-end generation logic."""

import torch
import torch.nn.functional as F

from graviton.decoding.sampling import Sampler
from graviton.decoding.speculative import SpeculativeDecoder
from graviton.core.config import DecodingConfig


# ── Sampler ─────────────────────────────────────────────────────────


def test_sampler_greedy():
    """Temperature 0 should always pick the argmax."""
    cfg = DecodingConfig(temperature=0.0, top_p=1.0, top_k=0)
    sampler = Sampler(cfg)

    logits = torch.tensor([[0.1, 0.3, 5.0, 0.2]])
    token = sampler(logits)
    assert token.item() == 2


def test_sampler_temperature_scaling():
    """Lower temperature should sharpen the distribution (more greedy)."""
    cfg_hot = DecodingConfig(temperature=2.0, top_p=1.0, top_k=0)
    cfg_cold = DecodingConfig(temperature=0.1, top_p=1.0, top_k=0)

    logits = torch.tensor([[1.0, 2.0, 5.0, 0.5]])

    torch.manual_seed(0)
    tokens_cold = [Sampler(cfg_cold)(logits.clone()).item() for _ in range(20)]
    assert all(t == 2 for t in tokens_cold), "Cold temperature should always pick the max"


def test_sampler_top_k():
    """Top-k should restrict sampling to the top-k tokens."""
    cfg = DecodingConfig(temperature=1.0, top_k=2, top_p=1.0)
    sampler = Sampler(cfg)

    logits = torch.tensor([[1.0, 5.0, 3.0, 0.1]])
    torch.manual_seed(42)
    tokens = set(sampler(logits.clone()).item() for _ in range(50))
    assert tokens.issubset({1, 2}), f"Top-2 should only pick indices 1 or 2, got {tokens}"


def test_sampler_top_p():
    """Top-p should restrict sampling to the nucleus."""
    cfg = DecodingConfig(temperature=1.0, top_k=0, top_p=0.01)
    sampler = Sampler(cfg)

    logits = torch.tensor([[0.1, 0.2, 10.0, 0.05]])
    torch.manual_seed(42)
    tokens = set(sampler(logits.clone()).item() for _ in range(20))
    assert tokens == {2}, f"Very tight top-p should only pick the top token, got {tokens}"


def test_sampler_repetition_penalty():
    """Repetition penalty should reduce the probability of previous tokens."""
    cfg_no_rep = DecodingConfig(temperature=0.0, repetition_penalty=1.0, top_k=0, top_p=1.0)
    cfg_rep = DecodingConfig(temperature=0.0, repetition_penalty=2.0, top_k=0, top_p=1.0)

    logits = torch.tensor([[3.0, 3.0, 3.01, 3.0]])
    prev = torch.tensor([[2]])

    token_no_rep = Sampler(cfg_no_rep)(logits.clone(), previous_tokens=prev).item()
    token_rep = Sampler(cfg_rep)(logits.clone(), previous_tokens=prev).item()

    assert token_no_rep == 2, "Without penalty token 2 should win (slightly higher)"
    assert token_rep != 2, "With penalty token 2 should be penalized away"


def test_sampler_output_shape():
    """Sampler should return [batch, 1] shape."""
    cfg = DecodingConfig(temperature=1.0)
    sampler = Sampler(cfg)

    logits = torch.randn(3, 100)
    tokens = sampler(logits)
    assert tokens.shape == (3, 1)


# ── SpeculativeDecoder ──────────────────────────────────────────────


def _make_dummy_model(vocab_size=16, hidden=32):
    """A simple linear model that maps token embeddings -> logits."""

    class TinyModel:
        def __init__(self):
            self.embed = torch.nn.Embedding(vocab_size, hidden)
            self.head = torch.nn.Linear(hidden, vocab_size, bias=False)

        def __call__(self, ids):
            h = self.embed(ids)
            return self.head(h)

    return TinyModel()


def test_speculative_decoder_all_accepted():
    """When draft == target, all tokens should be accepted."""
    torch.manual_seed(42)
    model = _make_dummy_model()
    decoder = SpeculativeDecoder(
        draft_forward_fn=model, target_forward_fn=model, gamma=3
    )

    prompt = torch.tensor([[1, 2, 3]])
    sampler = Sampler(DecodingConfig(temperature=0.0, top_k=0, top_p=1.0))

    result = decoder.decode_step(prompt, sampler_fn=sampler)

    assert result.shape[1] > prompt.shape[1]
    # Same model => all gamma should be accepted + 1 bonus
    assert decoder.accepted_tokens == 3
    assert result.shape[1] == prompt.shape[1] + 3 + 1  # original + gamma + bonus


def test_speculative_decoder_stats():
    """acceptance_rate should be correct after one step."""
    torch.manual_seed(0)
    model = _make_dummy_model()
    decoder = SpeculativeDecoder(
        draft_forward_fn=model, target_forward_fn=model, gamma=4
    )

    prompt = torch.tensor([[5, 6]])
    sampler = Sampler(DecodingConfig(temperature=0.0, top_k=0, top_p=1.0))
    decoder.decode_step(prompt, sampler_fn=sampler)

    assert decoder.total_speculated == 4
    assert 0.0 <= decoder.acceptance_rate <= 1.0
    assert decoder.steps == 1


def test_speculative_decoder_different_models():
    """With different draft/target, some tokens may be rejected."""
    torch.manual_seed(42)
    draft = _make_dummy_model()
    target = _make_dummy_model()

    decoder = SpeculativeDecoder(
        draft_forward_fn=draft, target_forward_fn=target, gamma=3
    )

    prompt = torch.tensor([[1, 2]])
    sampler = Sampler(DecodingConfig(temperature=0.0, top_k=0, top_p=1.0))
    result = decoder.decode_step(prompt, sampler_fn=sampler)

    # Should produce at least 1 new token (the correction or accepted ones)
    assert result.shape[1] > prompt.shape[1]
