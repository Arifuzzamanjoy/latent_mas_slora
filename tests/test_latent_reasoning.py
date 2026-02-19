"""
Tests for src/core/latent_reasoner.py — latent reasoning loop.

Validates the multi-step reason() method: forward passes, early exit,
and hidden-state evolution over steps.
"""

import pytest
import torch
import torch.nn.functional as F

from src.core.latent_reasoner import LatentReasoner


@pytest.fixture(scope="module")
def reasoner(model, device, dtype):
    """LatentReasoner wired to the 7B model."""
    return LatentReasoner(
        model=model,
        device=device,
        dtype=dtype,
        adaptive_steps=False,   # disable adaptive so we control steps exactly
        min_steps=1,
        max_steps=50,
    )


@pytest.fixture(scope="module")
def sample_input(tokenizer, device):
    """Tokenised 'Hello world' as a simple input."""
    enc = tokenizer("Hello world", return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


class TestSingleLatentStep:
    """reason() with num_steps=1."""

    def test_result_fields(self, reasoner, sample_input, hidden_dim):
        ids, mask = sample_input
        result = reasoner.reason(ids, mask, num_steps=1, early_exit_threshold=0)
        assert result.final_hidden is not None
        assert result.kv_cache is not None
        assert result.num_steps == 1

    def test_final_hidden_shape(self, reasoner, sample_input, hidden_dim):
        ids, mask = sample_input
        result = reasoner.reason(ids, mask, num_steps=1, early_exit_threshold=0)
        assert result.final_hidden.shape == (1, hidden_dim), (
            f"Expected (1, {hidden_dim}), got {result.final_hidden.shape}"
        )


class TestMultipleLatentSteps:
    """reason() with num_steps=5."""

    def test_num_steps_matches(self, reasoner, sample_input):
        ids, mask = sample_input
        result = reasoner.reason(ids, mask, num_steps=5, early_exit_threshold=0)
        assert result.num_steps == 5

    def test_final_hidden_shape(self, reasoner, sample_input, hidden_dim):
        ids, mask = sample_input
        result = reasoner.reason(ids, mask, num_steps=5, early_exit_threshold=0)
        assert result.final_hidden.shape == (1, hidden_dim)

    def test_all_hidden_states_collected(self, reasoner, sample_input):
        ids, mask = sample_input
        result = reasoner.reason(
            ids, mask, num_steps=5, return_all_hidden=True, early_exit_threshold=0,
        )
        # all_hidden includes initial + 5 steps = 6
        assert len(result.all_hidden_states) == 6, (
            f"Expected 6 hidden states (1 initial + 5 steps), got {len(result.all_hidden_states)}"
        )


class TestEarlyExit:
    """Aggressive threshold should cause early exit."""

    def test_exits_before_max_steps(self, reasoner, sample_input):
        ids, mask = sample_input
        result = reasoner.reason(
            ids, mask,
            num_steps=50,
            early_exit_threshold=0.5,  # very aggressive → converge fast
        )
        assert result.num_steps < 50, (
            f"Expected early exit before 50 steps, but took {result.num_steps}"
        )


class TestLatentStepsChangeHiddenState:
    """More steps should produce a different hidden state."""

    def test_zero_vs_five_steps_differ(self, reasoner, sample_input, device):
        ids, mask = sample_input

        # 0 latent steps ≈ just the initial forward pass hidden state
        r0 = reasoner.reason(ids, mask, num_steps=0, early_exit_threshold=0)
        # 5 latent steps
        r5 = reasoner.reason(ids, mask, num_steps=5, early_exit_threshold=0)

        cos = F.cosine_similarity(
            r0.final_hidden.float().view(1, -1),
            r5.final_hidden.float().view(1, -1),
            dim=-1,
        ).item()

        assert cos < 0.99, (
            f"Cosine sim between 0-step and 5-step is {cos:.4f}; "
            f"expected < 0.99 — reasoning should change the representation"
        )
