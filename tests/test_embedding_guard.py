"""
Tests for src/core/embedding_guard.py — EmbeddingGuard.

Validates that the guard correctly snapshots embedding hashes and
detects unauthorized modifications.
"""

import pytest
import torch

from src.core.embedding_guard import EmbeddingGuard


class TestGuardPassesOnFreshModel:
    """A freshly created guard on an unmodified model should pass."""

    def test_verify_returns_true(self, model):
        guard = EmbeddingGuard(model)
        assert guard.verify() is True

    def test_assert_frozen_does_not_raise(self, model):
        guard = EmbeddingGuard(model)
        guard.assert_frozen()  # should not raise

    def test_hashes_are_captured(self, model):
        guard = EmbeddingGuard(model)
        assert guard._embed_hash is not None, "embed_tokens hash should be captured"
        assert guard._lm_head_hash is not None, "lm_head hash should be captured"


class TestGuardDetectsModification:
    """Mutating embed_tokens must cause verify() to return False."""

    def test_modified_embed_tokens_detected(self, model):
        guard = EmbeddingGuard(model)

        # Get embedding layer and save original value
        base = guard._get_base_model()
        embed = base.get_input_embeddings()
        original_val = embed.weight.data[0, 0].item()

        # Mutate
        with torch.no_grad():
            embed.weight.data[0, 0] += 1.0

        try:
            assert guard.verify() is False, (
                "verify() should return False after embed_tokens modification"
            )
        finally:
            # Always restore
            with torch.no_grad():
                embed.weight.data[0, 0] = original_val

    def test_assert_frozen_raises(self, model):
        guard = EmbeddingGuard(model)

        base = guard._get_base_model()
        embed = base.get_input_embeddings()
        original_val = embed.weight.data[0, 0].item()

        with torch.no_grad():
            embed.weight.data[0, 0] += 1.0

        try:
            with pytest.raises(RuntimeError, match="Embedding layer weights changed"):
                guard.assert_frozen()
        finally:
            with torch.no_grad():
                embed.weight.data[0, 0] = original_val

    def test_restored_weight_passes_again(self, model):
        """After restoring the weight the guard (re-created) should pass."""
        guard = EmbeddingGuard(model)
        assert guard.verify() is True
