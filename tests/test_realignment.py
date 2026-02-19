"""
Tests for src/core/latent_reasoner.py — realignment logic.

Verifies the W_align matrix construction and the realign() transformation
that maps output hidden states back to input embedding space.
"""

import pytest
import torch
import torch.nn.functional as F

from src.core.latent_reasoner import LatentReasoner


@pytest.fixture(scope="module")
def reasoner(model, device, dtype):
    """Create a LatentReasoner with the 7B model."""
    return LatentReasoner(
        model=model,
        device=device,
        dtype=dtype,
        adaptive_steps=True,
        min_steps=3,
        max_steps=20,
    )


class TestBuildRealignmentMatrix:
    """Test _build_realignment_matrix() outputs."""

    def test_realign_matrix_is_not_none(self, reasoner):
        assert reasoner._realign_matrix is not None, (
            "Realignment matrix should be built on init"
        )

    def test_realign_matrix_shape(self, reasoner, hidden_dim):
        mat = reasoner._realign_matrix
        assert mat.shape == (hidden_dim, hidden_dim), (
            f"Expected ({hidden_dim}, {hidden_dim}), got {mat.shape}"
        )

    def test_target_norm_is_positive_scalar(self, reasoner):
        tn = reasoner._target_norm
        assert tn is not None
        assert tn.dim() == 0, f"Expected scalar, got tensor with {tn.dim()} dims"
        assert tn.item() > 0, f"Target norm should be positive, got {tn.item()}"


class TestRealignPreservesShape:
    """Realign must not change tensor shape."""

    def test_3d_input(self, reasoner, hidden_dim, device, dtype):
        h = torch.randn(1, 1, hidden_dim, device=device, dtype=dtype)
        out = reasoner.realign(h)
        assert out.shape == h.shape, f"Shape mismatch: {out.shape} vs {h.shape}"

    def test_2d_input(self, reasoner, hidden_dim, device, dtype):
        h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
        out = reasoner.realign(h)
        assert out.shape == h.shape, f"Shape mismatch: {out.shape} vs {h.shape}"

    def test_batch_input(self, reasoner, hidden_dim, device, dtype):
        h = torch.randn(4, 1, hidden_dim, device=device, dtype=dtype)
        out = reasoner.realign(h)
        assert out.shape == h.shape


class TestRealignMapsToEmbeddingSpace:
    """Realigned vectors should land in the neighborhood of embedding space."""

    def test_realigned_output_has_finite_values_and_reasonable_norm(self, model, reasoner, hidden_dim, device):
        """Realignment should produce finite outputs with norms in a reasonable range."""
        embed_weight = model.get_input_embeddings().weight  # [vocab, D]
        avg_embed_norm = embed_weight.detach().float().norm(dim=-1).mean().item()

        # Create a synthetic "model output" hidden state with similar magnitude
        h = torch.randn(1, 1, hidden_dim, device=device, dtype=reasoner.dtype)
        h = h * (avg_embed_norm / h.float().norm().item())

        realigned = reasoner.realign(h)
        r_norm = realigned.float().norm().item()

        assert torch.isfinite(realigned).all(), "Realigned output contains non-finite values"
        # Norm shouldn't explode (>100x embedding norm) or vanish (<1e-6)
        assert r_norm > 1e-6, f"Realigned norm is ~0 ({r_norm:.6f})"
        assert r_norm < avg_embed_norm * 100, (
            f"Realigned norm ({r_norm:.1f}) exploded relative to avg embed norm ({avg_embed_norm:.1f})"
        )


class TestRealignmentWithoutMatrix:
    """When _realign_matrix is None, realign() should be identity."""

    def test_returns_input_unchanged(self, model, hidden_dim, device, dtype):
        # Create a separate reasoner and disable realignment
        reasoner = LatentReasoner(model=model, device=device, dtype=dtype)
        reasoner._realign_matrix = None

        h = torch.randn(1, 1, hidden_dim, device=device, dtype=dtype)
        out = reasoner.realign(h)
        assert torch.equal(out, h), "With no realignment matrix, output should equal input"
