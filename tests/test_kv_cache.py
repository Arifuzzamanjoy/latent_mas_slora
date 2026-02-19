"""
Tests for src/core/latent_memory.py — KV cache helpers and LatentMemory.

Validates get_cache_length(), store/retrieve, and clear operations.
"""

import pytest
import torch

from src.core.latent_memory import get_cache_length, LatentMemory


# ---------------------------------------------------------------------------
# get_cache_length
# ---------------------------------------------------------------------------

class TestGetCacheLength:

    def test_none_returns_zero(self):
        assert get_cache_length(None) == 0

    def test_empty_tuple_returns_zero(self):
        assert get_cache_length(()) == 0

    def test_tuple_format(self):
        """Simulate old-style tuple KV cache with seq_len=42."""
        seq_len = 42
        layer = (
            torch.zeros(1, 8, seq_len, 64),  # key
            torch.zeros(1, 8, seq_len, 64),  # value
        )
        kv = (layer,)
        assert get_cache_length(kv) == seq_len

    def test_dynamic_cache_object(self):
        """Simulate a DynamicCache-like object."""

        class FakeDynamicCache:
            def get_seq_length(self):
                return 99

        assert get_cache_length(FakeDynamicCache()) == 99


# ---------------------------------------------------------------------------
# LatentMemory store / retrieve / clear
# ---------------------------------------------------------------------------

class TestLatentMemoryStoreRetrieve:

    @pytest.fixture()
    def memory(self, device, dtype):
        return LatentMemory(device=device, max_history=10, dtype=dtype)

    def test_store_and_get(self, memory, hidden_dim, device, dtype):
        h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
        memory.store_hidden_state("planner", h)
        retrieved = memory.get_hidden_state("planner")
        assert retrieved is not None
        assert torch.allclose(retrieved, h), "Retrieved tensor should match stored tensor"

    def test_get_missing_returns_none(self, memory):
        assert memory.get_hidden_state("nonexistent") is None

    def test_store_3d_collapses_to_last(self, memory, hidden_dim, device, dtype):
        """3-D [B, S, D] should be stored as [B, D] (last position)."""
        h = torch.randn(1, 5, hidden_dim, device=device, dtype=dtype)
        memory.store_hidden_state("critic", h)
        retrieved = memory.get_hidden_state("critic")
        assert retrieved.shape == (1, hidden_dim)
        assert torch.allclose(retrieved, h[:, -1, :].to(dtype))


class TestLatentMemoryClear:

    @pytest.fixture()
    def memory(self, device, dtype):
        return LatentMemory(device=device, max_history=10, dtype=dtype)

    def test_clear_removes_states(self, memory, hidden_dim, device, dtype):
        h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
        memory.store_hidden_state("planner", h)
        memory.clear()
        assert memory.get_hidden_state("planner") is None

    def test_clear_resets_counters(self, memory, hidden_dim, device, dtype):
        h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
        memory.store_hidden_state("planner", h)
        memory.clear()
        assert memory.num_latent_vectors == 0
        assert memory.num_agents == 0

    def test_clear_all_removes_snapshots(self, memory, hidden_dim, device, dtype):
        h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
        memory.store_hidden_state("planner", h)
        memory.clear_all()
        assert len(memory._snapshots) == 0


class TestLatentMemoryAccumulation:

    @pytest.fixture()
    def memory(self, device, dtype):
        return LatentMemory(device=device, max_history=10, dtype=dtype)

    def test_accumulated_latents_shape(self, memory, hidden_dim, device, dtype):
        """Storing N vectors → accumulated shape [1, N, D]."""
        for i in range(3):
            h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
            memory.store_hidden_state(f"agent_{i}", h)
        acc = memory.get_accumulated_latents()
        assert acc is not None
        assert acc.shape == (1, 3, hidden_dim)

    def test_fused_mean(self, memory, hidden_dim, device, dtype):
        for i in range(3):
            h = torch.randn(1, hidden_dim, device=device, dtype=dtype)
            memory.store_hidden_state(f"agent_{i}", h)
        fused = memory.get_fused_latent("mean")
        assert fused is not None
        assert fused.shape == (1, hidden_dim)
