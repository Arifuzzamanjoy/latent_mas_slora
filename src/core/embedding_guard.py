"""
Embedding Guard - Verifies LoRA adapters don't modify embedding layers.

Critical safety mechanism: LoRA adapters should target attention/MLP layers,
NOT embedding or language-model head layers. Modifying embeddings can silently
corrupt the shared vocabulary representation, breaking all downstream tasks.

Usage:
    guard = EmbeddingGuard(model)
    # ... load adapter ...
    guard.assert_frozen()  # Raises RuntimeError if embeddings changed
"""

import hashlib
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EmbeddingGuard:
    """Verifies LoRA adapters don't modify embedding layers."""

    def __init__(self, model):
        """Snapshot embed_tokens and lm_head weight hashes at init."""
        self._model = model
        self._embed_hash: Optional[str] = None
        self._lm_head_hash: Optional[str] = None
        self._snapshot()

    def _get_base_model(self):
        """Unwrap PEFT wrappers to get the base model."""
        m = self._model
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        elif hasattr(m, "base_model"):
            m = m.base_model
        return m

    @staticmethod
    def _hash_tensor(t: torch.Tensor) -> str:
        """Return SHA-256 hex digest of a tensor's raw bytes."""
        data = t.detach().cpu().float().contiguous().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def _snapshot(self) -> None:
        """Store SHA-256 hashes of embed_tokens and lm_head weights."""
        # Use the model directly — get_input/output_embeddings work
        # on both PeftModel and CausalLM without manual unwrapping.
        model = self._model

        embed = model.get_input_embeddings()
        if embed is not None and hasattr(embed, "weight"):
            self._embed_hash = self._hash_tensor(embed.weight)
            logger.debug("EmbeddingGuard: embed_tokens hash captured")

        lm_head = model.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            self._lm_head_hash = self._hash_tensor(lm_head.weight)
            logger.debug("EmbeddingGuard: lm_head hash captured")

    def verify(self) -> bool:
        """Compare current hashes to snapshots. Return True if unchanged."""
        model = self._model

        # Check embed_tokens
        if self._embed_hash is not None:
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, "weight"):
                current = self._hash_tensor(embed.weight)
                if current != self._embed_hash:
                    logger.error("EmbeddingGuard: embed_tokens weights have been modified!")
                    return False

        # Check lm_head
        if self._lm_head_hash is not None:
            lm_head = model.get_output_embeddings()
            if lm_head is None:
                lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                current = self._hash_tensor(lm_head.weight)
                if current != self._lm_head_hash:
                    logger.error("EmbeddingGuard: lm_head weights have been modified!")
                    return False

        return True

    def assert_frozen(self) -> None:
        """Raise RuntimeError if embeddings have been modified."""
        if not self.verify():
            raise RuntimeError(
                "Embedding layer weights changed after adapter load. "
                "This adapter likely targets 'embed_tokens' or 'lm_head', "
                "which is forbidden. Unload it immediately."
            )
