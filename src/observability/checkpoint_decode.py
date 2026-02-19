"""
Checkpoint Decoder – force-decode hidden states at any latent step.

The key debugging tool for LatentMAS.  At any point during latent
reasoning we can project the current hidden state through the LM head
and greedily decode a few tokens to see what the agent "would say"
if forced to produce text right now.

This never alters the pipeline — it is a read-only probe.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.latent_reasoner import LatentReasoningResult, get_cache_length

logger = logging.getLogger(__name__)


class CheckpointDecoder:
    """Force-decode from a KV-cache or hidden state to readable text.

    Parameters
    ----------
    model : PreTrainedModel
        The causal LM (or PeftModel wrapping one).
    tokenizer : PreTrainedTokenizer
        Matching tokenizer.
    device : str
        Torch device for any tensors this class creates.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_hidden(
        self,
        hidden_state: torch.Tensor,
        max_new_tokens: int = 50,
        kv_cache: Optional[Tuple] = None,
    ) -> str:
        """Force-decode from a hidden state.

        Projects *hidden_state* through the LM head to obtain a
        distribution over the vocabulary, then greedily extends
        the sequence up to *max_new_tokens*.

        Parameters
        ----------
        hidden_state : Tensor
            Shape ``[B, D]`` or ``[B, 1, D]``.
        max_new_tokens : int
            How many tokens to decode.
        kv_cache : tuple | None
            Optional existing KV-cache to condition on.

        Returns
        -------
        str
            Decoded text (greedy, skip special tokens).
        """
        h = hidden_state
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [B, 1, D]
        h = h.to(self.device)

        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            # Project through lm_head to get logits
            logits = self._lm_head_forward(h)          # [B, 1, V]
            next_id = logits[:, -1, :].argmax(dim=-1)  # [B]
            token_id = next_id.item()

            if token_id == self.tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)

            # Embed the chosen token for the next step
            h = self._embed_token(next_id)  # [B, 1, D]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def decode_kv_cache(
        self,
        kv_cache: Tuple,
        max_new_tokens: int = 50,
    ) -> str:
        """Force-decode by continuing generation from a KV-cache.

        Starts with a single BOS / pad token so the model has an
        ``input_ids`` tensor, but all context comes from the cache.

        Parameters
        ----------
        kv_cache : tuple
            Past key-values from a model forward pass.
        max_new_tokens : int
            Tokens to generate.

        Returns
        -------
        str
        """
        # Seed with a single padding token
        seed_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = torch.tensor([[seed_id]], device=self.device)

        cache_len = get_cache_length(kv_cache)
        attn_mask = torch.ones((1, cache_len + 1), dtype=torch.long, device=self.device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=kv_cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Pipeline trace
    # ------------------------------------------------------------------

    def decode_pipeline_trace(
        self,
        all_hidden_states: List[torch.Tensor],
        agent_names: Optional[List[str]] = None,
        interval: int = 3,
        max_new_tokens: int = 50,
    ) -> List[Dict[str, Any]]:
        """Decode checkpoints at every *interval* latent steps.

        Parameters
        ----------
        all_hidden_states : list[Tensor]
            Collected hidden states (one per latent step, flattened
            across agents).  Typically obtained by running the reasoner
            with ``return_all_hidden=True``.
        agent_names : list[str] | None
            If provided, maps indices to agent names for display.
        interval : int
            Decode every N-th hidden state (to save time).
        max_new_tokens : int
            Tokens per checkpoint decode.

        Returns
        -------
        list[dict]
            Each dict has ``{"agent": str, "step": int, "decoded_text": str}``.
        """
        trace: list[dict] = []
        for idx, hs in enumerate(all_hidden_states):
            if idx % interval != 0:
                continue
            text = self.decode_hidden(hs, max_new_tokens=max_new_tokens)
            agent = agent_names[idx] if agent_names and idx < len(agent_names) else f"step_{idx}"
            trace.append({
                "agent": agent,
                "step": idx,
                "decoded_text": text,
            })
        return trace

    # ------------------------------------------------------------------
    # Confidence probe
    # ------------------------------------------------------------------

    @torch.no_grad()
    def probe_confidence(
        self,
        hidden_state: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Probe the LM head to see top-k token probabilities.

        Useful for understanding how "decided" the model is at a
        given latent step.

        Returns
        -------
        dict
            ``{"top_tokens": [(token_str, prob), ...], "entropy": float,
              "max_prob": float}``
        """
        h = hidden_state
        if h.dim() == 2:
            h = h.unsqueeze(1)
        h = h.to(self.device)

        logits = self._lm_head_forward(h)[:, -1, :]  # [B, V]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_ids = probs.topk(top_k, dim=-1)
        top_tokens = [
            (self.tokenizer.decode([tid.item()]), round(p.item(), 4))
            for tid, p in zip(topk_ids[0], topk_probs[0])
        ]

        entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).item()
        max_prob = topk_probs[0, 0].item()

        return {
            "top_tokens": top_tokens,
            "entropy": round(entropy, 4),
            "max_prob": round(max_prob, 4),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state → logits through the LM head."""
        lm_head = self.model.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Cannot find lm_head on the model")
        return lm_head(hidden)

    def _embed_token(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids → hidden dim.  *token_ids* shape ``[B]``."""
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.unsqueeze(1)  # [B, 1]
        embed = self.model.get_input_embeddings()
        return embed(token_ids)  # [B, 1, D]
