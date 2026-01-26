"""
Latent Reasoner - Core latent space operations

Implements:
1. Latent space realignment (LatentMAS paper Eq. 3)
2. Continuous thought generation (Coconut-style)
3. Multi-step latent reasoning without token decoding

Optimized for 24-48GB VRAM with full precision.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


def get_cache_length(kv_cache) -> int:
    """Get the sequence length from a KV cache, handling both tuple and DynamicCache formats."""
    if kv_cache is None:
        return 0
    # Handle DynamicCache object (newer transformers)
    if hasattr(kv_cache, 'get_seq_length'):
        return kv_cache.get_seq_length()
    # Handle tuple format (older transformers)
    if isinstance(kv_cache, (tuple, list)) and len(kv_cache) > 0:
        if isinstance(kv_cache[0], (tuple, list)) and len(kv_cache[0]) > 0:
            return kv_cache[0][0].shape[-2]
    return 0


@dataclass
class LatentReasoningResult:
    """Result from latent reasoning pass"""
    final_hidden: torch.Tensor
    all_hidden_states: List[torch.Tensor]
    kv_cache: Optional[Tuple]
    num_steps: int


class LatentReasoner:
    """
    Core Latent Reasoning Engine.
    
    Implements the key insight from LatentMAS: reasoning can happen
    in continuous latent space without expensive token decoding.
    
    For 48GB VRAM:
    - Uses full BF16 precision
    - Supports up to 50 latent reasoning steps
    - Batch sizes up to 8 for parallel reasoning
    """
    
    def __init__(
        self,
        model,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        
        # Realignment matrices
        self._realign_matrix: Optional[torch.Tensor] = None
        self._target_norm: Optional[torch.Tensor] = None
        
        # Build realignment on init
        self._build_realignment_matrix()
    
    def _build_realignment_matrix(self) -> None:
        """
        Build latent space realignment matrix (LatentMAS paper Eq. 3).
        
        This maps output hidden states back to input embedding space,
        enabling continuous reasoning without token decoding.
        
        W_align = (W_out^T W_out + λI)^(-1) W_out^T W_in
        """
        try:
            # Get embedding layers
            base_model = self._get_base_model()
            input_embed = base_model.get_input_embeddings()
            output_embed = base_model.get_output_embeddings()
            
            if output_embed is None:
                output_embed = getattr(base_model, "lm_head", None)
            
            if input_embed is None or output_embed is None:
                print("[WARN] Cannot build realignment matrix - missing embeddings")
                return
            
            with torch.no_grad():
                # Get weights in float32 for numerical stability
                W_in = input_embed.weight.detach().float().to(self.device)
                W_out = output_embed.weight.detach().float().to(self.device)
                
                # Compute (W_out^T @ W_out + λI)
                gram = W_out.T @ W_out
                reg = 1e-5 * torch.eye(gram.shape[0], device=self.device, dtype=gram.dtype)
                gram = gram + reg
                
                # Solve for realignment matrix
                rhs = W_out.T @ W_in
                self._realign_matrix = torch.linalg.solve(gram, rhs)
                
                # Target norm for output scaling
                self._target_norm = W_in.norm(dim=1).mean()
                
                print(f"[INFO] Realignment matrix built: {self._realign_matrix.shape}")
                
        except Exception as e:
            print(f"[ERROR] Failed to build realignment matrix: {e}")
            self._realign_matrix = None
    
    def _get_base_model(self):
        """Get base model (handles PEFT wrapper)"""
        if hasattr(self.model, 'get_base_model'):
            return self.model.get_base_model()
        if hasattr(self.model, 'base_model'):
            return self.model.base_model
        return self.model
    
    def realign(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply latent space realignment.
        
        Maps output hidden states back to input embedding space
        for continuous reasoning.
        
        Args:
            hidden: Hidden state tensor [..., hidden_dim]
            
        Returns:
            Realigned tensor with same shape
        """
        if self._realign_matrix is None:
            return hidden
        
        original_dtype = hidden.dtype
        
        # Compute in float32
        h = hidden.float()
        aligned = h @ self._realign_matrix
        
        # Normalize to target embedding scale
        norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (self._target_norm / norm)
        
        return aligned.to(original_dtype)
    
    @torch.no_grad()
    def reason(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_steps: int = 10,
        past_key_values: Optional[Tuple] = None,
        return_all_hidden: bool = False,
    ) -> LatentReasoningResult:
        """
        Perform multi-step latent reasoning.
        
        This is the core of LatentMAS - each step:
        1. Forward pass to get hidden state
        2. Realign hidden state to input space
        3. Use as input for next step (no token decoding!)
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_steps: Number of latent reasoning steps
            past_key_values: Optional KV cache from previous agents
            return_all_hidden: Whether to return all intermediate hidden states
            
        Returns:
            LatentReasoningResult with final hidden state and optional intermediates
        """
        all_hidden = []
        
        # Initial forward pass
        if past_key_values is not None:
            past_len = get_cache_length(past_key_values)
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=self.device,
            )
            full_mask = torch.cat([past_mask, attention_mask], dim=-1)
        else:
            full_mask = attention_mask
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=full_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        
        kv_cache = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1:, :]  # [B, 1, D]
        
        if return_all_hidden:
            all_hidden.append(last_hidden.squeeze(1))
        
        # Latent reasoning steps
        for step in range(num_steps):
            # Realign to input embedding space
            aligned = self.realign(last_hidden)
            
            # Create attention mask for latent step
            cache_len = get_cache_length(kv_cache)
            step_mask = torch.ones(
                (aligned.shape[0], cache_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            
            # Forward pass with aligned embedding (NO TOKEN DECODING)
            outputs = self.model(
                inputs_embeds=aligned,
                attention_mask=step_mask,
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            
            kv_cache = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1:, :]
            
            if return_all_hidden:
                all_hidden.append(last_hidden.squeeze(1))
        
        return LatentReasoningResult(
            final_hidden=last_hidden.squeeze(1),
            all_hidden_states=all_hidden,
            kv_cache=kv_cache,
            num_steps=num_steps,
        )
    
    @torch.no_grad()
    def reason_from_hidden(
        self,
        hidden_state: torch.Tensor,
        num_steps: int = 5,
        kv_cache: Optional[Tuple] = None,
    ) -> LatentReasoningResult:
        """
        Continue latent reasoning from a hidden state.
        
        Useful for agent-to-agent latent transfer.
        
        Args:
            hidden_state: Starting hidden state [B, D] or [B, 1, D]
            num_steps: Number of reasoning steps
            kv_cache: Optional KV cache
            
        Returns:
            LatentReasoningResult
        """
        all_hidden = []
        
        # Ensure correct shape
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1)  # [B, 1, D]
        
        current_hidden = hidden_state
        current_cache = kv_cache
        
        for step in range(num_steps):
            aligned = self.realign(current_hidden)
            
            # Build attention mask
            cache_len = get_cache_length(current_cache)
            
            mask = torch.ones(
                (aligned.shape[0], cache_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            
            outputs = self.model(
                inputs_embeds=aligned,
                attention_mask=mask,
                past_key_values=current_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            
            current_cache = outputs.past_key_values
            current_hidden = outputs.hidden_states[-1][:, -1:, :]
            all_hidden.append(current_hidden.squeeze(1))
        
        return LatentReasoningResult(
            final_hidden=current_hidden.squeeze(1),
            all_hidden_states=all_hidden,
            kv_cache=current_cache,
            num_steps=num_steps,
        )


class LatentFusion:
    """
    Fusion strategies for combining latent representations.
    
    Used when aggregating outputs from multiple agents.
    """
    
    @staticmethod
    def mean_fusion(hiddens: List[torch.Tensor]) -> torch.Tensor:
        """Simple mean of hidden states"""
        return torch.stack(hiddens).mean(dim=0)
    
    @staticmethod
    def weighted_fusion(
        hiddens: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Weighted combination of hidden states"""
        if weights is None:
            weights = [1.0] * len(hiddens)
        
        weights = torch.tensor(weights, device=hiddens[0].device, dtype=hiddens[0].dtype)
        weights = weights / weights.sum()
        
        stacked = torch.stack(hiddens)  # [N, B, D]
        weights = weights.view(-1, 1, 1)
        
        return (stacked * weights).sum(dim=0)
    
    @staticmethod
    def attention_fusion(
        hiddens: List[torch.Tensor],
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attention-based fusion.
        
        If query is provided, uses it as query; otherwise uses mean as query.
        """
        stacked = torch.stack(hiddens, dim=1)  # [B, N, D]
        
        if query is None:
            query = stacked.mean(dim=1, keepdim=True)  # [B, 1, D]
        else:
            query = query.unsqueeze(1) if query.dim() == 2 else query
        
        # Scaled dot-product attention
        d_k = query.shape[-1]
        scores = torch.matmul(query, stacked.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        fused = torch.matmul(attn_weights, stacked).squeeze(1)
        return fused
    
    @staticmethod
    def concat_project(
        hiddens: List[torch.Tensor],
        projection: Optional[torch.nn.Linear] = None,
    ) -> torch.Tensor:
        """Concatenate and project to original dimension"""
        concatenated = torch.cat(hiddens, dim=-1)  # [B, N*D]
        
        if projection is not None:
            return projection(concatenated)
        else:
            # Simple mean projection
            n = len(hiddens)
            d = hiddens[0].shape[-1]
            return concatenated.view(-1, n, d).mean(dim=1)
