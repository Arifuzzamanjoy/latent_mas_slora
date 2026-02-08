"""
Latent Reasoner - Core latent space operations

Implements:
1. Latent space realignment (LatentMAS paper Eq. 3)
2. Continuous thought generation (Coconut-style)
3. Multi-step latent reasoning without token decoding
4. Adaptive latent steps based on complexity (NEW)
5. Parallel reasoning branches (NEW)

Optimized for 24-48GB VRAM with full precision.
"""

import logging
import re
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


@dataclass
class ComplexityEstimate:
    """Estimated complexity of a question"""
    score: float  # 0.0 (simple) to 1.0 (complex)
    recommended_steps: int
    features: dict


class ComplexityEstimator:
    """
    Estimate question complexity to determine optimal latent steps.
    
    Based on Inference Scaling Laws research - simple questions need
    fewer reasoning steps, while complex ones benefit from more.
    """
    
    # Complexity indicators
    MATH_PATTERNS = [
        r'calculate', r'compute', r'solve', r'equation', r'formula',
        r'\d+\s*[\+\-\*/\^]\s*\d+', r'integral', r'derivative', r'probability',
    ]
    
    CODE_PATTERNS = [
        r'function', r'code', r'implement', r'algorithm', r'program',
        r'python', r'javascript', r'debug', r'syntax',
    ]
    
    REASONING_PATTERNS = [
        r'compare', r'analyze', r'evaluate', r'explain\s+why',
        r'pros\s+and\s+cons', r'difference\s+between', r'vs\.?',
    ]
    
    MEDICAL_COMPLEX = [
        r'differential\s+diagnosis', r'mechanism\s+of\s+action',
        r'pathophysiology', r'contraindication', r'drug\s+interaction',
    ]
    
    @classmethod
    def estimate(cls, question: str, min_steps: int = 3, max_steps: int = 20) -> ComplexityEstimate:
        """
        Estimate complexity and recommend latent steps.
        
        Args:
            question: The input question
            min_steps: Minimum latent steps (for simple questions)
            max_steps: Maximum latent steps (for complex questions)
            
        Returns:
            ComplexityEstimate with score and recommended steps
        """
        q_lower = question.lower()
        
        features = {
            'word_count': len(question.split()),
            'has_math': any(re.search(p, q_lower) for p in cls.MATH_PATTERNS),
            'has_code': any(re.search(p, q_lower) for p in cls.CODE_PATTERNS),
            'has_reasoning': any(re.search(p, q_lower) for p in cls.REASONING_PATTERNS),
            'has_medical_complex': any(re.search(p, q_lower) for p in cls.MEDICAL_COMPLEX),
            'has_options': bool(re.search(r'[A-D][:.\)]', question)),
            'has_multi_part': '?' in question[:-1],  # Multiple question marks
        }
        
        # Compute complexity score
        score = 0.0
        
        # Word count contribution (0-0.3)
        word_score = min(features['word_count'] / 100, 0.3)
        score += word_score
        
        # Domain complexity (0-0.4)
        if features['has_math']:
            score += 0.15
        if features['has_code']:
            score += 0.15
        if features['has_reasoning']:
            score += 0.2
        if features['has_medical_complex']:
            score += 0.2
        
        # Structure complexity (0-0.3)
        if features['has_multi_part']:
            score += 0.15
        if not features['has_options']:
            # Open-ended questions are harder
            score += 0.1
        
        # Clamp to [0, 1]
        score = min(max(score, 0.0), 1.0)
        
        # Map to steps: linear interpolation
        recommended_steps = int(min_steps + score * (max_steps - min_steps))
        
        return ComplexityEstimate(
            score=score,
            recommended_steps=recommended_steps,
            features=features,
        )


class LatentReasoner:
    """
    Core Latent Reasoning Engine.
    
    Implements the key insight from LatentMAS: reasoning can happen
    in continuous latent space without expensive token decoding.
    
    For 48GB VRAM:
    - Uses full BF16 precision
    - Supports up to 50 latent reasoning steps
    - Batch sizes up to 8 for parallel reasoning
    - Adaptive latent steps based on complexity (NEW)
    """
    
    def __init__(
        self,
        model,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        adaptive_steps: bool = True,
        min_steps: int = 3,
        max_steps: int = 20,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        
        # Adaptive steps settings
        self.adaptive_steps = adaptive_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        
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
    
    def get_adaptive_steps(self, question: str, default_steps: int = 10) -> int:
        """
        Get optimal latent steps for a question based on complexity.
        
        Args:
            question: The input question
            default_steps: Default if adaptive is disabled
            
        Returns:
            Recommended number of latent steps
        """
        if not self.adaptive_steps:
            return default_steps
        
        estimate = ComplexityEstimator.estimate(
            question,
            min_steps=self.min_steps,
            max_steps=self.max_steps,
        )
        
        return estimate.recommended_steps
    
    @torch.no_grad()
    def reason(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_steps: int = 10,
        past_key_values: Optional[Tuple] = None,
        return_all_hidden: bool = False,
        question: Optional[str] = None,  # For adaptive steps
        early_exit_threshold: float = 0.02,  # Exit early if converged
    ) -> LatentReasoningResult:
        """
        Perform multi-step latent reasoning.
        
        This is the core of LatentMAS - each step:
        1. Forward pass to get hidden state
        2. Realign hidden state to input space
        3. Use as input for next step (no token decoding!)
        
        OPTIMIZED: Includes early exit when hidden states converge.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_steps: Number of latent reasoning steps
            past_key_values: Optional KV cache from previous agents
            return_all_hidden: Whether to return all intermediate hidden states
            question: Optional question text for adaptive steps
            early_exit_threshold: Cosine sim threshold for early exit (0 = disabled)
            
        Returns:
            LatentReasoningResult with final hidden state and optional intermediates
        """
        # Apply adaptive steps if question provided
        if question is not None and self.adaptive_steps:
            original_steps = num_steps
            num_steps = self.get_adaptive_steps(question, default_steps=num_steps)
            if num_steps != original_steps:
                logger.debug(f"Adaptive steps: {original_steps} → {num_steps} for complexity-based optimization")
        
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
        prev_hidden = None
        actual_steps = 0
        
        if return_all_hidden:
            all_hidden.append(last_hidden.squeeze(1))
        
        # Latent reasoning steps with early exit
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
            new_hidden = outputs.hidden_states[-1][:, -1:, :]
            actual_steps += 1
            
            if return_all_hidden:
                all_hidden.append(new_hidden.squeeze(1))
            
            # OPTIMIZATION: Early exit if hidden state has converged
            if prev_hidden is not None and early_exit_threshold > 0 and step >= 1:
                cos_sim = F.cosine_similarity(
                    prev_hidden.view(1, -1),
                    new_hidden.view(1, -1),
                    dim=-1
                ).item()
                
                if cos_sim > (1.0 - early_exit_threshold):
                    last_hidden = new_hidden
                    break
            
            prev_hidden = last_hidden
            last_hidden = new_hidden
        
        return LatentReasoningResult(
            final_hidden=last_hidden.squeeze(1),
            all_hidden_states=all_hidden,
            kv_cache=kv_cache,
            num_steps=actual_steps,
        )
    
    @torch.no_grad()
    def reason_from_hidden(
        self,
        hidden_state: torch.Tensor,
        num_steps: int = 5,
        kv_cache: Optional[Tuple] = None,
        early_exit_threshold: float = 0.02,  # Exit early if hidden state converges
    ) -> LatentReasoningResult:
        """
        Continue latent reasoning from a hidden state.
        
        Useful for agent-to-agent latent transfer.
        OPTIMIZED: Includes early exit when hidden states converge.
        
        Args:
            hidden_state: Starting hidden state [B, D] or [B, 1, D]
            num_steps: Number of reasoning steps
            kv_cache: Optional KV cache
            early_exit_threshold: Cosine sim threshold for early exit (0 = disabled)
            
        Returns:
            LatentReasoningResult
        """
        all_hidden = []
        
        # Ensure correct shape
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1)  # [B, 1, D]
        
        current_hidden = hidden_state
        current_cache = kv_cache
        prev_hidden = None
        actual_steps = 0
        
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
            new_hidden = outputs.hidden_states[-1][:, -1:, :]
            actual_steps += 1
            all_hidden.append(new_hidden.squeeze(1))
            
            # OPTIMIZATION: Early exit if hidden state has converged
            if prev_hidden is not None and early_exit_threshold > 0 and step >= 1:
                cos_sim = F.cosine_similarity(
                    prev_hidden.view(1, -1),
                    new_hidden.view(1, -1),
                    dim=-1
                ).item()
                
                if cos_sim > (1.0 - early_exit_threshold):
                    # Hidden state converged, exit early
                    current_hidden = new_hidden
                    break
            
            prev_hidden = current_hidden
            current_hidden = new_hidden
        
        return LatentReasoningResult(
            final_hidden=current_hidden.squeeze(1),
            all_hidden_states=all_hidden,
            kv_cache=current_cache,
            num_steps=actual_steps,  # Report actual steps taken
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


# ========================
# Properties for LatentReasoner (added after class definition for cleaner structure)
# ========================

# Monkey-patch properties onto LatentReasoner
def _get_realignment_matrix(self) -> Optional[torch.Tensor]:
    """Get the realignment matrix (read-only)"""
    return self._realign_matrix

def _get_realignment_enabled(self) -> bool:
    """Check if realignment is enabled"""
    return self._realign_matrix is not None

def _get_hidden_dim(self) -> int:
    """Get hidden dimension"""
    return self.hidden_dim

LatentReasoner.realignment_matrix = property(_get_realignment_matrix)
LatentReasoner.realignment_enabled = property(_get_realignment_enabled)
LatentReasoner.hidden_dimension = property(_get_hidden_dim)
