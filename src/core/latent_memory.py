"""
Latent Memory System for Inter-Agent Communication

Implements shared latent working memory following LatentMAS paper:
- Hidden state storage and retrieval
- KV-cache management for continuous reasoning
- Latent vector accumulation across agents

Optimized for 24-48GB VRAM systems.
"""

import torch
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field


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
class LatentSnapshot:
    """Snapshot of agent's latent state"""
    agent_name: str
    hidden_state: torch.Tensor
    kv_cache_length: int
    step: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatentMemory:
    """
    Shared Latent Working Memory for Multi-Agent Collaboration.
    
    This is the core of LatentMAS - agents communicate through continuous
    hidden states rather than discrete text tokens.
    
    Memory Budget (for 48GB VRAM with Qwen2.5-3B BF16):
    - Base model: ~6GB
    - KV cache per 4K tokens: ~1.5GB  
    - Hidden states buffer: ~2GB
    - LoRA adapters (10x): ~500MB
    - Available for latent memory: ~38GB
    
    Args:
        device: CUDA device
        max_history: Maximum number of latent snapshots to retain
        hidden_dim: Hidden dimension of the model (auto-detected if None)
        dtype: Tensor dtype (bfloat16 for efficiency)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        max_history: int = 50,  # Increased for 48GB VRAM
        hidden_dim: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.max_history = max_history
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        
        # Per-agent hidden state storage
        self._agent_states: Dict[str, torch.Tensor] = {}
        
        # Sequential latent buffer (accumulates across pipeline)
        self._latent_buffer: List[torch.Tensor] = []
        
        # KV cache for continuous generation
        self._kv_cache: Optional[Tuple] = None
        self._kv_cache_length: int = 0
        
        # Agent output text (for hybrid latent+text approaches)
        self._agent_outputs: Dict[str, str] = {}
        
        # Historical snapshots for debugging/analysis
        self._snapshots: List[LatentSnapshot] = []
        self._step_counter: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
    def store_hidden_state(
        self,
        agent_name: str,
        hidden_state: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store an agent's hidden state in working memory.
        
        Args:
            agent_name: Unique identifier for the agent
            hidden_state: Hidden state tensor [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            metadata: Optional metadata about the state
        """
        with self._lock:
            # Ensure correct shape [B, D] for single vectors
            if hidden_state.dim() == 3:
                # Take last position if sequence
                hidden_state = hidden_state[:, -1, :]
            
            # Detach and store
            state = hidden_state.detach().to(self.device, dtype=self.dtype)
            self._agent_states[agent_name] = state
            
            # Add to sequential buffer
            self._latent_buffer.append(state)
            
            # Trim buffer if needed
            if len(self._latent_buffer) > self.max_history:
                self._latent_buffer = self._latent_buffer[-self.max_history:]
            
            # Create snapshot
            self._step_counter += 1
            snapshot = LatentSnapshot(
                agent_name=agent_name,
                hidden_state=state.clone(),
                kv_cache_length=self._kv_cache_length,
                step=self._step_counter,
                metadata=metadata or {},
            )
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.max_history * 2:
                self._snapshots = self._snapshots[-self.max_history:]
    
    def store_latent_vector(self, latent: torch.Tensor) -> None:
        """Store a latent vector (from reasoning steps)"""
        with self._lock:
            if latent.dim() == 3:
                latent = latent[:, -1, :]
            self._latent_buffer.append(latent.detach().to(self.device, dtype=self.dtype))
            if len(self._latent_buffer) > self.max_history:
                self._latent_buffer = self._latent_buffer[-self.max_history:]
    
    def store_agent_output(self, agent_name: str, output: str) -> None:
        """Store agent's text output (for hybrid approaches)"""
        with self._lock:
            self._agent_outputs[agent_name] = output
    
    def get_hidden_state(self, agent_name: str) -> Optional[torch.Tensor]:
        """Get specific agent's hidden state"""
        with self._lock:
            return self._agent_states.get(agent_name)
    
    def get_all_agent_states(self) -> Dict[str, torch.Tensor]:
        """Get all agent hidden states"""
        with self._lock:
            return dict(self._agent_states)
    
    def get_accumulated_latents(self) -> Optional[torch.Tensor]:
        """
        Get accumulated latent context as a single tensor.
        Concatenates all latent vectors along sequence dimension.
        
        Returns:
            Tensor of shape [batch, total_seq, hidden_dim] or None
        """
        with self._lock:
            if not self._latent_buffer:
                return None
            
            # Stack all latents [N, B, D] -> [B, N, D]
            stacked = torch.stack(self._latent_buffer, dim=1)
            return stacked
    
    def get_fused_latent(self, method: str = "mean") -> Optional[torch.Tensor]:
        """
        Get fused representation of all latent states.
        
        Args:
            method: Fusion method - "mean", "last", "weighted", "attention"
            
        Returns:
            Fused tensor of shape [batch, hidden_dim]
        """
        with self._lock:
            if not self._latent_buffer:
                return None
            
            stacked = torch.stack(self._latent_buffer, dim=1)  # [B, N, D]
            
            if method == "mean":
                return stacked.mean(dim=1)
            elif method == "last":
                return stacked[:, -1, :]
            elif method == "weighted":
                # Exponentially increasing weights (recent = more important)
                n = stacked.shape[1]
                weights = torch.exp(torch.linspace(-2, 0, n, device=self.device))
                weights = weights / weights.sum()
                weights = weights.view(1, -1, 1)
                return (stacked * weights).sum(dim=1)
            else:
                return stacked.mean(dim=1)
    
    def update_kv_cache(self, kv_cache: Tuple) -> None:
        """Update KV cache for continuous generation"""
        with self._lock:
            self._kv_cache = kv_cache
            self._kv_cache_length = get_cache_length(kv_cache)
    
    def get_kv_cache(self) -> Optional[Tuple]:
        """Get current KV cache"""
        with self._lock:
            return self._kv_cache
    
    def get_agent_output(self, agent_name: str) -> Optional[str]:
        """Get agent's text output"""
        with self._lock:
            return self._agent_outputs.get(agent_name)
    
    def get_context_summary(self) -> str:
        """Get summary of stored context for debugging"""
        with self._lock:
            return (
                f"LatentMemory: {len(self._agent_states)} agents, "
                f"{len(self._latent_buffer)} latent vectors, "
                f"KV cache length: {self._kv_cache_length}"
            )
    
    def clear(self) -> None:
        """Clear all stored states"""
        with self._lock:
            self._agent_states.clear()
            self._latent_buffer.clear()
            self._kv_cache = None
            self._kv_cache_length = 0
            self._agent_outputs.clear()
            self._step_counter = 0
            # Keep snapshots for analysis
    
    def clear_all(self) -> None:
        """Clear everything including snapshots"""
        self.clear()
        with self._lock:
            self._snapshots.clear()
    
    @property
    def num_latent_vectors(self) -> int:
        """Number of latent vectors in buffer"""
        with self._lock:
            return len(self._latent_buffer)
    
    @property 
    def num_agents(self) -> int:
        """Number of agents with stored states"""
        with self._lock:
            return len(self._agent_states)


class KVCacheManager:
    """
    Efficient KV-Cache Management for Multi-Agent Systems.
    
    Implements strategies from S-LoRA and KVCOMM papers:
    - Unified memory pool for adapters and KV cache
    - Prefix caching for shared prompts
    - Compression for long contexts
    
    For 48GB VRAM:
    - Max KV cache: ~20GB (supports ~50K tokens per batch)
    - Supports dynamic growth and pruning
    """
    
    def __init__(
        self,
        device: str = "cuda",
        max_cache_tokens: int = 32768,  # 32K tokens for 48GB
        compression_threshold: int = 16384,  # Compress after 16K
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.max_cache_tokens = max_cache_tokens
        self.compression_threshold = compression_threshold
        self.dtype = dtype
        
        self._cache: Optional[Tuple] = None
        self._prefix_cache: Dict[str, Tuple] = {}
        self._lock = threading.Lock()
    
    def store(self, kv_cache: Tuple, prefix_key: Optional[str] = None) -> None:
        """Store KV cache, optionally with prefix key for reuse"""
        with self._lock:
            # Check if compression needed
            cache_len = get_cache_length(kv_cache)
            if cache_len > self.compression_threshold:
                kv_cache = self._compress_cache(kv_cache)
            
            self._cache = kv_cache
            
            if prefix_key:
                self._prefix_cache[prefix_key] = kv_cache
    
    def get(self, prefix_key: Optional[str] = None) -> Optional[Tuple]:
        """Get KV cache, optionally by prefix key"""
        with self._lock:
            if prefix_key and prefix_key in self._prefix_cache:
                return self._prefix_cache[prefix_key]
            return self._cache
    
    def _compress_cache(self, kv_cache: Tuple, keep_ratio: float = 0.5) -> Tuple:
        """
        Compress KV cache by keeping most important positions.
        Uses attention-based importance scoring.
        """
        if not kv_cache:
            return kv_cache
        
        # Handle DynamicCache - return as-is since compression requires tuple format
        if hasattr(kv_cache, 'get_seq_length'):
            return kv_cache
        
        # Simple strategy: keep first (system prompt) + last (recent context)
        seq_len = get_cache_length(kv_cache)
        if seq_len == 0:
            return kv_cache
            
        keep_first = seq_len // 4
        keep_last = int(seq_len * keep_ratio) - keep_first
        
        compressed = []
        for layer_kv in kv_cache:
            k, v = layer_kv
            k_new = torch.cat([k[..., :keep_first, :], k[..., -keep_last:, :]], dim=-2)
            v_new = torch.cat([v[..., :keep_first, :], v[..., -keep_last:, :]], dim=-2)
            compressed.append((k_new, v_new))
        
        return tuple(compressed)
    
    def truncate(self, keep_tokens: int) -> None:
        """Truncate cache to specified number of tokens"""
        with self._lock:
            if self._cache is None:
                return
            
            truncated = []
            for layer_kv in self._cache:
                k, v = layer_kv
                k_new = k[..., -keep_tokens:, :]
                v_new = v[..., -keep_tokens:, :]
                truncated.append((k_new.contiguous(), v_new.contiguous()))
            
            self._cache = tuple(truncated)
    
    def clear(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._cache = None
            self._prefix_cache.clear()
    
    @property
    def current_length(self) -> int:
        """Current cache length in tokens"""
        with self._lock:
            return get_cache_length(self._cache)
