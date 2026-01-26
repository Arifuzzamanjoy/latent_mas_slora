"""
LatentMAS + S-LoRA Multi-Agent Reasoning System

A production-grade implementation combining:
- LatentMAS: Latent-space multi-agent collaboration
- S-LoRA patterns: Scalable LoRA adapter serving
- Hierarchical reasoning: Planner → Critic → Refiner → Judger

Optimized for 24-48GB VRAM with full BF16 precision.
"""

from .system import LatentMASSystem
from .agents.configs import AgentConfig, AgentRole
from .core.latent_memory import LatentMemory
from .lora.adapter_manager import LoRAAdapterManager

__version__ = "0.1.0"
__all__ = [
    "LatentMASSystem",
    "AgentConfig", 
    "AgentRole",
    "LatentMemory",
    "LoRAAdapterManager",
]
