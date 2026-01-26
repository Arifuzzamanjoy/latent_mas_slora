"""
LatentMAS + S-LoRA Multi-Agent Reasoning System

A production-grade implementation combining:
- LatentMAS: Latent-space multi-agent collaboration
- S-LoRA patterns: Scalable LoRA adapter serving
- Hierarchical reasoning: Planner → Critic → Refiner → Judger
- Semantic routing: Auto-select pipeline based on prompt

Optimized for 24-48GB VRAM with full BF16 precision.
"""

from .system import LatentMASSystem, DOMAIN_AGENTS
from .agents.configs import AgentConfig, AgentRole, LoRASpec
from .core.latent_memory import LatentMemory
from .lora.adapter_manager import LoRAAdapterManager
from .routing import Domain, SemanticRouter, auto_route

__version__ = "0.1.0"
__all__ = [
    "LatentMASSystem",
    "AgentConfig", 
    "AgentRole",
    "LoRASpec",
    "LatentMemory",
    "LoRAAdapterManager",
    "Domain",
    "SemanticRouter",
    "auto_route",
    "DOMAIN_AGENTS",
]
