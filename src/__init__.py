"""
LatentMAS + S-LoRA Multi-Agent Reasoning System.

Core exports only.  For RAG, tools, conversations, training,
or routing import the sub-packages directly.
"""

from .system import LatentMASSystem, SystemConfig
from .model_loader import ModelLoader
from .agents.configs import AgentConfig, AgentRole, LoRASpec
from .pipelines.hierarchical import PipelineResult
from .core.embedding_guard import EmbeddingGuard
from .observability import CheckpointDecoder, MetricsCollector

__version__ = "1.0.0"
__all__ = [
    "LatentMASSystem",
    "SystemConfig",
    "ModelLoader",
    "AgentConfig",
    "AgentRole",
    "LoRASpec",
    "PipelineResult",
    "EmbeddingGuard",
    "CheckpointDecoder",
    "MetricsCollector",
]
