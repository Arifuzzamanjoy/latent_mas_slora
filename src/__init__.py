"""
LatentMAS + S-LoRA Multi-Agent Reasoning System

A production-grade implementation combining:
- LatentMAS: Latent-space multi-agent collaboration
- S-LoRA patterns: Scalable LoRA adapter serving
- Hierarchical reasoning: Planner → Critic → Refiner → Judger
- Semantic routing: Auto-select pipeline based on prompt

Extended features:
- RAG: Document intelligence and retrieval-augmented generation
- Training: Custom LoRA adapter training pipeline
- Tools: Basic tool use for agent actions
- Conversations: Multi-turn conversation continuity

Optimized for 24-48GB VRAM with full BF16 precision.
"""

from .system import LatentMASSystem, DOMAIN_AGENTS
from .agents.configs import AgentConfig, AgentRole, LoRASpec
from .core.latent_memory import LatentMemory
from .lora.adapter_manager import LoRAAdapterManager
from .routing import Domain, SemanticRouter, auto_route

# RAG module
from .rag import (
    RAGPipeline,
    DocumentStore,
    Document,
    DocumentChunk,
    RAGRetriever,
    RetrievalResult,
)

# Training module
from .training import (
    LoRATrainer,
    TrainingConfig,
    TrainingResult,
    TrainingDataset,
    prepare_dataset,
    evaluate_lora,
)

# Tools module
from .tools import (
    ToolRegistry,
    Tool,
    ToolResult,
    ToolExecutor,
    CalculatorTool,
    PythonExecutorTool,
    SearchTool,
    FileReaderTool,
    WebFetchTool,
)

# Conversation module
from .conversation import (
    ConversationManager,
    Conversation,
    Message,
    MessageRole,
    ContextWindow,
    ContextStrategy,
    SessionStore,
    Session,
)

__version__ = "0.2.0"
__all__ = [
    # Core system
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
    
    # RAG
    "RAGPipeline",
    "DocumentStore",
    "Document",
    "DocumentChunk",
    "RAGRetriever",
    "RetrievalResult",
    
    # Training
    "LoRATrainer",
    "TrainingConfig",
    "TrainingResult",
    "TrainingDataset",
    "prepare_dataset",
    "evaluate_lora",
    
    # Tools
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "ToolExecutor",
    "CalculatorTool",
    "PythonExecutorTool",
    "SearchTool",
    "FileReaderTool",
    "WebFetchTool",
    
    # Conversations
    "ConversationManager",
    "Conversation",
    "Message",
    "MessageRole",
    "ContextWindow",
    "ContextStrategy",
    "SessionStore",
    "Session",
]
