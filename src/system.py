"""
LatentMAS System - Main Entry Point

The unified interface for the LatentMAS + S-LoRA multi-agent system.
Optimized for 24-48GB VRAM with full BF16 precision.

Extended features:
- RAG for document intelligence
- Custom LoRA training pipeline
- Basic tool use
- Conversation continuity
"""

import os
import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model

from .core.latent_memory import LatentMemory, KVCacheManager
from .core.latent_reasoner import LatentReasoner
from .agents.configs import (
    AgentConfig, AgentRole, HIERARCHICAL_AGENTS,
    MEDICAL_PIPELINE_AGENTS, MATH_PIPELINE_AGENTS, CODING_PIPELINE_AGENTS,
)
from .agents.agent_pool import AgentPool
from .lora.adapter_manager import LoRAAdapterManager, QWEN25_LORA_REGISTRY
from .pipelines.hierarchical import HierarchicalPipeline, PipelineResult
from .pipelines.sequential import SequentialPipeline
from .routing import (
    Domain, SemanticRouter, auto_route,
    AdvancedHybridRouter, RoutingResult, get_advanced_router, advanced_route,
)

# New feature imports
from .rag import RAGPipeline, DocumentStore, RAGRetriever
from .tools import ToolRegistry, ToolExecutor, Tool
from .tools.builtin_tools import register_default_tools
from .conversation import ConversationManager, Conversation, SessionStore
from .training import LoRATrainer, TrainingConfig


# Domain to agent pipeline mapping
DOMAIN_AGENTS = {
    Domain.CODE: CODING_PIPELINE_AGENTS,
    Domain.MATH: MATH_PIPELINE_AGENTS,
    Domain.MEDICAL: MEDICAL_PIPELINE_AGENTS,
    Domain.REASONING: HIERARCHICAL_AGENTS,
    Domain.GENERAL: HIERARCHICAL_AGENTS,
}


@dataclass 
class SystemConfig:
    """Configuration for LatentMAS system"""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"
    cache_dir: str = "/home/caches"
    
    # Memory optimization (for 48GB, use BF16)
    dtype: str = "bfloat16"  # "bfloat16", "float16", or "4bit"
    
    # Latent reasoning settings (increased for 48GB)
    latent_steps: int = 15
    latent_realign: bool = True
    
    # KV cache settings
    max_kv_cache_tokens: int = 32768  # 32K for 48GB
    
    # LoRA settings
    max_loaded_adapters: int = 20


class LatentMASSystem:
    """
    Production-Grade LatentMAS + S-LoRA Multi-Agent System.
    
    Combines:
    - LatentMAS: Latent-space multi-agent collaboration
    - S-LoRA: Scalable LoRA adapter serving
    - Hierarchical/Sequential pipelines
    
    Optimized for 24-48GB VRAM:
    - BF16 precision (or 4-bit quantization)
    - 20+ concurrent LoRA adapters
    - 32K token context window
    - Up to 15 latent reasoning steps
    
    Example:
        system = LatentMASSystem(model_name="Qwen/Qwen2.5-3B-Instruct")
        system.add_agent(AgentConfig.planner())
        system.add_agent(AgentConfig.critic())
        system.add_agent(AgentConfig.judger())
        
        result = system.run("What is 2+2?")
        print(result.final_answer)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        cache_dir: str = "/home/caches",
        dtype: str = "bfloat16",
        latent_steps: int = 15,
        latent_realign: bool = True,
        max_loaded_adapters: int = 20,
    ):
        self.config = SystemConfig(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
            latent_steps=latent_steps,
            latent_realign=latent_realign,
            max_loaded_adapters=max_loaded_adapters,
        )
        
        # Validate device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU")
            self.config.device = "cpu"
        
        self.device = self.config.device
        
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"[INFO] Initializing LatentMAS System")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Dtype: {dtype}")
        print(f"[INFO] Latent Steps: {latent_steps}")
        
        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Initialize components (lazy - after first agent added)
        self._memory: Optional[LatentMemory] = None
        self._reasoner: Optional[LatentReasoner] = None
        self._pool: Optional[AgentPool] = None
        self._adapter_manager: Optional[LoRAAdapterManager] = None
        self._pipeline: Optional[HierarchicalPipeline] = None
        
        # New feature components (lazy initialized)
        self._rag_pipeline: Optional[RAGPipeline] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._conversation_manager: Optional[ConversationManager] = None
        self._session_store: Optional[SessionStore] = None
        self._semantic_router: Optional[SemanticRouter] = None
        self._advanced_router: Optional[AdvancedHybridRouter] = None
        self._fast_router = None  # FastKeywordRouter (lazy import)
        
        # Domain routing state
        self._domain_routing_enabled = False
        self._use_advanced_router = False  # Toggle for SOTA routing
        self._use_fast_router = False  # Toggle for ultra-fast routing
        self._current_domain_adapter: Optional[str] = None
        
        self._initialized = False
        self._agent_count = 0
        
        print(f"[INFO] Base model loaded successfully")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load base model with appropriate precision"""
        kwargs = {
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
        }
        
        if self.config.dtype == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **kwargs,
        )
        
        # Move to device if not quantized
        if self.config.dtype != "4bit":
            model = model.to(self.device)
        
        model.eval()
        
        # Enable KV cache
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        
        return model
    
    def _initialize_components(self, first_agent: AgentConfig) -> None:
        """Initialize components with first agent"""
        # Add first LoRA adapter
        self.model = get_peft_model(
            self.model,
            first_agent.lora_spec.to_peft_config(),
            adapter_name=first_agent.adapter_name,
        )
        
        # Initialize memory
        self._memory = LatentMemory(
            device=self.device,
            max_history=50,
        )
        
        # Initialize reasoner
        self._reasoner = LatentReasoner(
            model=self.model,
            device=self.device,
        )
        
        # Initialize pool
        self._pool = AgentPool(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self._pool._agents[first_agent.name] = first_agent
        self._pool._adapter_loaded[first_agent.adapter_name] = True
        self._pool._call_count[first_agent.name] = 0
        
        # Initialize adapter manager
        self._adapter_manager = LoRAAdapterManager(
            model=self.model,
            device=self.device,
            cache_dir=self.config.cache_dir,
            max_loaded_adapters=self.config.max_loaded_adapters,
        )
        
        # Initialize pipeline
        self._pipeline = HierarchicalPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            pool=self._pool,
            memory=self._memory,
            reasoner=self._reasoner,
            device=self.device,
            latent_steps=self.config.latent_steps,
        )
        
        self._initialized = True
        print(f"[INFO] System initialized with first agent: {first_agent.name}")
    
    def add_agent(self, config: AgentConfig) -> "LatentMASSystem":
        """
        Add an agent to the system.
        
        Args:
            config: Agent configuration
            
        Returns:
            self for chaining
        """
        if not self._initialized:
            self._initialize_components(config)
        else:
            self._pool.register(config)
        
        self._agent_count += 1
        return self
    
    def add_default_agents(self) -> "LatentMASSystem":
        """Add the standard hierarchical agents"""
        for config in HIERARCHICAL_AGENTS:
            self.add_agent(config)
        return self
    
    def load_external_lora(
        self,
        name: str,
        hf_path: str,
        **kwargs,
    ) -> bool:
        """
        Load an external LoRA adapter from HuggingFace.
        
        Args:
            name: Local name for the adapter
            hf_path: HuggingFace Hub path
            
        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("Add at least one agent first before loading external LoRAs")
        
        return self._adapter_manager.load_external_lora(name, hf_path, **kwargs)
    
    def load_from_registry(self, registry_name: str) -> bool:
        """Load a pre-defined LoRA from the registry"""
        if not self._initialized:
            raise RuntimeError("Add at least one agent first")
        
        return self._adapter_manager.load_from_registry(registry_name)
    
    def list_registry(self) -> Dict[str, Any]:
        """List available LoRAs in registry"""
        return QWEN25_LORA_REGISTRY
    
    def run(
        self,
        question: str,
        pipeline: str = "hierarchical",
        agents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        self_consistency: int = 1,
        true_latent: bool = False,
        turbo_mode: bool = False,  # Accuracy-first (set True for max speed, may reduce accuracy)
        **kwargs,
    ) -> PipelineResult:
        """
        Run multi-agent reasoning on a question.
        
        Args:
            question: Input question
            pipeline: "hierarchical", "sequential", or "true_latent"
            agents: List of agent names (default: all registered)
            max_new_tokens: Override max tokens
            temperature: Override temperature
            self_consistency: Number of samples for self-consistency voting (1 = disabled)
            true_latent: Use TRUE LatentMAS (only final agent generates text)
            turbo_mode: Enable aggressive speed optimizations (default: False for accuracy)
            
        Returns:
            PipelineResult with answer and traces
        """
        if not self._initialized:
            raise RuntimeError("No agents registered. Call add_agent() first.")
        
        # Default to all registered agents
        if agents is None:
            agents = self._pool.list_agents()
        
        # Step 1: Domain routing (if enabled)
        domain_info = None
        routing_method = None
        if self._domain_routing_enabled:
            try:
                # Priority: fast_router > advanced_router > semantic_router
                if self._use_fast_router and self._fast_router is not None:
                    domain, confidence = self._fast_router.route(question)
                    routing_method = "keyword_fast"
                    domain_info = {
                        "domain": domain,
                        "confidence": confidence,
                        "method": routing_method,
                    }
                elif self._use_advanced_router and self._advanced_router is not None:
                    result = self._advanced_router.route(question)
                    domain = result.domain
                    confidence = result.confidence
                    routing_method = result.method
                    domain_info = {
                        "domain": domain,
                        "confidence": confidence,
                        "method": routing_method,
                        "meta_features": result.meta_features,
                    }
                elif self._semantic_router is not None:
                    domain, confidence = self._semantic_router.get_best_domain(question)
                    routing_method = "semantic"
                    domain_info = {"domain": domain, "confidence": confidence, "method": routing_method}
                else:
                    domain, confidence = Domain.GENERAL, 0.0
                    routing_method = "fallback"
                
                # Load domain-specific LoRA if confidence > threshold
                if confidence > 0.20 and domain != Domain.GENERAL:
                    self._load_domain_adapter(domain)
                    print(f"[ROUTER] Domain: {domain.value} ({routing_method}, confidence: {confidence:.2f})")
            except Exception as e:
                print(f"[WARNING] Domain routing failed: {e}")
        
        # Step 2: RAG context retrieval (if enabled)
        augmented_question = question
        if self._rag_pipeline is not None:
            try:
                retrieval = self._rag_pipeline.retrieve(question, top_k=5)
                if retrieval.chunks:
                    context = "\n\n".join([
                        f"[Context {i+1}]:\n{chunk.text}"
                        for i, chunk in enumerate(retrieval.chunks[:3])
                    ])
                    augmented_question = f"{context}\n\n---\n\nQuestion: {question}"
            except Exception as e:
                print(f"[WARNING] RAG retrieval failed: {e}")
        
        # TRUE LatentMAS mode (with turbo optimizations)
        if true_latent or pipeline == "true_latent":
            result = self._pipeline.run_true_latent(
                augmented_question,
                agents=agents,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                turbo_mode=turbo_mode,
                **kwargs,
            )
            # Add domain routing info to result metadata
            if domain_info:
                result.metadata["domain"] = domain_info["domain"].value
                result.metadata["domain_confidence"] = domain_info["confidence"]
                result.metadata["adapter_used"] = self._current_domain_adapter or "none"
            return result
        
        if pipeline == "hierarchical":
            if self_consistency > 1:
                result = self._pipeline.run_with_self_consistency(
                    augmented_question,
                    num_samples=self_consistency,
                    agents=agents,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                result = self._pipeline.run(
                    augmented_question,
                    agents=agents,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            # Add domain routing info to result metadata
            if domain_info:
                result.metadata["domain"] = domain_info["domain"].value
                result.metadata["domain_confidence"] = domain_info["confidence"]
                result.metadata["adapter_used"] = self._current_domain_adapter or "none"
            return result
        elif pipeline == "sequential":
            seq_pipeline = SequentialPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                pool=self._pool,
                memory=self._memory,
                reasoner=self._reasoner,
                device=self.device,
                latent_steps=self.config.latent_steps,
            )
            return seq_pipeline.run(
                augmented_question,
                agents=agents,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "model_name": self.config.model_name,
            "device": self.device,
            "dtype": self.config.dtype,
            "latent_steps": self.config.latent_steps,
            "num_agents": self._agent_count,
            "initialized": self._initialized,
        }
        
        if self._initialized:
            stats["pool_stats"] = self._pool.get_stats()
            stats["memory_stats"] = self._adapter_manager.get_memory_stats() if self._adapter_manager else {}
            
            # Add new component stats
            if self._rag_pipeline:
                stats["rag_stats"] = self._rag_pipeline.get_stats()
            if self._tool_registry:
                stats["tools"] = self._tool_registry.list_tools()
            if self._conversation_manager:
                stats["conversations"] = len(self._conversation_manager._conversations)
        
        return stats
    
    def clear_memory(self) -> None:
        """Clear latent memory (useful between questions)"""
        if self._memory:
            self._memory.clear()
    
    # ========================
    # Public Properties for Component Access
    # ========================
    
    @property
    def latent_memory(self) -> Optional[LatentMemory]:
        """Access the latent memory system"""
        return self._memory
    
    @property
    def latent_reasoner(self) -> Optional[LatentReasoner]:
        """Access the latent reasoner"""
        return self._reasoner
    
    @property
    def agent_pool(self) -> Optional[AgentPool]:
        """Access the agent pool"""
        return self._pool
    
    @property
    def adapter_manager(self) -> Optional[LoRAAdapterManager]:
        """Access the LoRA adapter manager"""
        return self._adapter_manager
    
    @property
    def pipeline(self) -> Optional[HierarchicalPipeline]:
        """Access the hierarchical pipeline"""
        return self._pipeline
    
    @property
    def semantic_router(self) -> Optional[SemanticRouter]:
        """Access the semantic router (if enabled)"""
        return self._semantic_router
    
    @property
    def advanced_router(self) -> Optional[AdvancedHybridRouter]:
        """Access the advanced hybrid router (if enabled)"""
        return self._advanced_router
    
    @property
    def domain_routing_enabled(self) -> bool:
        """Check if domain routing is enabled"""
        return self._domain_routing_enabled
    
    @property
    def router(self) -> Optional[Union[SemanticRouter, AdvancedHybridRouter]]:
        """Access the active router (advanced or semantic)"""
        if self._use_advanced_router and self._advanced_router is not None:
            return self._advanced_router
        return self._semantic_router
    
    def explain_routing(self, prompt: str) -> str:
        """Get detailed routing explanation for a prompt"""
        if self._use_advanced_router and self._advanced_router is not None:
            return self._advanced_router.explain(prompt)
        elif self._semantic_router is not None:
            return self._semantic_router.explain(prompt)
        else:
            return "Domain routing not enabled"
    
    # ========================
    # RAG Integration Methods
    # ========================
    
    def enable_rag(
        self,
        chunk_size: int = 512,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> RAGPipeline:
        """
        Enable RAG (Retrieval-Augmented Generation) capabilities.
        
        Args:
            chunk_size: Size of document chunks
            embedding_model: Sentence transformer model for embeddings
            top_k: Default number of documents to retrieve
            
        Returns:
            RAGPipeline instance
        """
        self._rag_pipeline = RAGPipeline(
            system=self,
            chunk_size=chunk_size,
            embedding_model=embedding_model,
            top_k=top_k,
        )
        print("[INFO] RAG capabilities enabled")
        return self._rag_pipeline
    
    def enable_domain_routing(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        auto_load_adapters: bool = False,
        use_advanced_router: bool = True,
        use_fast_router: bool = False,
    ) -> Union[SemanticRouter, AdvancedHybridRouter, "FastKeywordRouter"]:  # noqa: F821
        """
        Enable domain-based routing for intelligent LoRA selection.
        
        Automatically routes queries to domain-specific LoRAs:
        - Medical queries → medical_reasoner LoRA
        - Math queries → math_instruct LoRA  
        - Code queries → coder_7b LoRA
        - Finance queries → (none, uses base model)
        
        Args:
            embedding_model: Model for semantic similarity
            auto_load_adapters: Preload domain LoRAs from registry
            use_advanced_router: Use SOTA AdvancedHybridRouter (recommended)
            use_fast_router: Use ultra-fast keyword router (~20μs, no ML)
            
        Returns:
            Router instance (SemanticRouter, AdvancedHybridRouter, or FastKeywordRouter)
        """
        if not self._initialized:
            raise RuntimeError("Add agents first before enabling domain routing")
        
        self._use_advanced_router = use_advanced_router
        self._use_fast_router = use_fast_router
        
        if use_fast_router:
            from .routing.fast_router import FastRouter
            self._fast_router = FastRouter()
            print("[INFO] Fast Keyword Router enabled (~20μs, 50k qps)")
        elif use_advanced_router:
            self._advanced_router = AdvancedHybridRouter(
                model_name=embedding_model,
                use_mlp=True,  # Enable MLP (will fallback to hybrid if not trained)
                cache_embeddings=True,
                cache_size=1000,
            )
            print("[INFO] Advanced Hybrid Router enabled (SOTA)")
        else:
            self._semantic_router = SemanticRouter(
                model_name=embedding_model,
                use_embeddings=True,
            )
            print("[INFO] Basic Semantic Router enabled")
        
        self._domain_routing_enabled = True
        
        # Optionally preload domain adapters
        if auto_load_adapters:
            print("[INFO] Preloading domain-specific LoRAs...")
            self._preload_domain_adapters()
        
        return self._advanced_router if use_advanced_router else self._semantic_router
    
    def _get_model_compatible_adapter(self, domain: Domain) -> Optional[str]:
        """
        Get the appropriate adapter for a domain based on the base model size.
        
        This ensures we load adapters that match the current model's architecture.
        For example, medical_reasoner is for 3B, medical_instruct is for 7B.
        """
        model_name = self.config.model_name.lower()
        
        # Domain-specific adapter mappings by model size
        if "7b" in model_name or "7B" in self.config.model_name:
            # 7B model adapters
            adapters = {
                Domain.MEDICAL: "medical_instruct",  # 7B compatible
                Domain.MATH: "math_instruct",
                Domain.CODE: "coder_7b",
                Domain.FINANCE: None,  # No finance-specific 7B adapter yet
                Domain.REASONING: "reasoning_lora",
            }
        else:
            # 3B or smaller model adapters
            adapters = {
                Domain.MEDICAL: "medical_reasoner",  # 3B compatible
                Domain.MATH: "math_basic",
                Domain.CODE: "coder_7b",
                Domain.FINANCE: None,  # No finance-specific 3B adapter yet, uses base model
                Domain.REASONING: None,  # Uses base model reasoning
            }
        
        return adapters.get(domain)
    
    def _preload_domain_adapters(self) -> None:
        """Preload common domain-specific LoRAs from registry"""
        domains_to_load = [Domain.MEDICAL, Domain.MATH, Domain.CODE]
        
        for domain in domains_to_load:
            adapter_name = self._get_model_compatible_adapter(domain)
            if adapter_name and adapter_name in QWEN25_LORA_REGISTRY:
                try:
                    self.load_from_registry(adapter_name)
                    print(f"  ✓ Loaded {domain.value} adapter: {adapter_name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {adapter_name}: {e}")
    
    def _load_domain_adapter(self, domain: Domain) -> bool:
        """Load appropriate LoRA adapter for a domain"""
        adapter_name = self._get_model_compatible_adapter(domain)
        if not adapter_name:
            return False
        
        # Check if already loaded
        if self._current_domain_adapter == adapter_name:
            return True
        
        # Try to load from registry if not already loaded
        if adapter_name in QWEN25_LORA_REGISTRY:
            try:
                if not self._adapter_manager.is_loaded(adapter_name):
                    self.load_from_registry(adapter_name)
                
                # Switch to domain adapter
                self.model.set_adapter(adapter_name)
                self._current_domain_adapter = adapter_name
                return True
            except Exception as e:
                print(f"[WARNING] Could not load domain adapter {adapter_name}: {e}")
                return False
        
        return False
    
    def load_documents(
        self,
        source: Union[str, List[str]],
        **kwargs,
    ) -> int:
        """
        Load documents for RAG.
        
        Args:
            source: File path, directory, or list of paths
            **kwargs: Additional arguments for document loading
            
        Returns:
            Number of documents loaded
        """
        if self._rag_pipeline is None:
            self.enable_rag()
        
        return self._rag_pipeline.load_documents(source, **kwargs)
    
    def query_with_rag(
        self,
        question: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Run a RAG-enhanced query.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for system.run()
            
        Returns:
            RAGResult with answer and sources
        """
        if self._rag_pipeline is None:
            raise RuntimeError("RAG not enabled. Call enable_rag() first.")
        
        return self._rag_pipeline.query(question, top_k=top_k, **kwargs)
    
    @property
    def rag(self) -> Optional[RAGPipeline]:
        """Access the RAG pipeline"""
        return self._rag_pipeline
    
    # ========================
    # Tool Use Methods
    # ========================
    
    def enable_tools(self, register_defaults: bool = True) -> ToolRegistry:
        """
        Enable tool use capabilities.
        
        Args:
            register_defaults: Whether to register default tools
            
        Returns:
            ToolRegistry instance
        """
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(self._tool_registry)
        
        if register_defaults:
            register_default_tools(self._tool_registry)
        
        print(f"[INFO] Tool use enabled with {len(self._tool_registry.list_tools())} tools")
        return self._tool_registry
    
    def register_tool(self, tool: Tool) -> None:
        """Register a custom tool"""
        if self._tool_registry is None:
            self.enable_tools(register_defaults=False)
        
        self._tool_registry.register(tool)
    
    def run_with_tools(
        self,
        question: str,
        max_tool_calls: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run a query with tool use support.
        
        Uses ReAct-style reasoning with tool calls.
        
        Args:
            question: User question
            max_tool_calls: Maximum tool calls per query
            **kwargs: Additional arguments for system.run()
            
        Returns:
            Dict with answer and tool call history
        """
        if self._tool_executor is None:
            self.enable_tools()
        
        from .tools.executor import ReActExecutor
        
        react_executor = ReActExecutor(
            registry=self._tool_registry,
            max_iterations=max_tool_calls,
        )
        
        return react_executor.execute_react_step(
            model=self.model,
            tokenizer=self.tokenizer,
            question=question,
            max_steps=max_tool_calls,
        )
    
    @property
    def tools(self) -> Optional[ToolRegistry]:
        """Access the tool registry"""
        return self._tool_registry
    
    # ========================
    # Conversation Continuity
    # ========================
    
    def enable_conversations(
        self,
        session_path: Optional[str] = None,
        default_system_prompt: Optional[str] = None,
    ) -> ConversationManager:
        """
        Enable multi-turn conversation support.
        
        Args:
            session_path: Path for session persistence
            default_system_prompt: Default system prompt for conversations
            
        Returns:
            ConversationManager instance
        """
        self._conversation_manager = ConversationManager(
            system=self,
            default_system_prompt=default_system_prompt,
        )
        
        if session_path:
            self._session_store = SessionStore(storage_path=session_path)
        
        print("[INFO] Conversation continuity enabled")
        return self._conversation_manager
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a message in a conversation.
        
        Maintains conversation history across turns.
        
        Args:
            message: User message
            conversation_id: Specific conversation (creates new if None)
            **kwargs: Additional arguments for system.run()
            
        Returns:
            Assistant response
        """
        if self._conversation_manager is None:
            self.enable_conversations()
        
        return self._conversation_manager.chat(message, conversation_id, **kwargs)
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        if self._conversation_manager is None:
            return None
        return self._conversation_manager.get_conversation(conversation_id)
    
    def new_conversation(
        self,
        system_prompt: Optional[str] = None,
        **metadata,
    ) -> Conversation:
        """Create a new conversation"""
        if self._conversation_manager is None:
            self.enable_conversations()
        
        return self._conversation_manager.create_conversation(
            system_prompt=system_prompt,
            **metadata,
        )
    
    @property
    def conversations(self) -> Optional[ConversationManager]:
        """Access the conversation manager"""
        return self._conversation_manager
    
    # ========================
    # Training Methods
    # ========================
    
    def create_trainer(self) -> LoRATrainer:
        """
        Create a LoRA trainer for custom adapter training.
        
        Returns:
            LoRATrainer instance
        """
        # Use base model for training (not PEFT wrapped)
        base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        
        return LoRATrainer(
            model=base_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
    
    def train_adapter(
        self,
        train_data: str,
        adapter_name: str,
        config: Optional[TrainingConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Train a custom LoRA adapter.
        
        Args:
            train_data: Path to training data (JSON/JSONL)
            adapter_name: Name for the new adapter
            config: Training configuration
            **kwargs: Override config parameters
            
        Returns:
            TrainingResult with adapter path
        """
        from .training import TrainingDataset
        
        config = config or TrainingConfig(**kwargs)
        config.output_dir = config.output_dir or f"./adapters/{adapter_name}"
        
        trainer = self.create_trainer()
        
        dataset = TrainingDataset.from_json(
            train_data,
            self.tokenizer,
            max_length=config.max_length,
        )
        
        return trainer.train(dataset, config)


# Convenience function
def create_system(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    with_default_agents: bool = True,
    **kwargs,
) -> LatentMASSystem:
    """
    Create a LatentMAS system with sensible defaults.
    
    Args:
        model_name: Base model to use
        with_default_agents: Whether to add hierarchical agents
        **kwargs: Additional arguments for LatentMASSystem
        
    Returns:
        Configured LatentMASSystem
    """
    system = LatentMASSystem(model_name=model_name, **kwargs)
    
    if with_default_agents:
        system.add_default_agents()
    
    return system
