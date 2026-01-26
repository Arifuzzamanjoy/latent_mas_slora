"""
LatentMAS System - Main Entry Point

The unified interface for the LatentMAS + S-LoRA multi-agent system.
Optimized for 24-48GB VRAM with full BF16 precision.
"""

import os
import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model

from .core.latent_memory import LatentMemory, KVCacheManager
from .core.latent_reasoner import LatentReasoner
from .agents.configs import AgentConfig, AgentRole, HIERARCHICAL_AGENTS
from .agents.agent_pool import AgentPool
from .lora.adapter_manager import LoRAAdapterManager, QWEN25_LORA_REGISTRY
from .pipelines.hierarchical import HierarchicalPipeline, PipelineResult
from .pipelines.sequential import SequentialPipeline


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
        **kwargs,
    ) -> PipelineResult:
        """
        Run multi-agent reasoning on a question.
        
        Args:
            question: Input question
            pipeline: "hierarchical" or "sequential"
            agents: List of agent names (default: all registered)
            max_new_tokens: Override max tokens
            temperature: Override temperature
            self_consistency: Number of samples for self-consistency voting (1 = disabled)
            
        Returns:
            PipelineResult with answer and traces
        """
        if not self._initialized:
            raise RuntimeError("No agents registered. Call add_agent() first.")
        
        # Default to all registered agents
        if agents is None:
            agents = self._pool.list_agents()
        
        if pipeline == "hierarchical":
            if self_consistency > 1:
                return self._pipeline.run_with_self_consistency(
                    question,
                    num_samples=self_consistency,
                    agents=agents,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                return self._pipeline.run(
                    question,
                    agents=agents,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
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
                question,
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
        
        return stats
    
    def clear_memory(self) -> None:
        """Clear latent memory (useful between questions)"""
        if self._memory:
            self._memory.clear()


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
