"""
Agent Pool - Dynamic Management of Multiple Agents

Handles:
- Agent registration and lifecycle
- Dynamic LoRA adapter switching
- Agent activation and deactivation
"""

import threading
from typing import Dict, List, Optional, Any
from .configs import AgentConfig, AgentRole


class AgentPool:
    """
    Pool of agents with dynamic LoRA adapter management.
    
    Enables efficient switching between specialized agents,
    following S-LoRA patterns for scalable serving.
    
    For 48GB VRAM:
    - Can hold 20+ LoRA adapters simultaneously
    - Zero-overhead switching between adapters
    - Supports runtime adapter loading
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self._agents: Dict[str, AgentConfig] = {}
        self._active_agent: Optional[str] = None
        self._adapter_loaded: Dict[str, bool] = {}
        
        self._lock = threading.RLock()
        self._call_count: Dict[str, int] = {}
    
    def register(self, config: AgentConfig, load_adapter: bool = True) -> None:
        """
        Register a new agent with its LoRA adapter.
        
        Args:
            config: Agent configuration
            load_adapter: Whether to load the LoRA adapter immediately
        """
        with self._lock:
            name = config.name
            
            if name in self._agents:
                print(f"[WARN] Agent '{name}' already registered, updating config...")
                self._agents[name] = config
                return
            
            self._agents[name] = config
            self._call_count[name] = 0
            
            if load_adapter:
                self._load_adapter(config)
            
            print(f"[INFO] Registered agent: {name} (role={config.role.value}, rank={config.lora_spec.rank})")
    
    def _load_adapter(self, config: AgentConfig) -> None:
        """Load LoRA adapter for an agent"""
        if config.adapter_name in self._adapter_loaded:
            return
        
        # Check if this is the first adapter
        existing_adapters = getattr(self.model, 'peft_config', {})
        
        if not existing_adapters or len(existing_adapters) == 0:
            # First adapter - need to wrap model
            from peft import get_peft_model
            self.model = get_peft_model(
                self.model,
                config.lora_spec.to_peft_config(),
                adapter_name=config.adapter_name,
            )
        else:
            # Additional adapter
            self.model.add_adapter(
                config.adapter_name,
                config.lora_spec.to_peft_config(),
            )
        
        self._adapter_loaded[config.adapter_name] = True
        
        # Count trainable params
        self.model.set_adapter(config.adapter_name)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[INFO] Loaded adapter '{config.adapter_name}': {trainable:,} trainable params")
    
    def activate(self, agent_name: str) -> AgentConfig:
        """
        Activate an agent by switching to its LoRA adapter.
        
        Args:
            agent_name: Name of the agent to activate
            
        Returns:
            AgentConfig of the activated agent
        """
        with self._lock:
            if agent_name not in self._agents:
                raise ValueError(f"Agent '{agent_name}' not registered. Available: {list(self._agents.keys())}")
            
            config = self._agents[agent_name]
            
            # Ensure adapter is loaded
            if config.adapter_name not in self._adapter_loaded:
                self._load_adapter(config)
            
            # Switch adapter if needed
            if self._active_agent != agent_name:
                self.model.set_adapter(config.adapter_name)
                self._active_agent = agent_name
            
            self._call_count[agent_name] += 1
            return config
    
    def get(self, agent_name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name"""
        return self._agents.get(agent_name)
    
    def get_active(self) -> Optional[AgentConfig]:
        """Get currently active agent"""
        if self._active_agent:
            return self._agents.get(self._active_agent)
        return None
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self._agents.keys())
    
    def list_by_role(self, role: AgentRole) -> List[AgentConfig]:
        """List agents by role"""
        return [cfg for cfg in self._agents.values() if cfg.role == role]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "total_agents": len(self._agents),
            "loaded_adapters": len(self._adapter_loaded),
            "active_agent": self._active_agent,
            "call_counts": dict(self._call_count),
        }
    
    def unload_adapter(self, agent_name: str) -> None:
        """Unload an agent's adapter to free memory"""
        with self._lock:
            if agent_name not in self._agents:
                return
            
            config = self._agents[agent_name]
            
            if config.adapter_name in self._adapter_loaded:
                # Delete adapter from model
                if hasattr(self.model, 'delete_adapter'):
                    self.model.delete_adapter(config.adapter_name)
                    del self._adapter_loaded[config.adapter_name]
                    print(f"[INFO] Unloaded adapter: {config.adapter_name}")
                    
                    # Switch to another adapter if this was active
                    if self._active_agent == agent_name:
                        self._active_agent = None
                        if self._adapter_loaded:
                            other = next(iter(self._adapter_loaded.keys()))
                            for name, cfg in self._agents.items():
                                if cfg.adapter_name == other:
                                    self.activate(name)
                                    break


class AgentExecutor:
    """
    Executor for running agent inference.
    
    Handles:
    - Prompt construction
    - Generation with agent-specific parameters
    - Output parsing
    """
    
    def __init__(
        self,
        pool: AgentPool,
        tokenizer,
        device: str = "cuda",
    ):
        self.pool = pool
        self.tokenizer = tokenizer
        self.device = device
    
    def build_prompt(
        self,
        agent: AgentConfig,
        question: str,
        context: str = "",
        chat_template: bool = True,
    ) -> str:
        """
        Build prompt for an agent.
        
        Args:
            agent: Agent configuration
            question: Input question
            context: Optional context from previous agents
            chat_template: Whether to use chat template
        """
        user_content = agent.user_prompt_template.format(question=question)
        
        if context:
            user_content = f"Context from previous analysis:\n{context}\n\n{user_content}"
        
        if chat_template:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": user_content},
            ]
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback to manual template
                return (
                    f"<|im_start|>system\n{agent.system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
        else:
            return f"{agent.system_prompt}\n\n{user_content}"
