#!/usr/bin/env python3
"""
Smart Inference - TRUE LatentMAS Implementation

Supports multiple modes:
- true_latent: Original LatentMAS (only final agent generates text) - FASTEST
- fast: 3 agents with text at each step
- normal: 4 agents with full text generation
"""

import sys
sys.path.insert(0, '/workspace/latent_mas_slora')

from src import (
    LatentMASSystem, 
    AgentConfig, 
    Domain, 
    SemanticRouter,
    DOMAIN_AGENTS,
)


# Agent configs for different modes
FAST_AGENTS = {
    Domain.CODE: [
        AgentConfig.planner(max_tokens=300),
        AgentConfig.coder(max_tokens=500),
        AgentConfig.judger(max_tokens=400),
    ],
    Domain.MATH: [
        AgentConfig.planner(max_tokens=300),
        AgentConfig.math(max_tokens=400),
        AgentConfig.judger(max_tokens=400),
    ],
    Domain.MEDICAL: [
        AgentConfig.planner(max_tokens=300),
        AgentConfig.medical(max_tokens=400),
        AgentConfig.judger(max_tokens=400),
    ],
    Domain.REASONING: [
        AgentConfig.planner(max_tokens=300),
        AgentConfig.refiner(max_tokens=400),
        AgentConfig.judger(max_tokens=400),
    ],
    Domain.GENERAL: [
        AgentConfig.planner(max_tokens=300),
        AgentConfig.refiner(max_tokens=400),
        AgentConfig.judger(max_tokens=400),
    ],
}

# TRUE LatentMAS: 4 agents, only judger generates text (larger output)
TRUE_LATENT_AGENTS = {
    Domain.CODE: [
        AgentConfig.planner(max_tokens=100),  # Won't generate text anyway
        AgentConfig.coder(max_tokens=100),
        AgentConfig.critic(max_tokens=100),
        AgentConfig.judger(max_tokens=800),   # Only this generates text
    ],
    Domain.MATH: [
        AgentConfig.planner(max_tokens=100),
        AgentConfig.math(max_tokens=100),
        AgentConfig.critic(max_tokens=100),
        AgentConfig.judger(max_tokens=800),
    ],
    Domain.MEDICAL: [
        AgentConfig.planner(max_tokens=100),
        AgentConfig.medical(max_tokens=100),
        AgentConfig.critic(max_tokens=100),
        AgentConfig.judger(max_tokens=800),
    ],
    Domain.REASONING: [
        AgentConfig.planner(max_tokens=100),
        AgentConfig.critic(max_tokens=100),
        AgentConfig.refiner(max_tokens=100),
        AgentConfig.judger(max_tokens=800),
    ],
    Domain.GENERAL: [
        AgentConfig.planner(max_tokens=100),
        AgentConfig.critic(max_tokens=100),
        AgentConfig.refiner(max_tokens=100),
        AgentConfig.judger(max_tokens=800),
    ],
}


class SmartInference:
    """
    Smart inference with TRUE LatentMAS support.
    
    Modes:
    - true_latent: 4 agents, only final generates text (~20-30s) ⚡ FASTEST
    - fast:        3 agents, text each step (~40-50s)
    - normal:      4 agents, text each step (~90-120s)
    """
    
    MODES = {
        "true_latent": {"latent_steps": 10, "true_latent": True,  "agents": "true_latent"},
        "fast":        {"latent_steps": 5,  "true_latent": False, "agents": "fast"},
        "normal":      {"latent_steps": 15, "true_latent": False, "agents": "normal"},
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        dtype: str = "bfloat16",
        mode: str = "true_latent",
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.mode = mode
        self.mode_config = self.MODES.get(mode, self.MODES["true_latent"])
        
        self.router = SemanticRouter()
        self._systems: dict = {}
        self._current_domain: Domain = None
        
        self._print_mode_info()
    
    def _print_mode_info(self):
        cfg = self.mode_config
        if cfg["true_latent"]:
            print(f"[Mode: {self.mode.upper()}] TRUE LatentMAS - 4 agents, only final generates text")
        else:
            agents = "3" if cfg["agents"] == "fast" else "4"
            print(f"[Mode: {self.mode.upper()}] latent_steps={cfg['latent_steps']}, {agents} agents")
    
    def _get_agents(self, domain: Domain):
        """Get agent configs based on mode"""
        agent_type = self.mode_config["agents"]
        if agent_type == "true_latent":
            return TRUE_LATENT_AGENTS[domain]
        elif agent_type == "fast":
            return FAST_AGENTS[domain]
        else:
            return DOMAIN_AGENTS[domain]
    
    def _get_system(self, domain: Domain) -> LatentMASSystem:
        """Get or create system for domain"""
        cache_key = (domain, self.mode)
        
        if cache_key not in self._systems:
            print(f"\n[Creating {domain.value} pipeline for {self.mode} mode...]")
            
            system = LatentMASSystem(
                model_name=self.model_name,
                dtype=self.dtype,
                latent_steps=self.mode_config["latent_steps"],
            )
            
            for agent_config in self._get_agents(domain):
                system.add_agent(agent_config)
            
            self._systems[cache_key] = system
        
        return self._systems[cache_key]
    
    def set_mode(self, mode: str):
        """Change inference mode"""
        if mode not in self.MODES:
            print(f"Unknown mode. Available: {list(self.MODES.keys())}")
            return
        
        self.mode = mode
        self.mode_config = self.MODES[mode]
        print(f"\n[Switched to {mode.upper()}]")
        self._print_mode_info()
    
    def run(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = 0.4,
        force_domain: Domain = None,
        show_routing: bool = True,
    ):
        """Run inference with automatic routing."""
        if force_domain:
            domain, confidence = force_domain, 1.0
        else:
            domain, confidence = self.router.get_best_domain(prompt)
        
        if show_routing:
            print("\n" + self.router.explain(prompt))
        
        system = self._get_system(domain)
        self._current_domain = domain
        
        # Use TRUE LatentMAS pipeline if enabled
        result = system.run(
            question=prompt,
            pipeline="true_latent" if self.mode_config["true_latent"] else "hierarchical",
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        return result, domain, confidence


def main():
    """Interactive inference"""
    print("=" * 60)
    print("🚀 LatentMAS Smart Inference")
    print("=" * 60)
    print("\nModes:")
    print("  true_latent - 4 agents, ONLY final outputs text (~20-30s) ⚡")
    print("  fast        - 3 agents, text at each step (~40-50s)")
    print("  normal      - 4 agents, full text generation (~90-120s)")
    print("\nCommands:")
    print("  /mode <mode>    - Switch mode")
    print("  /force <domain> - Force domain (code/math/medical/general)")
    print("  /auto           - Auto-routing")
    print("  /quit           - Exit")
    print("-" * 60)
    
    mode = input("\nSelect mode [true_latent/fast/normal] (default: true_latent): ").strip().lower()
    if mode not in ["true_latent", "fast", "normal"]:
        mode = "true_latent"
    
    engine = SmartInference(mode=mode)
    force_domain = None
    
    while True:
        try:
            prompt = input("\n📝 Enter prompt:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        if prompt == "/quit":
            break
        
        if prompt == "/auto":
            force_domain = None
            print("[Auto-routing enabled]")
            continue
        
        if prompt.startswith("/mode"):
            parts = prompt.split()
            if len(parts) > 1:
                engine.set_mode(parts[1])
            else:
                print("Usage: /mode <true_latent|fast|normal>")
            continue
        
        if prompt.startswith("/force"):
            parts = prompt.split()
            if len(parts) > 1:
                domain_map = {
                    "code": Domain.CODE,
                    "math": Domain.MATH, 
                    "medical": Domain.MEDICAL,
                    "general": Domain.GENERAL,
                }
                if parts[1] in domain_map:
                    force_domain = domain_map[parts[1]]
                    print(f"[Forcing: {force_domain.value}]")
                else:
                    print(f"Unknown domain. Use: {list(domain_map.keys())}")
            continue
        
        # Run inference
        try:
            result, domain, conf = engine.run(
                prompt,
                force_domain=force_domain,
            )
            
            print("\n" + "=" * 60)
            print(f"📤 RESPONSE [{domain.value.upper()}]")
            print("=" * 60)
            print(result.final_answer)
            print("-" * 60)
            
            # Performance stats
            tokens = result.total_tokens
            latency_s = result.total_latency_ms / 1000
            tok_per_s = tokens / latency_s if latency_s > 0 else 0
            
            # Show agent breakdown
            if result.metadata.get("mode") == "true_latent":
                latent_agents = result.metadata.get("latent_agents", 0)
                print(f"[TRUE LatentMAS: {latent_agents} latent + 1 text agent]")
            
            print(f"[{tokens} tokens | {latency_s:.1f}s | {tok_per_s:.1f} tok/s]")
            
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
