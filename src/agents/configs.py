"""
Agent Configurations and Role Definitions

Defines specialized agents for the multi-agent reasoning pipeline.
Optimized for 24-48GB VRAM with larger LoRA ranks.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class AgentRole(Enum):
    """Predefined agent roles with specialized behaviors"""
    PLANNER = "planner"
    CRITIC = "critic"
    REFINER = "refiner"
    JUDGER = "judger"
    CODER = "coder"
    MATH = "math"
    MEDICAL = "medical"
    RESEARCHER = "researcher"
    SUMMARIZER = "summarizer"
    CUSTOM = "custom"


@dataclass
class LoRASpec:
    """LoRA adapter specification - optimized for 48GB VRAM"""
    rank: int = 32  # Increased from 16 for better capacity
    alpha: int = 64  # 2x rank
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig"""
        from peft import LoraConfig, TaskType
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias="none",
        )


@dataclass
class AgentConfig:
    """
    Configuration for a specialized agent.
    
    Each agent has:
    - Unique name and role
    - LoRA adapter specification
    - Generation parameters (temperature, max_tokens)
    - System prompt for role-specific behavior
    """
    name: str
    role: AgentRole
    adapter_name: str
    lora_spec: LoRASpec = field(default_factory=LoRASpec)
    temperature: float = 0.7
    max_tokens: int = 512  # Increased for 48GB VRAM
    top_p: float = 0.9
    system_prompt: str = ""
    user_prompt_template: str = ""
    
    def __post_init__(self):
        if not self.adapter_name:
            self.adapter_name = f"{self.name.lower()}_lora"
        if not self.system_prompt:
            self.system_prompt = self._get_default_system_prompt()
        if not self.user_prompt_template:
            self.user_prompt_template = self._get_default_user_template()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on role"""
        prompts = {
            AgentRole.PLANNER: (
                "You are a Planning and Solving Agent. "
                "Break down problems into clear steps and solve them completely. "
                "Always show your work and state the final answer explicitly."
            ),
            AgentRole.CRITIC: (
                "You are a Critic Agent specialized in verification and error detection. "
                "Carefully check all calculations, logic, and reasoning for errors. "
                "If you find mistakes, provide the corrected solution."
            ),
            AgentRole.REFINER: (
                "You are a Refiner Agent specialized in producing accurate final solutions. "
                "Integrate all feedback, fix any errors, and produce a correct answer. "
                "Always state the final answer clearly."
            ),
            AgentRole.JUDGER: (
                "You are a Judger Agent responsible for determining the correct final answer. "
                "Review all reasoning carefully and provide the definitive answer. "
                "Always state your answer in the format: The answer is [ANSWER]"
            ),
            AgentRole.CODER: (
                "You are a Coding Agent specialized in software development. "
                "Write clean, efficient, well-documented code. "
                "Follow best practices and consider edge cases."
            ),
            AgentRole.MATH: (
                "You are a Mathematics Agent specialized in quantitative reasoning. "
                "Solve problems step-by-step with clear mathematical notation. "
                "Verify calculations and consider multiple approaches."
            ),
            AgentRole.MEDICAL: (
                "You are a Medical Reasoning Agent with clinical expertise. "
                "Apply medical knowledge systematically to diagnose and recommend. "
                "Consider differential diagnoses and evidence-based medicine."
            ),
            AgentRole.RESEARCHER: (
                "You are a Research Agent specialized in information gathering. "
                "Synthesize information from multiple sources objectively. "
                "Identify key findings and knowledge gaps."
            ),
            AgentRole.SUMMARIZER: (
                "You are a Summarizer Agent specialized in concise communication. "
                "Distill complex information into clear, actionable summaries. "
                "Preserve essential details while removing redundancy."
            ),
            AgentRole.CUSTOM: "You are a helpful AI assistant.",
        }
        return prompts.get(self.role, prompts[AgentRole.CUSTOM])
    
    def _get_default_user_template(self) -> str:
        """Get default user prompt template based on role"""
        templates = {
            AgentRole.PLANNER: (
                "Analyze and solve this problem step by step:\n\n"
                "{question}\n\n"
                "Show your complete work:\n"
                "1. Identify what is being asked\n"
                "2. Work through the solution step by step\n"
                "3. State the final answer clearly as: The answer is [ANSWER]"
            ),
            AgentRole.CRITIC: (
                "Evaluate this reasoning and check for errors:\n\n"
                "Question: {question}\n\n"
                "Previous Analysis (via latent context)\n\n"
                "Check carefully:\n"
                "1. Are the calculations correct?\n"
                "2. Is the logic sound?\n"
                "3. Provide corrections if any errors are found"
            ),
            AgentRole.REFINER: (
                "Refine the solution based on feedback:\n\n"
                "Question: {question}\n\n"
                "Provide:\n"
                "1. Corrected solution with all errors fixed\n"
                "2. Complete step-by-step work\n"
                "3. Final answer as: The answer is [ANSWER]"
            ),
            AgentRole.JUDGER: (
                "Provide the final answer:\n\n"
                "Question: {question}\n\n"
                "Based on all analysis, determine the correct answer.\n"
                "For multiple choice questions: state 'The answer is [LETTER]'\n"
                "For math/numerical questions: solve completely and state 'The answer is [NUMBER]'\n\n"
                "Show your reasoning and give the final answer."
            ),
            AgentRole.MEDICAL: (
                "Apply medical reasoning:\n\n"
                "{question}\n\n"
                "Consider:\n"
                "1. Key clinical features\n"
                "2. Differential diagnoses\n"
                "3. Most likely diagnosis/answer"
            ),
        }
        default = "Question: {question}\n\nProvide your analysis and answer."
        return templates.get(self.role, default)
    
    @classmethod
    def planner(cls, **kwargs) -> "AgentConfig":
        """Create a Planner agent with optimized settings"""
        defaults = {
            "name": "Planner",
            "role": AgentRole.PLANNER,
            "adapter_name": "planner_lora",
            "lora_spec": LoRASpec(rank=32, alpha=64),
            "temperature": 0.7,
            "max_tokens": 400,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def critic(cls, **kwargs) -> "AgentConfig":
        """Create a Critic agent with optimized settings"""
        defaults = {
            "name": "Critic",
            "role": AgentRole.CRITIC,
            "adapter_name": "critic_lora",
            "lora_spec": LoRASpec(rank=32, alpha=64),
            "temperature": 0.5,  # Lower for focused critique
            "max_tokens": 350,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def refiner(cls, **kwargs) -> "AgentConfig":
        """Create a Refiner agent with optimized settings"""
        defaults = {
            "name": "Refiner",
            "role": AgentRole.REFINER,
            "adapter_name": "refiner_lora",
            "lora_spec": LoRASpec(rank=48, alpha=96),  # Higher for synthesis
            "temperature": 0.6,
            "max_tokens": 450,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def judger(cls, **kwargs) -> "AgentConfig":
        """Create a Judger agent with optimized settings"""
        defaults = {
            "name": "Judger",
            "role": AgentRole.JUDGER,
            "adapter_name": "judger_lora",
            "lora_spec": LoRASpec(rank=64, alpha=128),  # Highest for final decision
            "temperature": 0.2,  # Low for decisive answers
            "max_tokens": 500,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def medical(cls, **kwargs) -> "AgentConfig":
        """Create a Medical expert agent"""
        defaults = {
            "name": "MedicalExpert",
            "role": AgentRole.MEDICAL,
            "adapter_name": "medical_lora",
            "lora_spec": LoRASpec(rank=64, alpha=128),
            "temperature": 0.4,
            "max_tokens": 600,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def math(cls, **kwargs) -> "AgentConfig":
        """Create a Math expert agent"""
        defaults = {
            "name": "MathExpert",
            "role": AgentRole.MATH,
            "adapter_name": "math_lora",
            "lora_spec": LoRASpec(rank=48, alpha=96),
            "temperature": 0.3,
            "max_tokens": 500,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def coder(cls, **kwargs) -> "AgentConfig":
        """Create a Coding expert agent"""
        defaults = {
            "name": "CodeExpert",
            "role": AgentRole.CODER,
            "adapter_name": "coder_lora",
            "lora_spec": LoRASpec(rank=64, alpha=128),
            "temperature": 0.4,
            "max_tokens": 800,
        }
        defaults.update(kwargs)
        return cls(**defaults)


# Pre-defined agent configurations for common pipelines
HIERARCHICAL_AGENTS = [
    AgentConfig.planner(),
    AgentConfig.critic(),
    AgentConfig.refiner(),
    AgentConfig.judger(),
]

MEDICAL_PIPELINE_AGENTS = [
    AgentConfig.planner(),
    AgentConfig.medical(),
    AgentConfig.critic(),
    AgentConfig.judger(),
]

CODING_PIPELINE_AGENTS = [
    AgentConfig.planner(),
    AgentConfig.coder(),
    AgentConfig.critic(),
    AgentConfig.refiner(),
]

MATH_PIPELINE_AGENTS = [
    AgentConfig.planner(),
    AgentConfig.math(),
    AgentConfig.critic(),
    AgentConfig.judger(),
]
