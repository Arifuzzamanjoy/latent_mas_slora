"""
Tests for src/agents/configs.py — LoRA safety and agent configurations.

Ensures no predefined agent config targets forbidden embedding layers.
"""

import pytest

from src.agents.configs import (
    AgentConfig,
    AgentRole,
    LoRASpec,
    HIERARCHICAL_AGENTS,
    MEDICAL_PIPELINE_AGENTS,
    MATH_PIPELINE_AGENTS,
    CODING_PIPELINE_AGENTS,
)

FORBIDDEN_MODULES = {"embed_tokens", "lm_head"}


class TestLoRASpecExcludesEmbeddings:
    """Default LoRASpec must never target embedding layers."""

    def test_default_spec(self):
        spec = LoRASpec()
        for mod in FORBIDDEN_MODULES:
            assert mod not in spec.target_modules, (
                f"Default LoRASpec should NOT include '{mod}' in target_modules"
            )

    def test_custom_spec_can_set_safe_targets(self):
        spec = LoRASpec(target_modules=["q_proj", "v_proj"])
        for mod in FORBIDDEN_MODULES:
            assert mod not in spec.target_modules


class TestPredefinedAgentsHaveSafeLoRA:
    """Every agent in every predefined pipeline must have safe LoRA targets."""

    ALL_PIPELINES = {
        "HIERARCHICAL": HIERARCHICAL_AGENTS,
        "MEDICAL": MEDICAL_PIPELINE_AGENTS,
        "MATH": MATH_PIPELINE_AGENTS,
        "CODING": CODING_PIPELINE_AGENTS,
    }

    @pytest.mark.parametrize(
        "pipeline_name, agents",
        [
            ("HIERARCHICAL", HIERARCHICAL_AGENTS),
            ("MEDICAL", MEDICAL_PIPELINE_AGENTS),
            ("MATH", MATH_PIPELINE_AGENTS),
            ("CODING", CODING_PIPELINE_AGENTS),
        ],
    )
    def test_no_forbidden_targets(self, pipeline_name, agents):
        for agent in agents:
            targets = set(agent.lora_spec.target_modules)
            dangerous = FORBIDDEN_MODULES & targets
            assert not dangerous, (
                f"[{pipeline_name}] Agent '{agent.name}' targets forbidden "
                f"modules {dangerous}"
            )

    @pytest.mark.parametrize(
        "factory",
        [
            AgentConfig.planner,
            AgentConfig.critic,
            AgentConfig.refiner,
            AgentConfig.judger,
            AgentConfig.medical,
            AgentConfig.math,
            AgentConfig.coder,
        ],
    )
    def test_factory_agents_safe(self, factory):
        agent = factory()
        targets = set(agent.lora_spec.target_modules)
        dangerous = FORBIDDEN_MODULES & targets
        assert not dangerous, (
            f"AgentConfig.{factory.__name__}() targets forbidden modules {dangerous}"
        )


class TestAgentConfigDefaults:
    """Basic sanity checks on AgentConfig."""

    def test_planner_has_system_prompt(self):
        p = AgentConfig.planner()
        assert len(p.system_prompt) > 0
        assert p.role == AgentRole.PLANNER

    def test_adapter_name_auto_set(self):
        cfg = AgentConfig(name="TestAgent", role=AgentRole.CUSTOM, adapter_name="")
        assert cfg.adapter_name == "testagent_lora"

    def test_lora_spec_to_peft_config(self):
        spec = LoRASpec()
        peft_cfg = spec.to_peft_config()
        assert peft_cfg.r == spec.rank
        assert peft_cfg.lora_alpha == spec.alpha
