"""
LatentMASSystem – thin orchestrator.

Wires together model loading, LoRA adapter management, agent pool,
latent reasoning, and pipeline execution.  Nothing else.

For RAG, tools, conversations, routing, or VLM inference see
``system_legacy.py`` (or future plugin modules).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import yaml
from peft import get_peft_model

from .model_loader import ModelLoader
from .core.embedding_guard import EmbeddingGuard
from .core.latent_memory import LatentMemory
from .core.latent_reasoner import LatentReasoner
from .agents.configs import AgentConfig, HIERARCHICAL_AGENTS
from .agents.agent_pool import AgentPool
from .lora.adapter_manager import LoRAAdapterManager
from .pipelines.hierarchical import HierarchicalPipeline, PipelineResult

if TYPE_CHECKING:
    from .observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class SystemConfig:
    """All knobs for a LatentMAS deployment."""

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    cache_dir: str = "/home/caches"

    # Latent reasoning
    latent_steps: int = 15
    latent_realign: bool = True
    early_exit_threshold: float = 0.02
    adaptive_steps: bool = True
    min_steps: int = 3
    max_steps: int = 20

    # LoRA
    max_loaded_adapters: int = 8

    # Pipeline
    pipeline_type: str = "hierarchical"

    # KV cache
    max_kv_cache_tokens: int = 32768

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load from a YAML config file (e.g. ``configs/base.yaml``)."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        model_sec = raw.get("model", {})
        latent_sec = raw.get("latent", {})
        lora_sec = raw.get("lora", {})
        pipe_sec = raw.get("pipeline", {})

        return cls(
            model_name=model_sec.get("name", cls.model_name),
            dtype=model_sec.get("dtype", cls.dtype),
            device=model_sec.get("device", cls.device),
            cache_dir=model_sec.get("cache_dir", cls.cache_dir),
            latent_steps=latent_sec.get("steps", cls.latent_steps),
            latent_realign=latent_sec.get("realign", cls.latent_realign),
            early_exit_threshold=latent_sec.get(
                "early_exit_threshold", cls.early_exit_threshold
            ),
            adaptive_steps=latent_sec.get("adaptive_steps", cls.adaptive_steps),
            min_steps=latent_sec.get("min_steps", cls.min_steps),
            max_steps=latent_sec.get("max_steps", cls.max_steps),
            max_loaded_adapters=lora_sec.get(
                "max_loaded_adapters", cls.max_loaded_adapters
            ),
            pipeline_type=pipe_sec.get("type", cls.pipeline_type),
        )


# ======================================================================
# System
# ======================================================================

class LatentMASSystem:
    """Thin orchestrator – model, agents, pipeline, done.

    Usage::

        system = LatentMASSystem("configs/base.yaml")
        system.add_agent(AgentConfig.planner())
        system.add_agent(AgentConfig.critic())
        system.add_agent(AgentConfig.refiner())
        system.add_agent(AgentConfig.judger())

        result = system.run("What is the capital of France?")
        print(result.final_answer)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
        **overrides: Any,
    ):
        # ---- config ------------------------------------------------
        if config_path and os.path.isfile(config_path):
            self.config = SystemConfig.from_yaml(config_path)
        else:
            self.config = SystemConfig()

        # Allow keyword overrides (model_name="…", latent_steps=20, …)
        if model_name is not None:
            self.config.model_name = model_name
        for key, val in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)

        # ---- model + tokenizer ------------------------------------
        self.model, self.tokenizer = ModelLoader.load(
            self.config.model_name,
            dtype=self.config.dtype,
            device=self.config.device,
            cache_dir=self.config.cache_dir,
        )
        self.device = self.config.device

        # ---- embedding guard (snapshot before any adapter) ---------
        self.guard = EmbeddingGuard(self.model)

        # ---- shared latent memory ----------------------------------
        self.memory = LatentMemory(device=self.device)

        # Components created on first add_agent() (PEFT wrapping
        # replaces self.model with a PeftModel; every component must
        # receive the *wrapped* reference).
        self._reasoner: Optional[LatentReasoner] = None
        self._pool: Optional[AgentPool] = None
        self._lora_manager: Optional[LoRAAdapterManager] = None
        self._pipeline: Optional[HierarchicalPipeline] = None
        self._initialized = False

        logger.info(
            "LatentMASSystem ready  model=%s  device=%s",
            self.config.model_name,
            self.device,
        )

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, config: AgentConfig) -> "LatentMASSystem":
        """Register an agent.  Optionally loads its LoRA adapter."""
        if not self._initialized:
            self._bootstrap(config)
        else:
            self._pool.register(config)
        return self

    def add_default_agents(self) -> "LatentMASSystem":
        """Add the standard Planner → Critic → Refiner → Judger set."""
        for cfg in HIERARCHICAL_AGENTS:
            self.add_agent(cfg)
        return self

    def load_external_lora(self, name: str, hf_path: str, **kwargs) -> bool:
        """Load an external LoRA adapter from HuggingFace Hub."""
        if not self._initialized:
            raise RuntimeError("Add at least one agent before loading external LoRAs")
        return self._lora_manager.load_external_lora(name, hf_path, **kwargs)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        pipeline: str = "hierarchical",
        observe: bool = False,
        agents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        true_latent: bool = False,
        turbo_mode: bool = False,
        self_consistency: int = 1,
        **kwargs: Any,
    ) -> PipelineResult:
        """Run the multi-agent pipeline.

        Parameters
        ----------
        question : str
            The user's question.
        pipeline : str
            ``"hierarchical"`` (default) or ``"true_latent"``.
        observe : bool
            Attach a :class:`MetricsCollector` and embed results
            in ``result.metadata["observability"]``.
        agents : list[str] | None
            Restrict to these agent names (default: all registered).
        true_latent : bool
            Shorthand for ``pipeline="true_latent"``.
        turbo_mode : bool
            Aggressive speed optimisation (may reduce accuracy).
        self_consistency : int
            Majority-vote over *N* samples (1 = disabled).
        """
        if not self._initialized:
            raise RuntimeError(
                "No agents registered.  Call add_agent() first."
            )

        if agents is None:
            agents = self._pool.list_agents()

        collector: Optional["MetricsCollector"] = None
        if observe:
            from .observability.metrics import MetricsCollector
            collector = MetricsCollector(
                question=question,
                pipeline_mode=pipeline,
            )

        common = dict(
            agents=agents,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            metrics_collector=collector,
            **kwargs,
        )

        if true_latent or pipeline == "true_latent":
            return self._pipeline.run_true_latent(
                question, turbo_mode=turbo_mode, **common,
            )

        if self_consistency > 1:
            return self._pipeline.run_with_self_consistency(
                question, num_samples=self_consistency, **common,
            )

        return self._pipeline.run(question, **common)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, benchmark: str, **kwargs: Any):
        """Run an evaluation benchmark.

        Parameters
        ----------
        benchmark : str
            One of ``"gsm8k"``, ``"medqa"``, ``"arc_challenge"``,
            ``"humaneval_plus"``.

        Returns
        -------
        BenchmarkResult
        """
        from .evaluation.benchmarks import BenchmarkRunner

        runner = BenchmarkRunner(self._pipeline, self.tokenizer)
        return runner.evaluate(benchmark, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> Optional[HierarchicalPipeline]:
        return self._pipeline

    @property
    def reasoner(self) -> Optional[LatentReasoner]:
        return self._reasoner

    @property
    def pool(self) -> Optional[AgentPool]:
        return self._pool

    @property
    def lora_manager(self) -> Optional[LoRAAdapterManager]:
        return self._lora_manager

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def clear_memory(self) -> None:
        """Clear latent memory between questions."""
        self.memory.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return basic system statistics."""
        stats: Dict[str, Any] = {
            "model_name": self.config.model_name,
            "device": self.device,
            "dtype": self.config.dtype,
            "latent_steps": self.config.latent_steps,
            "initialized": self._initialized,
        }
        if self._initialized:
            stats["num_agents"] = len(self._pool.list_agents())
            stats["pool_stats"] = self._pool.get_stats()
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bootstrap(self, first: AgentConfig) -> None:
        """Wrap model with PEFT on first agent and wire components."""
        self.model = get_peft_model(
            self.model,
            first.lora_spec.to_peft_config(),
            adapter_name=first.adapter_name,
        )

        # Re-snapshot embeddings AFTER PEFT wrapping
        self.guard = EmbeddingGuard(self.model)

        self._reasoner = LatentReasoner(
            model=self.model,
            device=self.device,
            adaptive_steps=self.config.adaptive_steps,
            min_steps=self.config.min_steps,
            max_steps=self.config.max_steps,
        )

        self._pool = AgentPool(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self._pool._agents[first.name] = first
        self._pool._adapter_loaded[first.adapter_name] = True
        self._pool._call_count[first.name] = 0

        self._lora_manager = LoRAAdapterManager(
            model=self.model,
            device=self.device,
            cache_dir=self.config.cache_dir,
            max_loaded_adapters=self.config.max_loaded_adapters,
        )

        self._pipeline = HierarchicalPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            pool=self._pool,
            memory=self.memory,
            reasoner=self._reasoner,
            device=self.device,
            latent_steps=self.config.latent_steps,
        )

        self._initialized = True
        logger.info("System bootstrapped with first agent: %s", first.name)
