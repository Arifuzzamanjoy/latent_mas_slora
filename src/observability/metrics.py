"""
Observability metrics for LatentMAS pipelines.

Provides three main constructs:

* **AgentMetrics** – per-agent timing, latent-step count, convergence curve.
* **PipelineMetrics** – aggregated across all agents in one pipeline run.
* **MetricsCollector** – lightweight hook object passed into a pipeline run to
  collect the data above *without* altering pipeline behaviour.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class AgentMetrics:
    """Metrics captured for a single agent during a pipeline run."""

    name: str
    latent_steps_taken: int = 0
    latent_steps_max: int = 0
    converged_at_step: Optional[int] = None
    wall_time_ms: float = 0.0
    kv_cache_size_tokens: int = 0
    hidden_state_norm: float = 0.0
    convergence_curve: List[float] = field(default_factory=list)
    mode: str = ""  # "latent+text" | "latent_only"

    # Extra optional payload the caller may add
    extra: Dict[str, Any] = field(default_factory=dict)

    def converged(self) -> bool:
        return self.converged_at_step is not None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "latent_steps_taken": self.latent_steps_taken,
            "latent_steps_max": self.latent_steps_max,
            "converged": self.converged(),
            "converged_at_step": self.converged_at_step,
            "wall_time_ms": round(self.wall_time_ms, 2),
            "kv_cache_size_tokens": self.kv_cache_size_tokens,
            "hidden_state_norm": round(self.hidden_state_norm, 4),
            "convergence_curve": [round(v, 6) for v in self.convergence_curve],
            "mode": self.mode,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class PipelineMetrics:
    """Aggregated metrics for one full pipeline run."""

    agents: List[AgentMetrics] = field(default_factory=list)
    total_wall_time_ms: float = 0.0
    total_latent_steps: int = 0
    peak_vram_mb: float = 0.0
    answer_confidence: Optional[float] = None

    # Metadata about the run
    question: str = ""
    pipeline_mode: str = ""   # "run" | "run_true_latent"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "pipeline_mode": self.pipeline_mode,
            "total_wall_time_ms": round(self.total_wall_time_ms, 2),
            "total_latent_steps": self.total_latent_steps,
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "answer_confidence": (
                round(self.answer_confidence, 4) if self.answer_confidence is not None else None
            ),
            "agents": [a.to_dict() for a in self.agents],
        }


# ======================================================================
# Collector
# ======================================================================

class MetricsCollector:
    """Hook object threaded through a pipeline run to collect metrics.

    Intended use (inside ``HierarchicalPipeline.run()``):

    .. code-block:: python

        collector = MetricsCollector(question=question)
        for agent_name in agents:
            collector.start_agent(agent_name, latent_steps_max=self.latent_steps)
            # ... latent reasoning ...
            collector.record_latent_result(latent_result)
            # ... text generation ...
            collector.end_agent(kv_cache=latent_result.kv_cache, mode="latent+text")
        pipeline_metrics = collector.finalize()

    All methods are no-ops if the collector is in a bad state, so
    pipeline code can call them unconditionally when a collector is
    provided.
    """

    def __init__(self, question: str = "", pipeline_mode: str = ""):
        self._question = question
        self._pipeline_mode = pipeline_mode

        self._agents: List[AgentMetrics] = []
        self._current: Optional[AgentMetrics] = None
        self._current_start: float = 0.0
        self._global_start: float = time.time()

        # Track hidden states to build convergence curve
        self._prev_hidden: Optional[torch.Tensor] = None

        self._finalized = False
        self._peak_vram: float = 0.0

    # ------------------------------------------------------------------
    # Per-agent hooks
    # ------------------------------------------------------------------

    def start_agent(self, name: str, latent_steps_max: int = 0) -> None:
        """Mark the beginning of an agent's processing."""
        if self._finalized:
            return
        # Flush previous agent if caller forgot end_agent
        if self._current is not None:
            self._flush_current()

        self._current = AgentMetrics(
            name=name,
            latent_steps_max=latent_steps_max,
        )
        self._current_start = time.time()
        self._prev_hidden = None
        self._snapshot_vram()

    def record_latent_step(
        self,
        step: int,
        hidden: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> None:
        """Record one latent reasoning step (called from inside the reasoner loop).

        Parameters
        ----------
        step : int
            Zero-based step index.
        hidden : Tensor
            Current hidden state ``[B, D]`` or ``[B, 1, D]``.
        prev_hidden : Tensor | None
            Previous hidden state (for cosine similarity).
            If *None* the collector uses the last hidden it saw.
        """
        if self._current is None:
            return

        h = hidden.view(1, -1)
        ref = prev_hidden.view(1, -1) if prev_hidden is not None else (
            self._prev_hidden.view(1, -1) if self._prev_hidden is not None else None
        )

        if ref is not None:
            cos = F.cosine_similarity(ref.float(), h.float(), dim=-1).item()
            self._current.convergence_curve.append(cos)

        self._prev_hidden = hidden.detach()
        self._current.latent_steps_taken = step + 1

    def record_latent_result(self, latent_result) -> None:
        """Convenience: extract metrics from a ``LatentReasoningResult``.

        If the result includes ``all_hidden_states`` the convergence
        curve is rebuilt from them; otherwise only step count and
        hidden norm are captured.
        """
        if self._current is None:
            return

        self._current.latent_steps_taken = latent_result.num_steps
        self._current.hidden_state_norm = float(
            latent_result.final_hidden.float().norm().item()
        )

        # Rebuild convergence curve from stored hidden states
        if latent_result.all_hidden_states:
            states = latent_result.all_hidden_states
            for idx in range(1, len(states)):
                prev = states[idx - 1].view(1, -1).float()
                curr = states[idx].view(1, -1).float()
                cos = F.cosine_similarity(prev, curr, dim=-1).item()
                self._current.convergence_curve.append(cos)

            # Detect convergence (same logic as reasoner: cos > 1 - threshold)
            for step_idx, cos_val in enumerate(self._current.convergence_curve):
                if cos_val > 0.98:  # 1 - 0.02 default threshold
                    self._current.converged_at_step = step_idx + 1
                    break

    def end_agent(
        self,
        kv_cache=None,
        mode: str = "",
    ) -> None:
        """Mark the end of an agent's processing."""
        if self._current is None:
            return
        self._current.wall_time_ms = (time.time() - self._current_start) * 1000
        self._current.mode = mode

        if kv_cache is not None:
            from ..core.latent_reasoner import get_cache_length
            self._current.kv_cache_size_tokens = get_cache_length(kv_cache)

        self._snapshot_vram()
        self._flush_current()

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> PipelineMetrics:
        """Produce the final ``PipelineMetrics`` object.

        Safe to call multiple times (returns the same result).
        """
        if self._current is not None:
            self._flush_current()

        if not self._finalized:
            self._finalized = True
            self._snapshot_vram()

        return PipelineMetrics(
            agents=list(self._agents),
            total_wall_time_ms=(time.time() - self._global_start) * 1000,
            total_latent_steps=sum(a.latent_steps_taken for a in self._agents),
            peak_vram_mb=self._peak_vram,
            question=self._question,
            pipeline_mode=self._pipeline_mode,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_current(self) -> None:
        if self._current is not None:
            self._agents.append(self._current)
            self._current = None
            self._prev_hidden = None

    def _snapshot_vram(self) -> None:
        if torch.cuda.is_available():
            try:
                mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                self._peak_vram = max(self._peak_vram, mb)
            except Exception:
                pass
