"""
Observability for LatentMAS pipelines.

Provides:
- CheckpointDecoder: force-decode hidden states to see what agents "would say"
- MetricsCollector / PipelineMetrics: lightweight hookable telemetry
"""

from .checkpoint_decode import CheckpointDecoder
from .metrics import MetricsCollector, PipelineMetrics, AgentMetrics

__all__ = [
    "CheckpointDecoder",
    "MetricsCollector",
    "PipelineMetrics",
    "AgentMetrics",
]
