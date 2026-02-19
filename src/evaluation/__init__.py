"""
Evaluation & Ablation Framework for LatentMAS.

Provides:
- BenchmarkRunner: run full eval loops on HF datasets (GSM8K, MedQA, ARC, HumanEval+)
- AblationStudy:   4-config ablation that validates the project vision
- paired_bootstrap_test: statistical significance between two systems
"""

from .benchmarks import BenchmarkRunner, BenchmarkResult
from .ablation import AblationStudy
from .statistical import paired_bootstrap_test

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "AblationStudy",
    "paired_bootstrap_test",
]
