"""
LatentMAS evaluation pipeline.

A method-agnostic harness for comparing reasoning configurations on the same
items, with reproducible dataset segmentation and paired statistics.

See docs/EVAL.md for the full settings reference.
"""

from .config import EvalConfig, GenSettings
from .data import EvalItem, load_dataset, segment
from .extract import ExtractResult, extract_answer
from .methods import METHOD_REGISTRY, build_method, list_methods

__all__ = [
    "EvalConfig",
    "GenSettings",
    "EvalItem",
    "load_dataset",
    "segment",
    "extract_answer",
    "ExtractResult",
    "METHOD_REGISTRY",
    "build_method",
    "list_methods",
]
