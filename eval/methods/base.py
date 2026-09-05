"""
Method interface.

A "method" is one complete way of turning an item into an answer. Methods are
fully independent of each other and of the runner: they are selected by name on
the command line, configured by their own argument dict, and produce one Sample
per call. Self-consistency, seeding, permutation, scoring and statistics are
applied identically to every method by the runner, so any difference in the
numbers comes from the method itself.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..backends import Backend, GenOutput
from ..config import EvalConfig, GenSettings
from ..data import EvalItem
from ..extract import extract_answer


@dataclass
class Sample:
    """One sample from one method on one item."""

    text: str
    pred: str
    extract_rule: str = "raw"
    extract_failed: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Method:
    """Base class. Subclasses implement sample()."""

    name: str = "base"
    backend_kind: str = "hf"  # hf | system | mock | none
    description: str = ""

    def __init__(self, backend: Backend, args: Dict[str, Any], cfg: EvalConfig):
        self.backend = backend
        self.args = args
        self.cfg = cfg

    # -- helpers ------------------------------------------------------------
    def _finish(self, out: GenOutput, item: EvalItem, extra: Optional[Dict] = None) -> Sample:
        ex = extract_answer(out.text, item.task_type, item.num_choices)
        return Sample(
            text=out.text,
            pred=ex.answer,
            extract_rule=ex.rule,
            extract_failed=ex.failed,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=out.latency_ms,
            extra={**(out.extra or {}), **(extra or {})},
        )

    # -- interface ----------------------------------------------------------
    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "backend": self.backend_kind, "args": self.args}

    # -- what this method is scored against ---------------------------------
    def gold_of(self, item: EvalItem) -> str:
        return item.gold

    def task_type_of(self, item: EvalItem) -> str:
        return item.task_type
