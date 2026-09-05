"""
Method registry.

Add a method by writing a Method subclass and listing it here; it becomes
selectable as --methods <name> with no other changes.
"""

from typing import Any, Dict, List

from ..backends import Backend
from ..config import EvalConfig
from .base import Method, Sample
from .baselines import (
    CoTBaseline,
    DirectBaseline,
    JudgerBaseline,
    LogLikelihoodBaseline,
)
from .mas import LatentKVMAS, LatentMAS, LatentMASPaper, RouterOnly, SequentialMAS, TextMAS
from .multilora import MultiLoRA

_ALL = [
    DirectBaseline,
    CoTBaseline,
    JudgerBaseline,
    LogLikelihoodBaseline,
    TextMAS,
    LatentMAS,
    LatentKVMAS,
    LatentMASPaper,
    SequentialMAS,
    MultiLoRA,
    RouterOnly,
]

METHOD_REGISTRY: Dict[str, type] = {cls.name: cls for cls in _ALL}

# Convenience groups usable anywhere a method name is accepted.
METHOD_GROUPS: Dict[str, List[str]] = {
    "all": [c.name for c in _ALL],
    "baselines": ["baseline-direct", "baseline-cot", "baseline-judger", "baseline-loglik"],
    "mas": ["text-mas", "latent-mas", "latent-mas-kv", "latent-mas-paper", "sequential-mas"],
    "latent": ["latent-mas", "latent-mas-kv", "latent-mas-paper"],
    "core": ["baseline-cot", "baseline-judger", "text-mas", "latent-mas", "latent-mas-kv"],
    # each rung changes exactly one thing from the rung below
    "ladder": [
        "baseline-cot",
        "baseline-judger",
        "latent-mas",
        "latent-mas-kv",
        "latent-mas-paper",
    ],
    # the recommended architecture against the two controls that matter
    "recommended": ["baseline-cot", "latent-mas-paper", "multi-lora"],
}


def expand_methods(names: List[str]) -> List[str]:
    """Expand group names, drop duplicates, preserve order."""
    out: List[str] = []
    for n in names:
        for m in METHOD_GROUPS.get(n.strip(), [n.strip()]):
            if m not in out:
                out.append(m)
    unknown = [m for m in out if m not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown method(s): {', '.join(unknown)}. "
            f"Available: {', '.join(METHOD_REGISTRY)} "
            f"(groups: {', '.join(METHOD_GROUPS)})"
        )
    return out


def build_method(name: str, backend: Backend, cfg: EvalConfig) -> Method:
    """
    Resolve a method's arguments.

    Precedence: global config defaults < the method class's own defaults <
    explicit --set overrides. The middle layer is what lets latent-mas and
    latent-mas-paper differ by construction while still honouring --set.
    """
    cls = METHOD_REGISTRY[name]
    args = dict(cfg.args_for(name))
    explicit = set(cfg.method_args.get(name, {}))
    for k, v in getattr(cls, "defaults", {}).items():
        if k not in explicit:
            args[k] = v
    return cls(backend, args, cfg)


def backend_for(name: str) -> str:
    return METHOD_REGISTRY[name].backend_kind


def list_methods() -> List[Dict[str, Any]]:
    return [{"name": c.name, "backend": c.backend_kind, "description": c.description} for c in _ALL]


__all__ = [
    "Method",
    "Sample",
    "METHOD_REGISTRY",
    "METHOD_GROUPS",
    "expand_methods",
    "build_method",
    "backend_for",
    "list_methods",
]
