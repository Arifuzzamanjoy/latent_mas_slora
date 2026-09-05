"""
Configuration objects for the eval pipeline.

Everything the harness does is a function of an EvalConfig, and the config is
written into every result file so a run can be reproduced from its output.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GenSettings:
    """Decoding settings applied uniformly to every method."""

    max_new_tokens: int = 512
    temperature: float = 0.0  # 0.0 -> greedy (deterministic)
    top_p: float = 0.9
    seed: int = 0

    def for_sample(self, k: int) -> "GenSettings":
        """Settings for the k-th self-consistency sample (varies only the seed)."""
        return GenSettings(self.max_new_tokens, self.temperature, self.top_p, self.seed + k)


@dataclass
class EvalConfig:
    # ── model ──
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32 | 4bit
    cache_dir: str = "/home/caches"

    # ── data ──
    dataset: str = "sample"
    split: Optional[str] = None
    fraction: float = 1.0
    limit: Optional[int] = None
    offset: int = 0
    data_seed: int = 0
    shuffle: bool = True
    stratify_by: Optional[str] = "domain"
    min_per_group: int = 1

    # ── methods ──
    methods: List[str] = field(default_factory=lambda: ["baseline-cot"])
    method_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── decoding ──
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    seeds: List[int] = field(default_factory=lambda: [0])
    self_consistency: int = 1

    # ── MAS knobs (defaults for every mas-* method; override per method) ──
    latent_steps: int = 10
    agents: Optional[List[str]] = None
    use_router: bool = True
    adaptive_latent_steps: bool = True
    # Set by run_eval.py when --latent-steps was actually typed. Without it a
    # pinned step count cannot be told apart from the dataclass default, and
    # adaptive stepping would silently override what the user asked for.
    latent_steps_explicit: bool = False
    kv_handoff: bool = True
    prompt_style: str = "reason_first"
    adapter_policy: str = "logo"
    top_k: int = 3
    loras: List[str] = field(default_factory=list)

    # ── protocol ──
    permute_options: str = "none"  # none | cyclic | all
    scoring: str = "generate"  # generate | loglikelihood

    # ── output ──
    out_dir: str = "eval_runs"
    run_name: Optional[str] = None
    resume: bool = True
    save_generations: bool = True
    report_markdown: bool = True
    plots: bool = True
    live_plot: bool = False
    live_every: int = 5
    progress_every: int = 1

    # ── execution ──
    dry_run: bool = False

    # ── analysis ──
    bootstrap: int = 2000
    ci: float = 0.95
    compare_to: Optional[str] = None

    def gen(self, seed: int = 0) -> GenSettings:
        return GenSettings(self.max_new_tokens, self.temperature, self.top_p, seed)

    def args_for(self, method: str) -> Dict[str, Any]:
        """Global MAS defaults merged with per-method overrides from --set."""
        base = {
            "latent_steps": self.latent_steps,
            "agents": self.agents,
            "use_router": self.use_router,
            "adaptive_latent_steps": self.adaptive_latent_steps,
            "kv_handoff": self.kv_handoff,
            "prompt_style": self.prompt_style,
            "adapter_policy": self.adapter_policy,
            "top_k": self.top_k,
            "loras": self.loras,
            "scoring": self.scoring,
        }
        base.update(self.method_args.get(method, {}))
        return base

    def fingerprint(self) -> str:
        """Stable hash of the settings that affect results (not of output paths)."""
        d = asdict(self)
        for k in (
            "out_dir",
            "run_name",
            "resume",
            "save_generations",
            "report_markdown",
            "bootstrap",
            "ci",
            "compare_to",
        ):
            d.pop(k, None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_file(cls, path: str) -> "EvalConfig":
        raw = json.loads(Path(path).read_text())
        raw.pop("fingerprint", None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def parse_fraction(value: str) -> float:
    """
    Accept a fraction or a percentage.

        1.0    0.5    0.01    0.001    0.0001     -> taken as fractions
        50%    1%     0.1%    0.01%               -> divided by 100

    There is no lower bound beyond zero: a fraction that rounds below one item
    still yields one item, so --fraction 0.0001 is a valid one-item smoke test.
    """
    v = str(value).strip()
    if v.endswith("%"):
        f = float(v[:-1]) / 100.0
    else:
        f = float(v)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1] or a percentage; got {value!r}")
    return f


def parse_set_args(pairs: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse --set method.key=value into {method: {key: value}}.

    Values are parsed as JSON when possible, so:
        --set latent-mas.latent_steps=25
        --set latent-mas.agents='["Planner","Judger"]'
        --set text-mas.use_router=false
    """
    out: Dict[str, Dict[str, Any]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects method.key=value, got: {pair}")
        lhs, _, value = pair.partition("=")
        if "." not in lhs:
            raise ValueError(f"--set expects method.key=value, got: {pair}")
        method, _, key = lhs.partition(".")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        out.setdefault(method.strip(), {})[key.strip()] = parsed
    return out
