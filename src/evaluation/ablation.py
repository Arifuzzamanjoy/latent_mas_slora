"""
Ablation Study – the 4-config experiment that validates the LatentMAS vision.

Configurations
--------------
A  Single model     – no agents, standard ``model.generate()``
B  LatentMAS vanilla – 4 agents, latent communication, NO LoRA
C  LatentMAS + same  – 4 agents, SAME adapter on every agent
D  LatentMAS + spec  – 4 agents, DIFFERENT specialised adapters

Running all four on the same benchmarks with the same seed produces a
controlled comparison.  A paired bootstrap test on B-vs-D answers the core
question: *does specialised LoRA help latent multi-agent reasoning?*
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    """Results for one ablation configuration on one benchmark."""
    config_name: str
    dataset_name: str
    accuracy: float
    num_correct: int
    num_total: int
    avg_tokens: float
    avg_latency_ms: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    correctness_vector: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Aggregated results for the full ablation study."""
    configs: Dict[str, Dict[str, ConfigResult]] = field(default_factory=dict)
    significance: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._serialisable(), f, indent=2, default=str)
        logger.info("Ablation results saved → %s", p)

    def _serialisable(self) -> dict:
        out: dict = {"wall_time_s": self.wall_time_s, "metadata": self.metadata}
        configs_ser: dict = {}
        for cfg_name, bench_dict in self.configs.items():
            configs_ser[cfg_name] = {
                bname: asdict(cr) for bname, cr in bench_dict.items()
            }
        out["configs"] = configs_ser
        out["significance"] = self.significance
        return out

    # ------------------------------------------------------------------
    def comparison_table(self) -> str:
        """Return a pretty ASCII table comparing all configs."""
        lines: list[str] = []
        header = f"{'Config':<28} {'Benchmark':<18} {'Acc %':>7}  {'95% CI':>15}  {'Tok':>6}  {'ms':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for cfg_name in sorted(self.configs):
            for bname in sorted(self.configs[cfg_name]):
                cr = self.configs[cfg_name][bname]
                ci_str = f"[{100*cr.ci_lower:.1f}, {100*cr.ci_upper:.1f}]"
                lines.append(
                    f"{cfg_name:<28} {bname:<18} {100*cr.accuracy:>6.2f}  "
                    f"{ci_str:>15}  {cr.avg_tokens:>6.1f}  {cr.avg_latency_ms:>7.0f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """Run the 4-config ablation that validates the LatentMAS vision.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    device : str
        Torch device.
    cache_dir : str
        HF cache directory.
    dtype : str
        ``"bfloat16"`` | ``"float16"`` | ``"4bit"``.
    lora_configs : dict | None
        Mapping ``{"planner": "hf/path", "critic": "hf/path", …}``
        used in Config D.  If ``None`` the study will use untrained
        random-init LoRA (still structurally valid for framework testing).
    shared_lora_path : str | None
        Single adapter path used in Config C.
    latent_steps : int
        Latent reasoning steps for multi-agent configs.
    """

    CONFIGS = ("A_single_model", "B_latentmas_nolora", "C_latentmas_samelora", "D_latentmas_speclora")

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        cache_dir: str = "/home/caches",
        dtype: str = "bfloat16",
        lora_configs: Optional[Dict[str, str]] = None,
        shared_lora_path: Optional[str] = None,
        latent_steps: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.lora_configs = lora_configs or {}
        self.shared_lora_path = shared_lora_path
        self.latent_steps = latent_steps

        # Lazily initialised per config
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _ensure_base_model(self):
        """Load the base model + tokenizer once, reuse across configs."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading base model: %s", self.model_name)

        tok_kwargs: dict = {"cache_dir": self.cache_dir, "trust_remote_code": True}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict = {"cache_dir": self.cache_dir, "trust_remote_code": True}
        if self.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.dtype not in ("4bit",):
            self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("Base model loaded on %s  dtype=%s", self.device, self.dtype)

    def _unload_adapters(self):
        """Remove any PEFT adapters so the base weights are clean."""
        from peft import PeftModel
        if isinstance(self._model, PeftModel):
            self._model = self._model.unload()
            self._model.eval()
            logger.info("Adapters unloaded – back to base model")

    # ------------------------------------------------------------------
    # Config setup
    # ------------------------------------------------------------------

    def setup_config(self, config_name: str):
        """Set up one of the 4 configurations and return a callable pipeline.

        Returns
        -------
        pipeline : callable
            ``pipeline(question) → str``
        """
        self._ensure_base_model()
        self._unload_adapters()
        torch.cuda.empty_cache()

        builder = {
            "A_single_model": self._setup_A,
            "B_latentmas_nolora": self._setup_B,
            "C_latentmas_samelora": self._setup_C,
            "D_latentmas_speclora": self._setup_D,
        }
        if config_name not in builder:
            raise ValueError(f"Unknown config '{config_name}'. Choose from {self.CONFIGS}")

        pipeline = builder[config_name]()
        logger.info("Config '%s' ready", config_name)
        return pipeline

    # ---- Config A -------------------------------------------------------
    def _setup_A(self):
        """Single model, no agents – vanilla generate."""
        model = self._model
        tok = self._tokenizer

        @torch.no_grad()
        def _pipeline(question: str, **kw) -> str:
            prompt = f"Question: {question}\n\nAnswer:"
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
            ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            out = model.generate(
                ids, attention_mask=attn,
                max_new_tokens=kw.get("max_new_tokens", 512),
                do_sample=False, pad_token_id=tok.pad_token_id,
            )
            return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        return _pipeline

    # ---- Config B -------------------------------------------------------
    def _setup_B(self):
        """LatentMAS 4-agent pipeline, NO LoRA adapters."""
        from ..system import LatentMASSystem
        from ..agents.configs import AgentConfig, LoRASpec

        system = self._build_mas_system()
        # Agents with minimal (rank=1) random LoRA – structurally required by
        # PeftModel but effectively a no-op at rank 1 / alpha 1.
        noop_spec = LoRASpec(rank=1, alpha=1, dropout=0.0)
        system.add_agent(AgentConfig.planner(lora_spec=noop_spec))
        system.add_agent(AgentConfig.critic(lora_spec=noop_spec))
        system.add_agent(AgentConfig.refiner(lora_spec=noop_spec))
        system.add_agent(AgentConfig.judger(lora_spec=noop_spec))
        return system

    # ---- Config C -------------------------------------------------------
    def _setup_C(self):
        """LatentMAS + the SAME LoRA adapter on every agent."""
        from ..system import LatentMASSystem
        from ..agents.configs import AgentConfig

        system = self._build_mas_system()
        system.add_agent(AgentConfig.planner())
        system.add_agent(AgentConfig.critic())
        system.add_agent(AgentConfig.refiner())
        system.add_agent(AgentConfig.judger())

        if self.shared_lora_path:
            for name in ("planner_lora", "critic_lora", "refiner_lora", "judger_lora"):
                try:
                    system.load_external_lora(name, self.shared_lora_path)
                    logger.info("Config C: loaded shared adapter → %s", name)
                except Exception as exc:
                    logger.warning("Config C: failed to load shared adapter for %s: %s", name, exc)
        return system

    # ---- Config D -------------------------------------------------------
    def _setup_D(self):
        """LatentMAS + DIFFERENT specialised LoRA adapters."""
        from ..system import LatentMASSystem
        from ..agents.configs import AgentConfig

        system = self._build_mas_system()
        system.add_agent(AgentConfig.planner())
        system.add_agent(AgentConfig.critic())
        system.add_agent(AgentConfig.refiner())
        system.add_agent(AgentConfig.judger())

        role_to_adapter = {
            "planner": "planner_lora",
            "critic": "critic_lora",
            "refiner": "refiner_lora",
            "judger": "judger_lora",
        }
        for role, adapter_name in role_to_adapter.items():
            path = self.lora_configs.get(role)
            if path:
                try:
                    system.load_external_lora(adapter_name, path)
                    logger.info("Config D: loaded %s → %s", path, adapter_name)
                except Exception as exc:
                    logger.warning("Config D: failed to load adapter for %s: %s", role, exc)
        return system

    # ---- Shared builder -------------------------------------------------
    def _build_mas_system(self):
        """Return a fresh LatentMASSystem that reuses our already-loaded model.

        To avoid loading the model twice (32 GB) we monkey-patch the system
        to skip its own ``__init__`` model loading.
        """
        from ..system import LatentMASSystem, SystemConfig
        from ..core.embedding_guard import EmbeddingGuard
        from ..core.latent_memory import LatentMemory

        system = LatentMASSystem.__new__(LatentMASSystem)
        system.config = SystemConfig(
            model_name=self.model_name,
            device=self.device,
            cache_dir=self.cache_dir,
            dtype=self.dtype,
            latent_steps=self.latent_steps,
        )
        system.device = self.device
        system.tokenizer = self._tokenizer
        system.model = self._model
        system.guard = EmbeddingGuard(self._model)
        system.memory = LatentMemory(device=self.device)
        system._reasoner = None
        system._pool = None
        system._lora_manager = None
        system._pipeline = None
        system._initialized = False
        return system

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run_all(
        self,
        benchmarks: List[str],
        max_samples: int = -1,
        seed: int = 42,
        output_dir: str = "results/ablation/",
        configs: Optional[List[str]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AblationResult:
        """Run all (or selected) configs on all benchmarks.

        Parameters
        ----------
        benchmarks : list[str]
            E.g. ``["gsm8k", "medqa"]``.
        max_samples : int
            Cap per benchmark (``-1`` = full set).
        seed : int
            For reproducible sub-sampling.
        output_dir : str
            Where to write JSON results.
        configs : list[str] | None
            Subset of ``self.CONFIGS`` to run.  ``None`` = all four.
        pipeline_kwargs : dict | None
            Extra kwargs forwarded to each pipeline call.

        Returns
        -------
        AblationResult
        """
        from .benchmarks import BenchmarkRunner

        configs = configs or list(self.CONFIGS)
        result = AblationResult(metadata={
            "model": self.model_name,
            "benchmarks": benchmarks,
            "max_samples": max_samples,
            "seed": seed,
            "configs_run": configs,
        })

        t_start = time.time()

        for cfg_name in configs:
            logger.info("=" * 60)
            logger.info("Setting up config: %s", cfg_name)
            logger.info("=" * 60)

            pipeline = self.setup_config(cfg_name)
            runner = BenchmarkRunner(
                pipeline=pipeline,
                tokenizer=self._tokenizer,
                device=self.device,
                cache_dir=self.cache_dir,
            )

            result.configs[cfg_name] = {}

            for bench in benchmarks:
                logger.info("--- %s / %s ---", cfg_name, bench)
                br = runner.evaluate(
                    bench,
                    max_samples=max_samples,
                    seed=seed,
                    pipeline_kwargs=pipeline_kwargs,
                )
                cr = ConfigResult(
                    config_name=cfg_name,
                    dataset_name=bench,
                    accuracy=br.accuracy,
                    num_correct=br.num_correct,
                    num_total=br.num_total,
                    avg_tokens=br.avg_tokens,
                    avg_latency_ms=br.avg_latency_ms,
                    ci_lower=br.ci_lower,
                    ci_upper=br.ci_upper,
                    correctness_vector=br.correctness_vector,
                )
                result.configs[cfg_name][bench] = cr

            # Free adapter state before next config
            self._unload_adapters()
            torch.cuda.empty_cache()

        result.wall_time_s = time.time() - t_start

        # Significance tests (B vs D on each benchmark)
        result.significance = self.compute_significance(result)

        # Persist
        out_path = Path(output_dir) / "ablation_results.json"
        result.save(out_path)

        # Print comparison table
        table = result.comparison_table()
        print("\n" + table + "\n")
        for bench, sig in result.significance.items():
            print(f"[SIG] {bench}: {sig.get('summary', 'n/a')}")

        return result

    # ------------------------------------------------------------------
    # Significance
    # ------------------------------------------------------------------
    def compute_significance(self, result: AblationResult) -> Dict[str, Any]:
        """Paired bootstrap between Config B and Config D on each benchmark.

        Returns dict mapping benchmark name → BootstrapTestResult summary.
        """
        from .statistical import paired_bootstrap_test

        sig: Dict[str, Any] = {}

        b_results = result.configs.get("B_latentmas_nolora", {})
        d_results = result.configs.get("D_latentmas_speclora", {})

        for bench in b_results:
            if bench not in d_results:
                continue
            cr_b = b_results[bench]
            cr_d = d_results[bench]
            if len(cr_b.correctness_vector) != len(cr_d.correctness_vector):
                logger.warning(
                    "Length mismatch for %s: B=%d, D=%d – skipping significance",
                    bench, len(cr_b.correctness_vector), len(cr_d.correctness_vector),
                )
                continue

            bt = paired_bootstrap_test(
                cr_b.correctness_vector,
                cr_d.correctness_vector,
                n_bootstrap=10_000,
                name_a="B_latentmas_nolora",
                name_b="D_latentmas_speclora",
            )
            sig[bench] = {
                "delta": bt.delta,
                "p_value": bt.p_value,
                "significant": bt.significant,
                "ci_lower": bt.ci_lower,
                "ci_upper": bt.ci_upper,
                "summary": bt.summary,
            }

        return sig
