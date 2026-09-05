"""
Model backends.

Two real backends are needed because the baselines must not see the PEFT
wrapper that the multi-agent system installs:

  hf      - plain AutoModelForCausalLM, used by every baseline method
  system  - LatentMASSystem (PEFT-wrapped, agent pool, latent reasoner)

Methods declare which one they need; the runner loads one at a time and frees
it before the next, so a 7B eval fits in 24GB.

  mock    - no weights at all, deterministic fake outputs. Use --dry-run to
            exercise the whole pipeline (segmentation, voting, metrics,
            reports) on CPU in seconds.
"""

import gc
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .config import EvalConfig, GenSettings


@dataclass
class GenOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    extra: Dict[str, Any] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _seed_everything(seed: int) -> None:
    import random as _random

    import torch

    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Backend:
    kind = "base"

    def free(self) -> None:
        pass

    def peak_vram_gb(self) -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / 1e9
        except Exception:
            pass
        return 0.0


# ─── Plain HuggingFace ───────────────────────────────────────────────────────


class HFBackend(Backend):
    kind = "hf"

    def __init__(self, cfg: EvalConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        print(f"[backend:hf] loading {cfg.model} ({cfg.dtype})")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {"cache_dir": cfg.cache_dir}
        if cfg.dtype == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )
        else:
            kwargs["dtype"] = dtype_map.get(cfg.dtype, torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
        if cfg.dtype != "4bit":
            self.model = self.model.to(cfg.device)
        self.model.eval()
        print(f"[backend:hf] ready in {time.time() - t0:.1f}s")

    def chat_prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, gen: GenSettings, num_choices: int = 4) -> GenOutput:
        import torch

        _seed_everything(gen.seed)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
        kw: Dict[str, Any] = {
            "max_new_tokens": gen.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if gen.temperature > 0:
            kw.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p)
        else:
            kw["do_sample"] = False

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        n_in = enc["input_ids"].shape[1]
        new = out[0][n_in:]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        return GenOutput(text, n_in, int(new.shape[0]), int((time.time() - t0) * 1000), {})

    def loglikelihood(self, prompt: str, continuations: List[str]) -> Tuple[List[float], int]:
        """Mean per-token logprob of each continuation given the prompt.

        Used by scoring=loglikelihood, which removes answer extraction from the
        measurement entirely.
        """
        import torch

        scores: List[float] = []
        ctx = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
        n_ctx = ctx["input_ids"].shape[1]
        for cont in continuations:
            cont_ids = self.tokenizer(cont, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(self.cfg.device)
            ids = torch.cat([ctx["input_ids"], cont_ids], dim=1)
            with torch.no_grad():
                logits = self.model(ids).logits
            logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            target = ids[:, 1:]
            picked = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)
            cont_lp = picked[:, n_ctx - 1 :]
            scores.append(float(cont_lp.mean().item()))
        return scores, n_ctx

    def free(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── LatentMAS system ────────────────────────────────────────────────────────


class SystemBackend(Backend):
    kind = "system"

    def __init__(self, cfg: EvalConfig, agent_max_tokens: int = 50):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src import AgentConfig, LatentMASSystem

        self.cfg = cfg
        print(f"[backend:system] building LatentMASSystem on {cfg.model}")
        t0 = time.time()
        self.system = LatentMASSystem(
            model_name=cfg.model,
            device=cfg.device,
            dtype=cfg.dtype,
            cache_dir=cfg.cache_dir,
            latent_steps=cfg.latent_steps,
        )
        for a in [
            AgentConfig.planner(max_tokens=agent_max_tokens),
            AgentConfig.medical(max_tokens=agent_max_tokens),
            AgentConfig.math(max_tokens=agent_max_tokens),
            AgentConfig.coder(max_tokens=agent_max_tokens),
            AgentConfig.critic(max_tokens=agent_max_tokens),
            AgentConfig.refiner(max_tokens=agent_max_tokens),
            AgentConfig.judger(max_tokens=cfg.max_new_tokens),
        ]:
            self.system.add_agent(a)

        for lora in cfg.loras:
            ok = self.system.load_from_registry(lora)
            print(f"[backend:system] registry LoRA '{lora}': {'loaded' if ok else 'FAILED'}")

        self.tokenizer = self.system.tokenizer
        print(
            f"[backend:system] ready in {time.time() - t0:.1f}s "
            f"({len(self.system._pool.list_agents())} agents)"
        )

    def free(self) -> None:
        import torch

        del self.system
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── Mock ────────────────────────────────────────────────────────────────────


class MockBackend(Backend):
    """Deterministic pseudo-model for --dry-run.

    Answers are a hash of (question, seed), so accuracy lands near chance and
    self-consistency produces genuine vote spread - enough to exercise every
    downstream code path without a GPU.
    """

    kind = "mock"

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.tokenizer = None

    def chat_prompt(self, system: str, user: str) -> str:
        return f"<system>{system}</system>\n<user>{user}</user>"

    def _pick(self, prompt: str, gen: GenSettings, n: int) -> str:
        h = hashlib.sha256(f"{prompt}|{gen.seed}|{gen.temperature}".encode()).hexdigest()
        return "ABCDEFGHIJ"[int(h[:8], 16) % max(2, n)]

    def generate(self, prompt: str, gen: GenSettings, num_choices: int = 4) -> GenOutput:
        letter = self._pick(prompt, gen, num_choices)
        text = f"Reasoning omitted (mock backend).\n\\boxed{{{letter}}}"
        time.sleep(0.001)
        return GenOutput(text, len(prompt) // 4, 24, 1, {"mock": True})

    def loglikelihood(self, prompt: str, continuations: List[str]) -> Tuple[List[float], int]:
        scores = []
        for i, c in enumerate(continuations):
            h = hashlib.sha256(f"{prompt}|{c}|{i}".encode()).hexdigest()
            scores.append(-(int(h[:6], 16) % 1000) / 100.0)
        return scores, len(prompt) // 4


# ─── Factory ─────────────────────────────────────────────────────────────────


def build_backend(kind: str, cfg: EvalConfig, dry_run: bool = False) -> Backend:
    if dry_run or kind == "mock":
        return MockBackend(cfg)
    if kind == "hf":
        return HFBackend(cfg)
    if kind == "system":
        return SystemBackend(cfg)
    raise ValueError(f"unknown backend kind: {kind}")
