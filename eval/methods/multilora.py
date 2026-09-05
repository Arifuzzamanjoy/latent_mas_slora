"""
Composed multi-LoRA reasoning on a single prompt.

The sequential design - one adapter per agent, KV cache handed between them -
has two problems this method sidesteps. First, a cache computed under adapter A
is not valid input for adapter B, since LoRA modifies the Q/K/V projections from
the first token (see Activated-LoRA 2512.17910, LRAgent 2602.01053, ForkKV
2604.06370). Second, role decomposition by prompting is documented not to
produce synergy below GPT-4 scale (Solo Performance Prompting, 2307.05300).

So instead of moving state between differently-adapted agents, this composes the
adapters themselves for one instance and runs a single prompt:

    probe every resident adapter -> score -> top-k -> weighted merge
    -> one weight set -> latent thought steps -> decode

One prompt, one weight set, one KV cache: the cross-adapter cache problem cannot
arise. Selection follows LoGo (2511.07129), which is training-free - it scores
adapters from activation signals in a single probe pass rather than learning
composition weights.

The prompt is deliberately the same one baseline-cot uses, so `multi-lora` vs
`baseline-cot` isolates exactly one thing: adapter composition plus latent depth.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import GenSettings
from ..data import EvalItem
from ..extract import extract_answer
from .base import Method, Sample
from .baselines import SYS_COT, USR_COT_MCQ, USR_COT_NUM

MIX_ADAPTER = "_logo_mix"


class MultiLoRA(Method):
    name = "multi-lora"
    backend_kind = "system"
    description = ("LoGo-style: probe resident adapters, merge top-k per instance, "
                   "one prompt + latent steps, single KV cache.")
    defaults = {
        "top_k": 3,                  # LoGo merges top-k; 3 is sane for 7 adapters
        "score_mode": "norm",        # norm | entropy
        "adapter_policy": "logo",    # logo | merge | same | none
        "adapter": None,             # used when adapter_policy == "same"
        "latent_steps": 50,          # paper's optimum band is 40-80
    }

    def __init__(self, backend, args, cfg):
        super().__init__(backend, args, cfg)
        self.mock = getattr(backend, "kind", "") == "mock"
        self.system = None if self.mock else backend.system
        self.policy = args.get("adapter_policy", "logo")
        self.top_k = int(args.get("top_k", 3))
        self.score_mode = args.get("score_mode", "norm")
        self._warned = False

    # -- prompt ------------------------------------------------------------
    def _prompt(self, item: EvalItem) -> str:
        tmpl = USR_COT_MCQ if item.task_type == "mcq" else USR_COT_NUM
        msgs = [{"role": "system", "content": SYS_COT},
                {"role": "user", "content": tmpl.format(q=item.question)}]
        return self.system.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def _latent_prompt(self, item: EvalItem) -> str:
        """Prompt for the latent pass - thinking only, never decoded."""
        msgs = [{"role": "system", "content": SYS_COT},
                {"role": "user",
                 "content": f"{item.question}\n\nThink through this problem carefully."}]
        return self.system.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    # -- adapter selection -------------------------------------------------
    def _candidates(self) -> List[str]:
        """
        Every adapter resident on the model, not just the role adapters.

        Adapters pulled in with --loras are registered on the model but belong to
        no agent, so a pool-only list made them unreachable: they would load,
        occupy memory, and never participate. Composition is exactly where an
        externally trained adapter is useful, so it is included here.
        """
        names = [self.system._pool.get(n).adapter_name
                 for n in self.system._pool.list_agents()]
        for extra in getattr(self.system.model, "peft_config", {}):
            if extra not in names and extra != MIX_ADAPTER:
                names.append(extra)
        return names

    def _probe(self, input_ids, attn) -> List[Tuple[str, float]]:
        """
        Score each adapter by how much it *changes* the computation.

        LoGo's signal is the norm of the LoRA activations, i.e. the adapter's
        contribution - not the norm of the final hidden state. Scoring the latter
        is actively wrong here: an identity adapter inherits the base model's
        norm, while a trained adapter shifts it in either direction, so the
        untrained adapters can outrank the real one. Measuring the deviation
        from the base model instead makes an identity adapter score exactly 0.0
        by construction, so it can never win.
        """
        import torch

        def last_hidden():
            with torch.no_grad():
                out = self.system.model(input_ids=input_ids, attention_mask=attn,
                                        output_hidden_states=True, return_dict=True)
            return out.hidden_states[-1][:, -1, :].float(), out.logits[:, -1, :].float()

        model = self.system.model
        try:
            with model.disable_adapter():
                h_base, lg_base = last_hidden()
        except Exception:                      # no PEFT wrapper: nothing to compare
            h_base, lg_base = last_hidden()

        scores: List[Tuple[str, float]] = []
        for adapter in self._candidates():
            try:
                model.set_adapter(adapter)
            except Exception:
                continue
            h_a, lg_a = last_hidden()
            if self.score_mode == "entropy":
                lp_a = torch.log_softmax(lg_a, dim=-1)
                lp_b = torch.log_softmax(lg_base, dim=-1)
                # KL(adapter || base): how much the adapter moves the distribution
                score = float((lp_a.exp() * (lp_a - lp_b)).sum(-1).item())
            else:
                score = float((h_a - h_base).norm().item())
            scores.append((adapter, score))
        # deterministic order so ties never depend on registration order
        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores

    @staticmethod
    def is_degenerate(scores: List[Tuple[str, float]]) -> bool:
        """
        True when the probe cannot tell the adapters apart.

        Zero-initialised LoRA adapters are identity functions, so every one of
        them produces identical activations and selection silently becomes a
        uniform merge. That looks like a working pipeline and measures nothing,
        so it is reported rather than left to be discovered in the results.
        """
        if len(scores) < 2:
            return False
        vals = [v for _, v in scores]
        if max(abs(v) for v in vals) < 1e-9:
            return True                      # every adapter contributes nothing
        spread = max(vals) - min(vals)
        scale = max(abs(v) for v in vals) or 1.0
        return spread / scale < 1e-6

    def _compose(self, scores: List[Tuple[str, float]]) -> Tuple[List[str], List[float]]:
        """Top-k by score, weights normalized over the survivors."""
        # deterministic: score desc, then name - never registration order
        ranked = sorted(scores, key=lambda kv: (-kv[1], kv[0]))
        # an adapter that changes nothing is not a candidate while a real one exists
        live = [kv for kv in ranked if kv[1] > 1e-9]
        ranked = (live or ranked)[:max(1, self.top_k)]
        names = [n for n, _ in ranked]
        raw = [s for _, s in ranked]
        lo = min(raw)
        shifted = [s - lo + 1e-6 for s in raw]      # keep weights non-negative
        total = sum(shifted) or 1.0
        return names, [w / total for w in shifted]

    def _activate(self, names: List[str], weights: List[float]) -> str:
        """Merge the selected adapters into one weight set and activate it."""
        model = self.system.model
        if len(names) == 1:
            model.set_adapter(names[0])
            return names[0]

        try:
            if MIX_ADAPTER in getattr(model, "peft_config", {}):
                model.delete_adapter(MIX_ADAPTER)
        except Exception:
            pass

        # 'cat' tolerates adapters of differing rank (this pool mixes 32/48/64);
        # 'linear' is tried first because it preserves the weighting exactly.
        for combo in ("linear", "cat"):
            try:
                model.add_weighted_adapter(names, weights, MIX_ADAPTER,
                                           combination_type=combo)
                model.set_adapter(MIX_ADAPTER)
                return f"{MIX_ADAPTER}({combo})"
            except Exception:
                continue

        model.set_adapter(names[0])       # composition unavailable: fall back to top-1
        return names[0]

    # -- main --------------------------------------------------------------
    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        if self.mock:
            out = self.backend.generate(f"{self.name}|{item.question}", gen,
                                        num_choices=item.num_choices)
            return self._finish(out, item, {"mock": True, "adapter_policy": self.policy})

        import torch
        from ..backends import _seed_everything
        _seed_everything(gen.seed)

        p = self.system._pipeline
        tok = self.system.tokenizer
        device = self.system.device
        steps = int(self.args.get("latent_steps", 50))

        prompt = self._prompt(item)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        t0 = time.time()
        degenerate = False
        chosen: List[str] = []
        weights: List[float] = []
        scores: List[Tuple[str, float]] = []
        probe_ms = 0

        if self.policy == "none":
            active = "base"
        elif self.policy == "same":
            active = self.args.get("adapter") or self._candidates()[0]
            self.system.model.set_adapter(active)
            chosen, weights = [active], [1.0]
        elif self.policy == "merge":
            chosen = self._candidates()
            weights = [1.0 / len(chosen)] * len(chosen)
            active = self._activate(chosen, weights)
        else:                                        # logo
            t_probe = time.time()
            scores = self._probe(input_ids, attn)
            probe_ms = int((time.time() - t_probe) * 1000)
            degenerate = self.is_degenerate(scores)
            if degenerate and not self._warned:
                self._warned = True
                print("[multi-lora] WARNING: all adapters scored identically - they are "
                      "indistinguishable (zero-initialised LoRA is an identity function). "
                      "Selection is a no-op and this run measures the base model. "
                      "Train the adapters before drawing conclusions.")
            chosen, weights = self._compose(scores)
            active = self._activate(chosen, weights)

        # Latent thoughts, then decode conditioned on them. Both passes run under
        # the SAME merged weight set, which is the whole point: the cache the
        # decoder reads was written by the weights it is running.
        with torch.no_grad():
            cache = None
            latent_prefix = 0
            if steps > 0:
                think = tok(self._latent_prompt(item), return_tensors="pt",
                            truncation=True, max_length=4096)
                res = p.reasoner.reason(
                    input_ids=think["input_ids"].to(device),
                    attention_mask=think["attention_mask"].to(device),
                    num_steps=steps, past_key_values=None,
                )
                cache = res.kv_cache
                from src.core.latent_reasoner import get_cache_length
                latent_prefix = get_cache_length(cache)

            text, n_new = p._decode_with_cache(
                input_ids, cache, gen.temperature, gen.top_p, gen.max_new_tokens)

        latency = int((time.time() - t0) * 1000)
        ex = extract_answer(text, item.task_type, item.num_choices)
        return Sample(
            text=text, pred=ex.answer, extract_rule=ex.rule, extract_failed=ex.failed,
            prompt_tokens=int(input_ids.shape[1]), completion_tokens=n_new,
            latency_ms=latency,
            extra={
                "adapter_policy": self.policy,
                "active_adapter": active,
                "adapters": chosen,
                "adapter_weights": [round(w, 4) for w in weights],
                "adapter_scores": [(n, round(s, 3)) for n, s in scores],
                "probe_ms": probe_ms,
                "degenerate_probe": degenerate,
                "latent_steps": steps,
                "latent_prefix_len": latent_prefix,
                "score_mode": self.score_mode,
                "top_k": self.top_k,
            },
        )
