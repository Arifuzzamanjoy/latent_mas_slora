"""
Multi-agent methods.

All of these drive src/ - they are the thing under test. Each one is selectable
independently so a run can isolate a single mechanism:

  text-mas       every agent decodes text (the classic multi-agent control)
  latent-mas     src's true_latent as written
  latent-mas-kv  true_latent with the latent KV cache actually handed to the
                 final agent's decoder (see the note in LatentKVMAS)
  sequential-mas the chain-of-agents pipeline
  router-only    routing decision only, no generation
"""

import time
from typing import Any, Dict, List

from ..config import GenSettings
from ..data import EvalItem
from ..extract import extract_answer
from .base import Method, Sample

DEFAULT_PIPELINES = {
    "medical": ["Planner", "MedicalExpert", "Critic", "Judger"],
    "math": ["Planner", "MathExpert", "Critic", "Judger"],
    "code": ["Planner", "CodeExpert", "Critic", "Judger"],
    "reasoning": ["Planner", "Critic", "Refiner", "Judger"],
    "general": ["Planner", "Critic", "Refiner", "Judger"],
}

DEFAULT_LATENT_STEPS = {
    "medical": 15,
    "math": 12,
    "code": 10,
    "reasoning": 12,
    "general": 8,
}


class _MASMethod(Method):
    backend_kind = "system"
    pipeline_name = "hierarchical"

    def __init__(self, backend, args, cfg):
        super().__init__(backend, args, cfg)
        self.mock = getattr(backend, "kind", "") == "mock"
        self.system = None if self.mock else backend.system
        self._router = None
        if args.get("use_router", True) and not self.mock:
            from src.routing import SemanticRouter

            self._router = SemanticRouter()
        if not self.mock:
            self._apply_policy()

    def _apply_policy(self) -> None:
        """
        Push this method's configuration onto the shared system.

        The system backend is shared by every method in its group, so each
        method sets the knobs it owns immediately before it runs. That is what
        lets kv_handoff and prompt_style be per-method conditions rather than
        one global setting.
        """
        from src.agents.configs import AgentConfig

        self.system._pipeline.kv_handoff = bool(self.args.get("kv_handoff", True))
        self.system.kv_handoff = self.system._pipeline.kv_handoff

        style = self.args.get("prompt_style", "reason_first")
        judger = self.system._pool.get("Judger")
        if judger is not None and judger.prompt_style != style:
            self.system._pool.register(
                AgentConfig.judger(max_tokens=judger.max_tokens, prompt_style=style)
            )

    def sample_prelude(self) -> None:
        self._apply_policy()

    def _mock_sample(self, item, gen):
        """Dry-run path: no weights, but the same record shape."""
        agents = self.args.get("agents") or DEFAULT_PIPELINES["general"]
        steps = self.args.get("latent_steps", self.cfg.latent_steps)
        prompt = f"{self.name}|{'>'.join(agents)}|{item.question}"
        out = self.backend.generate(prompt, gen, num_choices=item.num_choices)
        return self._finish(
            out,
            item,
            {"routed_domain": item.domain, "agents": agents, "latent_steps": steps, "mock": True},
        )

    def _plan(self, item: EvalItem):
        """Choose agent list and latent steps for this item."""
        forced = self.args.get("agents")
        steps = self.args.get("latent_steps", self.cfg.latent_steps)
        domain, conf = item.domain, None

        if self._router is not None:
            d, conf = self._router.get_best_domain(item.question)
            domain = d.value

        agents = forced or DEFAULT_PIPELINES.get(domain, DEFAULT_PIPELINES["general"])
        if self.args.get("adaptive_latent_steps", True) and "latent_steps" not in self.args:
            steps = DEFAULT_LATENT_STEPS.get(domain, steps)
        return agents, steps, domain, conf

    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        if self.mock:
            return self._mock_sample(item, gen)
        from ..backends import _seed_everything

        self._apply_policy()
        _seed_everything(gen.seed)

        agents, steps, domain, conf = self._plan(item)
        self.system._pipeline.latent_steps = steps

        t0 = time.time()
        res = self.system.run(
            question=item.question,
            pipeline=self.pipeline_name,
            agents=agents,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            self_consistency=1,  # voting is the runner's job, for all methods alike
        )
        latency = int((time.time() - t0) * 1000)

        ex = extract_answer(res.final_answer, item.task_type, item.num_choices)
        completion = sum(o.get("output_tokens", 0) for o in res.agent_outputs)
        prompt_toks = max(0, res.total_tokens - completion)
        return Sample(
            text=res.final_answer,
            pred=ex.answer,
            extract_rule=ex.rule,
            extract_failed=ex.failed,
            prompt_tokens=prompt_toks,
            completion_tokens=completion,
            latency_ms=latency,
            extra={
                "routed_domain": domain,
                "router_confidence": round(conf, 4) if conf is not None else None,
                "agents": agents,
                "latent_steps": steps,
                "kv_handoff": self.args.get("kv_handoff", True),
                "prompt_style": self.args.get("prompt_style", "reason_first"),
                "latent_steps_total": res.latent_steps_total,
                # keep the adapter on each step: the chain's whole claim is that a
                # different LoRA is active at each hop, so the trace has to show it
                "per_agent": [
                    {
                        k: o.get(k)
                        for k in (
                            "agent",
                            "adapter",
                            "output_tokens",
                            "latency_ms",
                            "mode",
                            "latent_prefix_len",
                        )
                    }
                    for o in res.agent_outputs
                ],
            },
        )


class TextMAS(_MASMethod):
    name = "text-mas"
    pipeline_name = "hierarchical"
    description = "Hierarchical pipeline, every agent decodes text into the next agent's prompt."


class LatentMAS(_MASMethod):
    """As shipped before the fixes: cache discarded, answer-first judger prompt."""

    name = "latent-mas"
    pipeline_name = "true_latent"
    description = "true_latent as originally shipped (cache discarded, answer-first prompt)."
    defaults = {"kv_handoff": False, "prompt_style": "answer_first"}


class LatentMASPaper(_MASMethod):
    """
    The LatentMAS reference configuration: the latent working memory reaches the
    decoder and the judger reasons before it answers.

    latent-mas -> latent-mas-kv -> latent-mas-paper is an ablation ladder; each
    rung changes exactly one thing, so a paired test attributes the difference.
    """

    name = "latent-mas-paper"
    pipeline_name = "true_latent"
    description = "Reference configuration: KV handoff + reason-first judger prompt."
    defaults = {"kv_handoff": True, "prompt_style": "reason_first"}


class SequentialMAS(_MASMethod):
    name = "sequential-mas"
    pipeline_name = "sequential"
    description = "Chain-of-agents sequential pipeline."


class LatentKVMAS(_MASMethod):
    """
    true_latent with the latent state actually reaching the answer.

    In src/pipelines/hierarchical.py the accumulated KV cache is stored in
    LatentMemory but the final model.generate() call is given only input_ids and
    attention_mask, so the decoder never attends to it. This method runs the same
    latent loop and then decodes conditioned on that cache, which makes the
    mechanism testable: latent-mas vs latent-mas-kv is the ablation that says
    whether latent collaboration does anything at all.

    Decoding is a manual loop rather than model.generate() so the cache
    semantics do not depend on the transformers version.
    """

    name = "latent-mas-kv"
    pipeline_name = "true_latent"
    description = "KV handoff only, legacy prompt - isolates the cache fix."
    defaults = {"kv_handoff": True, "prompt_style": "answer_first"}

    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        if self.mock:
            return self._mock_sample(item, gen)
        from src.core.latent_reasoner import get_cache_length

        from ..backends import _seed_everything

        self._apply_policy()
        _seed_everything(gen.seed)
        agents, steps, domain, conf = self._plan(item)

        p = self.system._pipeline
        tok = self.system.tokenizer
        device = self.system.device
        p.memory.clear()
        p.latent_steps = steps

        t0 = time.time()
        prompt_tokens = 0
        per_agent: List[Dict[str, Any]] = []

        # 1. Intermediate agents: latent only, accumulating a shared KV cache.
        for name in agents[:-1]:
            a_start = time.time()
            cfg_a = p.pool.activate(name)
            prompt = p.executor.build_prompt(cfg_a, item.question, "")
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            res = p.reasoner.reason(
                input_ids=input_ids,
                attention_mask=attn,
                num_steps=steps,
                past_key_values=p.memory.get_kv_cache(),
            )
            p.memory.store_hidden_state(name, res.final_hidden)
            p.memory.update_kv_cache(res.kv_cache)
            prompt_tokens += int(input_ids.shape[1])
            per_agent.append(
                {
                    "agent": name,
                    "output_tokens": 0,
                    "mode": "latent_only",
                    "latency_ms": int((time.time() - a_start) * 1000),
                }
            )

        # 2. Final agent decodes *conditioned on* the accumulated latent cache.
        final_name = agents[-1]
        cfg_f = p.pool.activate(final_name)
        prompt = p.executor.build_prompt(cfg_f, item.question, "")
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        n_prompt = int(input_ids.shape[1])
        prompt_tokens += n_prompt

        cache = p.memory.get_kv_cache()
        past_len = get_cache_length(cache)
        text, n_new = self._decode(
            input_ids,
            cache,
            past_len,
            gen,
            tok,
            device,
            temperature=gen.temperature,
            top_p=gen.top_p,
        )

        latency = int((time.time() - t0) * 1000)
        ex = extract_answer(text, item.task_type, item.num_choices)
        per_agent.append(
            {
                "agent": final_name,
                "output_tokens": n_new,
                "mode": "latent+text(kv)",
                "latency_ms": latency,
            }
        )

        return Sample(
            text=text,
            pred=ex.answer,
            extract_rule=ex.rule,
            extract_failed=ex.failed,
            prompt_tokens=prompt_tokens,
            completion_tokens=n_new,
            latency_ms=latency,
            extra={
                "routed_domain": domain,
                "router_confidence": round(conf, 4) if conf is not None else None,
                "agents": agents,
                "latent_steps": steps,
                "kv_handoff": True,
                "prompt_style": self.args.get("prompt_style", "answer_first"),
                "latent_prefix_len": past_len,
                "latent_steps_total": steps * (len(agents) - 1),
                "per_agent": per_agent,
            },
        )

    def _decode(self, input_ids, cache, past_len, gen, tok, device, temperature, top_p):
        """Greedy / nucleus decode continuing from an existing KV cache.

        Written by hand instead of model.generate() because passing an external
        cache through generate() has version-dependent semantics; here the
        attention mask and cache growth are explicit.
        """
        import torch

        model = self.system.model
        eos_ids = {tok.eos_token_id}
        extra_eos = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(extra_eos, int) and extra_eos >= 0:
            eos_ids.add(extra_eos)

        generated: List[int] = []
        cur = input_ids
        seen = past_len
        with torch.no_grad():
            for _ in range(gen.max_new_tokens):
                seen += int(cur.shape[1])
                mask = torch.ones((1, seen), dtype=torch.long, device=device)
                out = model(
                    input_ids=cur,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out.past_key_values
                logits = out.logits[:, -1, :].float()

                if temperature and temperature > 0:
                    logits = logits / max(temperature, 1e-5)
                    probs = torch.softmax(logits, dim=-1)
                    if top_p and 0 < top_p < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                        cum = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = (cum - sorted_probs) > top_p
                        sorted_probs[cutoff] = 0.0
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        pick = torch.multinomial(sorted_probs, 1)
                        nxt = sorted_idx.gather(-1, pick)
                    else:
                        nxt = torch.multinomial(probs, 1)
                else:
                    nxt = logits.argmax(dim=-1, keepdim=True)

                tok_id = int(nxt.item())
                if tok_id in eos_ids:
                    break
                generated.append(tok_id)
                cur = nxt

        text = tok.decode(generated, skip_special_tokens=True).strip()
        return text, len(generated)


class RouterOnly(Method):
    """
    Routing decision only - no generation.

    Scored against the item's dataset domain label, so it answers "is the
    semantic router picking the right pipeline?" separately from "does the
    pipeline answer correctly?". Cheap enough to run on 100% of the data.
    """

    name = "router-only"
    backend_kind = "none"
    description = "Semantic router domain classification, scored against dataset domain labels."

    def __init__(self, backend, args, cfg):
        super().__init__(backend, args, cfg)
        from src.routing import SemanticRouter

        self.router = SemanticRouter(
            model_name=args.get("router_model", "all-MiniLM-L6-v2"),
            use_embeddings=args.get("use_embeddings", not cfg.dry_run),
        )
        self.threshold = args.get("confidence_threshold", 0.15)

    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        t0 = time.time()
        domain, conf = self.router.get_best_domain(
            item.question, confidence_threshold=self.threshold
        )
        ranked = self.router.route(item.question, top_k=5)
        return Sample(
            text=domain.value,
            pred=domain.value,
            extract_rule="router",
            extract_failed=False,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=int((time.time() - t0) * 1000),
            extra={
                "confidence": round(conf, 4),
                "ranked": [(d.value, round(s, 4)) for d, s in ranked],
                "gold_domain": item.domain,
            },
        )

    def gold_of(self, item: EvalItem) -> str:
        return item.domain

    def task_type_of(self, item: EvalItem) -> str:
        return "text"
