#!/usr/bin/env python3
"""
Validate that the agent chain really switches LoRA adapters, and that each
agent's work reaches the one that answers.

The chain's claim has three separable parts, and each can fail independently
while the run still looks healthy:

  1. SWITCH   - a different adapter is active at every hop, in order
  2. APPLY    - the active adapter actually changes that step's computation
  3. TRANSFER - the latent working memory carries it to the final decoder

Part 2 fails silently when adapters are untrained (zero-initialised LoRA is an
identity function). Part 3 fails silently when the KV cache is built and then
dropped before decoding. Both look exactly like a working pipeline.

Usage:
  python tools/validate_chain.py                          # small model, CPU-friendly
  python tools/validate_chain.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import AgentConfig, LatentMASSystem
from src.core.latent_reasoner import get_cache_length

QUESTION = (
    "A patient has bloody diarrhea with pseudopolyps. Greatest risk?\n"
    "A. HUS\nB. Oral ulcers\nC. Colorectal cancer\nD. Pancreatic cancer"
)


def b_norm(model, adapter: str) -> float:
    return (
        sum(
            p.detach().float().norm().item() ** 2
            for n, p in model.named_parameters()
            if "lora_B" in n and adapter in n
        )
        ** 0.5
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, help="default: bfloat16 on cuda, float32 on cpu")
    ap.add_argument("--latent-steps", type=int, default=8)
    ap.add_argument(
        "--probe-adapter",
        default="medical_lora",
        help="adapter to perturb when testing APPLY and TRANSFER",
    )
    a = ap.parse_args()
    dtype = a.dtype or ("bfloat16" if a.device == "cuda" else "float32")

    s = LatentMASSystem(
        model_name=a.model, device=a.device, dtype=dtype, latent_steps=a.latent_steps
    )
    for cfg in (
        AgentConfig.planner(max_tokens=40),
        AgentConfig.medical(max_tokens=40),
        AgentConfig.critic(max_tokens=40),
        AgentConfig.judger(max_tokens=60),
    ):
        s.add_agent(cfg)
    tok, p, dev = s.tokenizer, s._pipeline, s.device
    chain = ["Planner", "MedicalExpert", "Critic", "Judger"]
    ok = True

    # 1. SWITCH -------------------------------------------------------------
    seen = []
    original = s.model.set_adapter
    s.model.set_adapter = lambda n, *ar, **kw: (seen.append(n), original(n, *ar, **kw))[1]
    res = s.run(
        question=QUESTION, pipeline="true_latent", agents=chain, max_new_tokens=40, temperature=0.0
    )
    s.model.set_adapter = original

    expected = [s._pool.get(n).adapter_name for n in chain]
    print("\n1. SWITCH — one adapter per hop, in order")
    for step, (want, got) in enumerate(zip(expected, seen), 1):
        mark = "ok" if want == got else "MISMATCH"
        print(f"     hop {step}: {chain[step - 1]:<15} -> {got:<16} [{mark}]")
    if seen[: len(expected)] != expected:
        print(f"     FAIL: expected {expected}, saw {seen}")
        ok = False

    print("\n   per-agent trace recorded in the result:")
    for o in res.agent_outputs:
        print(
            f"     {o['agent']:<15} adapter={o['adapter']:<16} mode={o.get('mode', '-'):<18}"
            f" prefix={o.get('latent_prefix_len', '-')}"
        )

    # 2. APPLY --------------------------------------------------------------
    def expert_hidden():
        s._pool.activate("MedicalExpert")
        e = tok(QUESTION, return_tensors="pt")
        r = p.reasoner.reason(
            input_ids=e["input_ids"].to(dev),
            attention_mask=e["attention_mask"].to(dev),
            num_steps=4,
        )
        return r.final_hidden.float().clone()

    h0 = expert_hidden()
    before = b_norm(s.model, a.probe_adapter)
    with torch.no_grad():  # stand in for a trained adapter
        for n, q in s.model.named_parameters():
            if "lora_B" in n and a.probe_adapter in n:
                q.normal_(0, 0.002)
    after = b_norm(s.model, a.probe_adapter)
    d_hidden = (expert_hidden() - h0).abs().max().item()

    print("\n2. APPLY — does the active adapter change its own step?")
    print(f"     ||B|| {a.probe_adapter}: {before:.4f} -> {after:.4f}")
    print(f"     max|Δ hidden| = {d_hidden:.3e}  [{'ok' if d_hidden > 1e-4 else 'FAIL'}]")
    if before > 1e-9:
        print("     note: adapter already had weights; APPLY result is still valid")
    if d_hidden <= 1e-4:
        print("     FAIL: the switch does not reach the computation")
        ok = False

    # 3. TRANSFER -----------------------------------------------------------
    def judger_logits():
        p.memory.clear()
        for name in chain[:-1]:
            cfg = p.pool.activate(name)
            e = tok(
                p.executor.build_prompt(cfg, QUESTION, ""),
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            r = p.reasoner.reason(
                input_ids=e["input_ids"].to(dev),
                attention_mask=e["attention_mask"].to(dev),
                num_steps=a.latent_steps,
                past_key_values=p.memory.get_kv_cache(),
            )
            p.memory.update_kv_cache(r.kv_cache)
        cfg = p.pool.activate("Judger")
        e = tok(
            p.executor.build_prompt(cfg, QUESTION, ""),
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        iid = e["input_ids"].to(dev)
        cache = p.memory.get_kv_cache()
        n = get_cache_length(cache)
        mask = torch.ones((1, n + iid.shape[1]), dtype=torch.long, device=dev)
        with torch.no_grad():
            out = s.model(
                input_ids=iid,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        return out.logits[0, -1].float().clone(), n

    l0, prefix = judger_logits()
    with torch.no_grad():
        for n, q in s.model.named_parameters():
            if "lora_B" in n and a.probe_adapter in n:
                q.normal_(0, 0.004)
    l1, _ = judger_logits()
    d_logit = (l1 - l0).abs().max().item()

    print("\n3. TRANSFER — does the expert's latent work reach the Judger?")
    print(f"     latent prefix: {prefix} tokens")
    print(f"     max|Δ judger logit| = {d_logit:.3e}  [{'ok' if d_logit > 1e-3 else 'FAIL'}]")
    print(
        f"     top-1 token: {tok.decode([int(l0.argmax())])!r} -> "
        f"{tok.decode([int(l1.argmax())])!r}"
    )
    if d_logit <= 1e-3:
        print("     FAIL: the judger is not attending to the latent working memory")
        ok = False

    print(f"\n{'=' * 62}\n  chain validation: {'PASS' if ok else 'FAIL'}\n{'=' * 62}")
    print(
        "  Note: a live pathway is not the same as a useful one. These checks\n"
        "  prove signal flows; whether it improves answers is what run_eval.py\n"
        "  measures, and untrained adapters make part 2 vacuous in practice.\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
