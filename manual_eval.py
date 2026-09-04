#!/usr/bin/env python3
"""
Manual evaluation harness — compare configurations head to head.

Modes:
  baseline-judger : bare model, repo's Judger prompt (answer-first). No agents, no latent, no LoRA.
  baseline-cot    : bare model, reason-first CoT prompt.
  pipeline        : the full true_latent multi-agent pipeline from evaluate_latent_collaboration.py

Usage:
  python manual_eval.py --mode baseline-cot
  python manual_eval.py --mode baseline-judger --temp 0.0
  python manual_eval.py --mode pipeline --sc 3
  python manual_eval.py --mode all
"""

import os, sys, json, re, time, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch

MODEL = os.environ.get("LATENTMAS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
CACHE = os.environ.get("HF_HOME", "/home/caches")

# ─── Prompts ─────────────────────────────────────────────────────────────────

SYS_JUDGER = (
    "You are a Judger Agent responsible for final decisions. "
    "Evaluate all evidence and reasoning to select the best answer. "
    "Be decisive and provide clear justification. "
    "You MUST always end your response with \\boxed{LETTER} where LETTER is A, B, C, or D. "
    "State your final answer early in your reasoning, then justify it."
)
USR_JUDGER = (
    "Make the final decision:\n\nQuestion: {q}\n\n"
    "Based on all analysis, select the best answer.\n"
    "For multiple choice, you MUST format your final answer as: \\boxed{{LETTER}}\n"
    "State your chosen answer letter FIRST, then provide reasoning.\n\nFinal Answer:"
)

SYS_COT = (
    "You are an expert clinician, mathematician and computer scientist. "
    "Reason carefully step by step, then commit to one option."
)
USR_COT = (
    "{q}\n\n"
    "Work through the problem step by step. Consider each option and rule out the wrong ones. "
    "Only after your reasoning is complete, give the final answer on its own last line "
    "as \\boxed{{LETTER}}."
)

# ─── Answer extraction ───────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    if not text:
        return "UNKNOWN"
    sc = re.match(r'\[Self-Consistency Vote:\s*([A-D])\]', text)
    if sc:
        return sc.group(1).upper()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        m = re.search(r'([ABCD])', boxed[-1].strip().upper())
        if m:
            return m.group(1)
    for pat in [
        r'(?:final\s+)?answer\s*(?:is|:)\s*[\*\_]*\s*\(?([ABCD])\)?',
        r'(?:correct\s+)?option\s*(?:is|:)\s*[\*\_]*\s*\(?([ABCD])\)?',
        r'(?:choice\s+)?([ABCD])\s*(?:is\s+correct|is\s+the\s+answer)',
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        letters = re.findall(r'\b([ABCD])\b', lines[-1].upper())
        if letters:
            return letters[-1]
    return "UNKNOWN"


def load_questions():
    path = SCRIPT_DIR / "data" / "sample_data.json"
    with open(path) as f:
        return json.load(f)


def gold_of(item):
    g = item.get("gold_letter", item.get("answer", "")).strip().upper()
    m = re.search(r'([ABCD])', g)
    return m.group(1) if m else g


# ─── Baseline runner ─────────────────────────────────────────────────────────

def run_baseline(mode, questions, temp, max_tokens, sc, verbose):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {MODEL} (bf16)")
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir=CACHE, dtype=torch.bfloat16
    ).to("cuda").eval()
    print(f"[load] VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    sys_p, usr_p = (SYS_JUDGER, USR_JUDGER) if mode == "baseline-judger" else (SYS_COT, USR_COT)

    def ask(q):
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p.format(q=q)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        kw = dict(max_new_tokens=max_tokens, pad_token_id=tok.eos_token_id)
        if temp > 0:
            kw.update(do_sample=True, temperature=temp, top_p=0.9)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        n_in = enc["input_ids"].shape[1]
        new = out[0][n_in:]
        return tok.decode(new, skip_special_tokens=True).strip(), n_in + len(new)

    return _loop(questions, ask, sc, verbose)


# ─── Pipeline runner ─────────────────────────────────────────────────────────

def run_pipeline(questions, temp, max_tokens, sc, verbose):
    from src import LatentMASSystem, AgentConfig, Domain
    from src.routing import SemanticRouter

    DOMAIN_LATENT_STEPS = {
        Domain.MEDICAL: 15, Domain.MATH: 12, Domain.CODE: 10,
        Domain.REASONING: 12, Domain.GENERAL: 8,
    }
    DOMAIN_PIPELINES = {
        Domain.MEDICAL:   ["Planner", "MedicalExpert", "Critic", "Judger"],
        Domain.MATH:      ["Planner", "MathExpert", "Critic", "Judger"],
        Domain.CODE:      ["Planner", "CodeExpert", "Critic", "Judger"],
        Domain.REASONING: ["Planner", "Critic", "Refiner", "Judger"],
        Domain.GENERAL:   ["Planner", "Critic", "Refiner", "Judger"],
    }

    router = SemanticRouter()
    system = LatentMASSystem(model_name=MODEL, device="cuda", dtype="bfloat16", latent_steps=10)
    for a in [AgentConfig.planner(max_tokens=50), AgentConfig.medical(max_tokens=50),
              AgentConfig.math(max_tokens=50), AgentConfig.coder(max_tokens=50),
              AgentConfig.critic(max_tokens=50), AgentConfig.refiner(max_tokens=50),
              AgentConfig.judger(max_tokens=max_tokens)]:
        system.add_agent(a)

    def ask(q):
        domain, conf = router.get_best_domain(q)
        agents = DOMAIN_PIPELINES.get(domain, DOMAIN_PIPELINES[Domain.GENERAL])
        system._pipeline.latent_steps = DOMAIN_LATENT_STEPS.get(domain, 10)
        if verbose:
            print(f"      route={domain.value} ({conf:.1%})  {' -> '.join(agents)}")
        res = system.run(question=q, pipeline="true_latent", agents=agents,
                         max_new_tokens=max_tokens, temperature=temp, self_consistency=1)
        return res.final_answer, res.total_tokens

    return _loop(questions, ask, sc, verbose)


# ─── Shared scoring loop ─────────────────────────────────────────────────────

def _loop(questions, ask, sc, verbose):
    from collections import Counter
    rows = []
    for i, item in enumerate(questions):
        q, gold = item["question"], gold_of(item)
        t0 = time.time()
        votes, toks, texts = [], 0, []
        for _ in range(sc):
            text, n = ask(q)
            votes.append(extract_answer(text))
            texts.append(text)
            toks += n
        pred = Counter(votes).most_common(1)[0][0]
        dt = time.time() - t0
        ok = pred == gold
        rows.append(dict(n=i+1, pred=pred, gold=gold, ok=ok, votes=votes,
                         secs=dt, tokens=toks, text=texts[0]))
        print(f"  Q{i+1}: {pred} vs {gold}  {'PASS' if ok else 'FAIL'}"
              f"  votes={','.join(votes)}  {dt:.1f}s  {toks} tok")
        if verbose:
            print(f"       {texts[0][:300].replace(chr(10), ' ')}\n")
    return rows


def summarize(name, rows):
    ok = sum(r["ok"] for r in rows)
    print(f"\n  {name}: {ok}/{len(rows)} = {100*ok/len(rows):.0f}%   "
          f"avg {sum(r['secs'] for r in rows)/len(rows):.1f}s   "
          f"{sum(r['tokens'] for r in rows)} tokens total")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline-cot",
                    choices=["baseline-judger", "baseline-cot", "pipeline", "all"])
    ap.add_argument("--sc", type=int, default=1, help="self-consistency samples")
    ap.add_argument("--temp", type=float, default=0.0, help="0.0 = greedy")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--verbose", action="store_true", help="print model output")
    ap.add_argument("--save", default=None, help="write results JSON here")
    args = ap.parse_args()

    if args.sc > 1 and args.temp == 0.0:
        print("[warn] --sc > 1 with greedy decoding gives identical samples; "
              "use --temp 0.7 for real voting diversity.\n")

    questions = load_questions()
    print(f"Loaded {len(questions)} questions | mode={args.mode} "
          f"temp={args.temp} sc={args.sc}\n")

    modes = ["baseline-judger", "baseline-cot", "pipeline"] if args.mode == "all" else [args.mode]
    out = {}
    for m in modes:
        print("=" * 70)
        print(m)
        print("=" * 70)
        if m == "pipeline":
            rows = run_pipeline(questions, args.temp, args.max_tokens, args.sc, args.verbose)
        else:
            rows = run_baseline(m, questions, args.temp, args.max_tokens, args.sc, args.verbose)
        summarize(m, rows)
        out[m] = rows
        # Free the 7B weights before loading the next mode (24 GB cannot hold two)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print()

    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
