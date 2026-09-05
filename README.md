# LatentMAS + S-LoRA

Latent-space multi-agent reasoning with composable LoRA adapters, and an
evaluation harness built to measure whether any of it actually helps.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What this is

An implementation of [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639)
(LatentMAS) — agents that exchange continuous hidden states instead of text —
extended so each agent can carry its own specialization via a switchable LoRA
adapter.

On the name: this is **[S-LoRA](https://arxiv.org/abs/2311.03285)-inspired, not
S-LoRA**. Adapters are switched with PEFT's `set_adapter()`, one active per hop.
S-LoRA's contributions are unified memory paging and custom kernels for batched
heterogeneous serving; neither is implemented here, and there is no concurrency.
The scheduling idea is borrowed, the serving layer is not.

It ships with a full evaluation harness, and **the harness is the point**. Claims
about multi-agent reasoning are easy to make and hard to substantiate; every
number below came out of `run_eval.py` on this code, and the ones that did not
survive measurement have been removed rather than restated.

```bash
make install-dev
make eval-dry          # exercise everything: no GPU, no weights, seconds
make test              # 65 tests
```

## Measured results

Qwen2.5-7B-Instruct, MedQA, **127 items**, greedy decoding, 50 latent steps,
2048-token budget. `ladder` runs five configurations over identical items so a
paired McNemar test can attribute any difference.

| method | accuracy | 95% CI | output tok | total tok | p50 latency |
|---|---:|---|---:|---:|---:|
| baseline-cot | 57.5% | [48.8, 65.7] | 529 | 828 | 11.7s |
| baseline-judger | 59.1% | [50.4, 67.2] | 228 | 568 | **4.9s** |
| latent-mas *(as originally shipped)* | 59.8% | [51.1, 68.0] | 215 | 1552 | 21.5s |
| latent-mas-kv | 59.1% | [50.4, 67.2] | 263 | 1562 | 21.1s |
| latent-mas-slora *(reference config)* | **61.4%** | [52.7, 69.4] | 519 | 1842 | 36.4s |

Paired, same items:

| comparison | Δ | wins | losses | p |
|---|---:|---:|---:|---:|
| + KV cache handed to decoder | −0.8 | 8 | 9 | **1.000** |
| + reason-first judger prompt | +2.4 | 15 | 12 | 0.701 |
| best pipeline vs plain CoT | +3.9 | 19 | 14 | 0.487 |

**No accuracy difference here is statistically significant.** Read honestly:

- **The latent pipeline as originally shipped is equivalent to a single prompt.**
  `latent-mas` and `baseline-judger` disagreed on 3 of 127 items. The cache was
  built and then dropped before decoding; that is fixed, and fixing it changed
  nothing measurable (p=1.000, replicated on MMLU).
- At n=127 this design resolves roughly a 12pp difference. It rules out a large
  effect, not any effect.
- Comparing output tokens against a single CoT prompt was the wrong axis, and
  this table invites it. `baseline-judger` is one prompt with no agents at all,
  and it reaches 228 output tokens — so a saving measured against `baseline-cot`
  is attributable to the prompt, not to latent collaboration. The comparison the
  paper actually makes is against **text-based MAS**, which is measured below.

### Latent vs text-based multi-agent

The paper's comparator is text-mediated agent communication, not prompting. Same
four agents, same pipeline, same items — the only difference is whether agents
exchange text or hidden state. GSM8K, 26 items, greedy, 4 latent steps
(`eval_runs/all-baselines-gsm8k`):

| method | accuracy | output tok | total tok | p50 latency |
|---|---:|---:|---:|---:|
| text-mas | 96.2% | 1452 | 2418 | 62.7s |
| latent-mas-slora | 92.3% | **236** | **880** | **11.4s** |

**Output tokens −83.7%, total tokens −63.6%, latency 5.5× faster**, at accuracy
that is statistically indistinguishable (1 discordant item of 26, McNemar
p=1.000). The paper reports 70.8–83.7% output-token reduction; this lands at the
top of that range on an independent implementation.

The scope of that result is worth stating plainly: the saving is against text
mediation, not against prompting. Against a single CoT prompt the same pipeline
costs 2.2× the total tokens for no accuracy gain, so latent collaboration is
worth its overhead when you were already paying for a multi-agent system — not
as a replacement for one well-posed prompt.

| baseline (GSM8K, same 26 items) | accuracy |
|---|---:|
| baseline-direct | 23.1% |
| baseline-judger | 34.6% |
| baseline-cot | 96.2% |
| text-mas | 96.2% |
| latent-mas-slora | 92.3% |

`baseline-direct` and `baseline-judger` are answer-first prompts and are floors
on a chain-of-thought dataset, not controls. Measured against them the pipeline
"gains" 57–69pp, which is a statement about chain-of-thought rather than about
this work; `baseline-cot` and `text-mas` are the comparisons that mean
something.

Two conditions have never been tested, because the variable was pinned:
**the role adapters are untrained**, so all seven are identity functions and the
adapter layer currently switches between identical weights (see
[Known limitations](#known-limitations)).

## Install

```bash
git clone <repo> && cd latent_mas_slora
make install-dev          # or: pip install -r requirements-dev.txt
export HF_TOKEN=...       # only for gated datasets; never commit it
```

## Quick start

```python
from src import LatentMASSystem, AgentConfig

system = LatentMASSystem(model_name="Qwen/Qwen2.5-7B-Instruct", latent_steps=50)
for cfg in (
    AgentConfig.planner(),
    AgentConfig.medical(),
    AgentConfig.critic(),
    AgentConfig.judger(),
):
    system.add_agent(cfg)

result = system.run(
    question="...",
    pipeline="true_latent",  # only the final agent emits tokens
    agents=["Planner", "MedicalExpert", "Critic", "Judger"],
)
print(result.final_answer)
```

## Evaluating

```bash
# the ablation ladder: each rung changes exactly one thing
python run_eval.py --methods ladder --dataset medqa --fraction 0.1 --max-new-tokens 2048

# a single method on 10% of a dataset, with a live plot
python run_eval.py --methods latent-mas-slora --dataset medqa --fraction 0.1 --live-plot

# data-scaling curve
python run_eval.py --methods core --dataset medqa --fractions 0.01,0.1,1.0
```

11 methods, 11 dataset loaders, nested fractional slices, paired statistics and
eight plot types. **[docs/EVAL.md](docs/EVAL.md) is the full settings reference.**

## Verifying the chain

The multi-agent chain makes three claims that fail independently and silently:
a different adapter is active at each hop, that adapter changes the computation,
and its work reaches the agent that answers.

```bash
make validate     # or: python tools/validate_chain.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

```
1. SWITCH    planner_lora -> medical_lora -> critic_lora -> judger_lora   [ok]
2. APPLY     max|Δ hidden| = 75.5                                          [ok]
3. TRANSFER  max|Δ judger logit| = 2.125 through an 812-token prefix       [ok]
```

Run it after any change to the pipeline, the agent pool or the adapter manager.

## Project layout

```
src/
├── system.py                 LatentMASSystem — the entry point
├── core/
│   ├── latent_memory.py      shared latent working memory + KV cache
│   └── latent_reasoner.py    realignment matrix, continuous reasoning loop
├── agents/
│   ├── configs.py            role configs, prompts, LoRA specs
│   └── agent_pool.py         registration and adapter switching
├── lora/adapter_manager.py   external adapter loading, merging, registry
├── pipelines/
│   ├── hierarchical.py       hierarchical + true-latent pipelines
│   └── sequential.py         chain-of-agents
└── routing/                  semantic router, domain profiles

eval/                         evaluation harness (see docs/EVAL.md)
tools/validate_chain.py       chain switch/apply/transfer validator
tests/                        65 tests, no GPU or network required
run_eval.py                   evaluation CLI
```

## Known limitations

These are measured, not hypothetical. Read them before trusting any result.

1. **The role adapters are untrained.** `get_peft_model()` zero-initializes
   `lora_B`, so every adapter is an exact identity: switching between them
   produces bit-identical logits (max difference 0.0). Adapter switching works
   mechanically and contributes nothing until adapters are trained.
2. **Cross-adapter KV transfer is unsound once adapters differ.** LoRA modifies
   the Q/K/V projections from the first token, so a cache written under one
   adapter is not valid input for another. Invisible today only because the
   adapters are identical. See Activated-LoRA (2512.17910), LRAgent (2602.01053),
   ForkKV (2604.06370); `multi-lora` sidesteps it by composing adapters into one
   weight set instead of switching mid-chain.
3. **Role decomposition may not help at this scale.** Solo Performance Prompting
   (2307.05300) reports that multi-persona synergy emerges at GPT-4 scale and is
   absent in smaller models. The reference paper's gains are on Qwen3-4B/8B/14B;
   this runs Qwen2.5-7B. An untested backbone change is the cheapest remaining
   explanation for the flat ladder.
4. **`--max-new-tokens` applies to every agent**, not just the final one, which
   inflates `text-mas` cost and makes its token counts non-comparable.
5. **Router accuracy is measured on three domains, not five.** There is no
   `code` dataset in `eval/data.py`, so `code` and `general` routing are
   exercised only by hand-written probes. The 96.2% below is math / medical /
   reasoning; treat the other two as untested.
6. **Router gold labels are a repo convention.** `arc` is labelled `reasoning`
   and `gsm8k` is labelled `math` by the loaders. That is a reasonable choice,
   not ground truth about which pipeline answers best — the router is scored
   against the labels, not against downstream accuracy.

### Fixed, and worth knowing about

**The semantic router was returning a near-uniform posterior.** Four compounding
faults, each measurable on 450 labelled items (150 GSM8K / MedQA / ARC):

| stage | accuracy | macro-F1 |
|---|---:|---:|
| as shipped | 51.3% | 0.384 |
| + keyword hygiene | 63.3% | 0.408 |
| + word boundaries, centred softmax | 71.6% | 0.436 |
| + math word-problem exemplars | 85.6% | 0.518 |
| + reasoning exemplars, tiebreaker removed | **96.2%** | **0.722** |

1. `"if "`, `"for "`, `"while "`, `"class"` and `"return"` were `code` keywords.
   They fire on 164/400 and 127/400 GSM8K word problems — *"**If** there are 30
   sheets…"*, *"…**for** his drawing"*.
2. Keywords matched as substrings, so `"sin"` fired on "u**sin**g", `"iv"` on
   "g**iv**e"/"f**iv**e", `"log"` on "bio**log**y", `"tan"` on "impor**tan**t".
3. `_semantic_score` returned `(cos + 1) / 2`. Cosines against these centroids
   live in roughly [0.0, 0.35], so adding a constant 0.5 to all five domains and
   normalizing compressed every confidence into a 0.22–0.28 band. Because that
   band sits entirely below the `confidence < 0.30` guard in `get_best_domain`,
   the keyword-rescoring tiebreaker fired on **199 of 200 items** — the keyword
   score, with `"if "` in it, silently decided every route.
4. `math` exemplars were all symbolic (`"Find the derivative of sin(x²)"`), so
   the centroid sat far from grade-school arithmetic.

Scores are now centred before a temperature-0.10 softmax, keyword evidence is a
bounded `tanh` nudge rather than 40% of the blend, and the tiebreaker is gone.
Confidence is calibrated — accuracy by confidence bin runs 64% / 86% / 99% / 99%
— so the `GENERAL` fallback threshold means something again.

**Answer formats are per task, not per role** (`ANSWER_FORMAT` in
`src/agents/configs.py`). The Judger template used to end with "(the option
letter for multiple choice)" on every item, inherited from MedQA. On GSM8K the
model would solve the problem, state the right number in prose, and then emit
`\boxed{A}` — scored as a confident wrong answer. It caused four of six errors
in `eval_runs/20260905-195324-458ba3227aec` and cost ~15pp of apparent accuracy.

The harness reported `parse fail 0.0%` throughout, because `extract_numeric()`
treated a box holding no number as a successful extraction. Both are fixed:
role prompts are task-neutral and the format instruction is appended from the
item's `task_type`, and every result now carries `strict_accuracy` and
`format_violation_rate` alongside `accuracy` (see [docs/EVAL.md](docs/EVAL.md)).
Any future run where the model answers in the wrong format shows it in the
table instead of losing it into the accuracy column.

## Development

```bash
make help          # list all targets
make test          # full suite
make lint          # ruff check + format check
make fmt           # apply fixes
make clean
```

CI runs lint and tests on Python 3.10 and 3.12. The suite needs neither torch nor
network, so it finishes in seconds.

## Citation

```bibtex
@article{latentmas2025,
  title  = {Latent Collaboration in Multi-Agent Systems},
  journal= {arXiv preprint arXiv:2511.20639},
  year   = {2025}
}
@inproceedings{slora2024,
  title    = {S-LoRA: Serving Thousands of Concurrent LoRA Adapters},
  booktitle= {MLSys},
  year     = {2024}
}
```

## License

MIT — see [LICENSE](LICENSE).
