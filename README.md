# LatentMAS + S-LoRA

Latent-space multi-agent reasoning with switchable LoRA adapters, plus the evaluation harness built to find out whether it helps.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An independent implementation of [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639), where agents pass continuous hidden states to each other instead of text, extended so each agent carries its own specialization through a switchable adapter.

Every number here came out of `run_eval.py` on this code. The claims that did not survive measurement were deleted rather than softened.

```bash
make install-dev
make eval-dry     # exercises everything: no GPU, no weights, a few seconds
make test         # 65 tests
```

## What it does

Against text-mediated multi-agent reasoning, latent collaboration is much cheaper at the same accuracy.

GSM8K, 26 items, Qwen2.5-7B-Instruct, greedy, 4 latent steps, same four agents and same pipeline in both rows. The only variable is whether agents exchange text or hidden state (`eval_runs/all-baselines-gsm8k`).

| method | accuracy | output tok | total tok | p50 latency |
|---|---:|---:|---:|---:|
| text-mas | 96.2% | 1452 | 2418 | 62.7s |
| latent-mas-slora | 92.3% | **236** | **880** | **11.4s** |

Output tokens down 83.7%, total tokens down 63.6%, latency 5.5x faster. The accuracy gap is one discordant item out of 26 (McNemar p=1.000). The paper reports 70.8 to 83.7% output-token reduction; this lands at the top of that range on a separate implementation.

Scope matters here. The saving is against text mediation, not against prompting. Plain chain-of-thought scores 96.2% on the same items, and the latent pipeline costs 2.2x its total tokens for no accuracy gain. Latent collaboration pays for itself when you were already running a multi-agent system. It does not beat one well-posed prompt.

## What it does not do

The adapter ladder is flat, and it is worth saying so up front.

MedQA, 127 items, 50 latent steps, 2048-token budget. `ladder` runs five configurations over identical items so a paired McNemar test can attribute any difference.

| method | accuracy | 95% CI | output tok | total tok | p50 latency |
|---|---:|---|---:|---:|---:|
| baseline-cot | 57.5% | [48.8, 65.7] | 529 | 828 | 11.7s |
| baseline-judger | 59.1% | [50.4, 67.2] | 228 | 568 | **4.9s** |
| latent-mas *(as originally shipped)* | 59.8% | [51.1, 68.0] | 215 | 1552 | 21.5s |
| latent-mas-kv | 59.1% | [50.4, 67.2] | 263 | 1562 | 21.1s |
| latent-mas-slora *(reference config)* | **61.4%** | [52.7, 69.4] | 519 | 1842 | 36.4s |

| paired comparison | delta | wins | losses | p |
|---|---:|---:|---:|---:|
| KV cache handed to decoder | -0.8 | 8 | 9 | **1.000** |
| reason-first judger prompt | +2.4 | 15 | 12 | 0.701 |
| best pipeline vs plain CoT | +3.9 | 19 | 14 | 0.487 |

No accuracy difference in that table is statistically significant. Three things follow:

The pipeline as originally shipped was equivalent to a single prompt. `latent-mas` and `baseline-judger` disagreed on 3 of 127 items. The cache was being built and then dropped before decoding. That is fixed, and fixing it changed nothing measurable (p=1.000, replicated on MMLU).

At n=127 the design resolves roughly a 12pp difference. It rules out a large effect, not any effect.

Comparing output tokens against a single CoT prompt was the wrong axis, and the table invites it. `baseline-judger` has no agents at all and still reaches 228 output tokens, so a saving measured against `baseline-cot` belongs to the prompt, not to latent collaboration. The comparison that means something is the text-MAS one above.

The same caution applies to the GSM8K floors. `baseline-direct` (23.1%) and `baseline-judger` (34.6%) are answer-first prompts on a chain-of-thought dataset. Measured against those, the pipeline "gains" 57 to 69pp, which is a fact about chain-of-thought rather than about this work.

## Known limitations

Measured, not hypothetical. Read these before trusting any number above.

1. **The role adapters are untrained.** `get_peft_model()` zero-initializes `lora_B`, so all seven adapters are exact identities and switching between them produces bit-identical logits. The switching machinery works; it contributes nothing until the adapters are trained.
2. **Cross-adapter KV transfer is unsound once adapters differ.** LoRA modifies the Q/K/V projections from the first token, so a cache written under one adapter is not valid input for another. This is invisible today only because the adapters are identical. See Activated-LoRA (2512.17910), LRAgent (2602.01053), ForkKV (2604.06370). `multi-lora` sidesteps it by composing adapters into one weight set instead of switching mid-chain.
3. **Role decomposition may not help at this scale.** Solo Performance Prompting (2307.05300) finds multi-persona synergy at GPT-4 scale and absent in smaller models. The reference paper's gains are on Qwen3-4B/8B/14B; this runs Qwen2.5-7B. An untested backbone change is the cheapest remaining explanation for the flat ladder.
4. **`--max-new-tokens` applies to every agent**, not just the last one, which inflates `text-mas` cost and makes its token counts only roughly comparable.
5. **Router accuracy covers three domains, not five.** There is no `code` dataset in `eval/data.py`, so `code` and `general` routing are exercised only by hand-written probes. Treat those two as untested.
6. **Router gold labels are a repo convention.** The loaders label `arc` as `reasoning` and `gsm8k` as `math`. That is a reasonable choice, not ground truth about which pipeline answers best. The router is scored against the labels, not against downstream accuracy.

## Two bugs worth reading

**The semantic router was returning a near-uniform posterior.** Four compounding faults, each measured on 450 labelled items (150 each from GSM8K, MedQA, ARC).

| stage | accuracy | macro-F1 |
|---|---:|---:|
| as shipped | 51.3% | 0.384 |
| keyword hygiene | 63.3% | 0.408 |
| word boundaries, centred softmax | 71.6% | 0.436 |
| math word-problem exemplars | 85.6% | 0.518 |
| reasoning exemplars, tiebreaker removed | **96.2%** | **0.722** |

The two that mattered most: `_semantic_score` returned `(cos + 1) / 2`, and since cosines against these centroids live in roughly [0.0, 0.35], every confidence was squeezed into a 0.22 to 0.28 band. That band sits entirely below the `confidence < 0.30` guard, so the keyword tiebreaker fired on 199 of 200 items. And the keyword list had `"if "`, `"for "` and `"return"` marked as `code`, which fire on hundreds of GSM8K word problems ("**If** there are 30 sheets..."). The keyword score, with `"if "` in it, was silently deciding every route.

Scores are now centred before a temperature-0.10 softmax, keyword evidence is a bounded `tanh` nudge instead of 40% of the blend, and the tiebreaker is gone. Confidence is calibrated: accuracy by confidence bin runs 64 / 86 / 99 / 99%.

**Answer format was tied to the role instead of the task.** The Judger template ended with "(the option letter for multiple choice)" on every item, inherited from MedQA. On GSM8K the model would solve the problem, state the right number in prose, then emit `\boxed{A}` and get scored wrong. Four of six errors in `eval_runs/20260905-195324-458ba3227aec`, worth about 15pp of apparent accuracy. The harness reported `parse fail 0.0%` throughout, because `extract_numeric()` counted a box with no number in it as a successful extraction.

Both fixed. Role prompts are task-neutral, the format instruction comes from the item's `task_type`, and every result now carries `strict_accuracy` and `format_violation_rate` next to `accuracy`.

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
    pipeline="true_latent",   # only the final agent emits tokens
    agents=["Planner", "MedicalExpert", "Critic", "Judger"],
)
print(result.final_answer)
```

Install with `make install-dev`, or `pip install -r requirements-dev.txt`. Set `HF_TOKEN` only if you need gated datasets, and never commit it.

## Evaluating

```bash
# ablation ladder: each rung changes exactly one thing
python run_eval.py --methods ladder --dataset medqa --fraction 0.1 --max-new-tokens 2048

# one method on 10% of a dataset, live plot
python run_eval.py --methods latent-mas-slora --dataset medqa --fraction 0.1 --live-plot

# data-scaling curve
python run_eval.py --methods core --dataset medqa --fractions 0.01,0.1,1.0
```

11 methods, 11 dataset loaders, nested fractional slices, paired statistics, eight plot types. [docs/EVAL.md](docs/EVAL.md) is the full settings reference.

## Verifying the chain

The multi-agent chain makes three claims that can each fail silently: a different adapter is active at each hop, that adapter changes the computation, and its work reaches the agent that answers.

```bash
make validate
# python tools/validate_chain.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

```
1. SWITCH    planner_lora -> medical_lora -> critic_lora -> judger_lora   [ok]
2. APPLY     max|d hidden| = 75.5                                        [ok]
3. TRANSFER  max|d judger logit| = 2.125 through an 812-token prefix      [ok]
```

Run it after any change to the pipeline, the agent pool or the adapter manager.

## On the name

This is [S-LoRA](https://arxiv.org/abs/2311.03285)-inspired, not S-LoRA. Adapters are switched with PEFT's `set_adapter()`, one active per hop. S-LoRA's actual contributions are unified memory paging and custom kernels for batched heterogeneous serving. Neither is implemented here, and there is no concurrency. The scheduling idea is borrowed; the serving layer is not.

## Project layout

```
src/
├── system.py                 LatentMASSystem, the entry point
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

## Development

```bash
make help     # list all targets
make test     # full suite
make lint     # ruff check + format check
make fmt      # apply fixes
make clean
```

CI runs lint and tests on Python 3.10 and 3.12. The suite needs neither torch nor network, so it finishes in seconds.

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

MIT. See [LICENSE](LICENSE).
