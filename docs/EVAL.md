# Evaluation Pipeline — Settings Reference

Complete reference for `run_eval.py` and the `eval/` package: every method, every
dataset, every flag, and what the output means.

```bash
python run_eval.py --list-methods --list-datasets     # what is available
python run_eval.py --methods all --dry-run            # exercise everything, no GPU
python run_eval.py --methods core --dataset medqa --fractions 0.1,0.5,1.0
```

---

## Table of contents

1. [Design](#1-design)
2. [Methods](#2-methods)
3. [Datasets](#3-datasets)
4. [Data segmentation (10% / 50% / 100%)](#4-data-segmentation)
5. [All CLI settings](#5-all-cli-settings)
6. [Per-method overrides (`--set`)](#6-per-method-overrides)
7. [Plots](#7-plots)
8. [Metrics](#8-metrics)
9. [Output files](#9-output-files)
10. [Recipes](#10-recipes)
11. [Reproducibility](#11-reproducibility)
12. [Known confounds in this repo](#12-known-confounds-in-this-repo)
13. [Extending](#13-extending)

---

## 1. Design

One rule drives the layout: **anything that could differ between methods and
change the score is handled by the runner, not by the method.** Segmentation,
seeding, self-consistency voting, option permutation, answer extraction, scoring
and statistics are applied identically to `baseline-cot` and to `latent-mas-kv`.
A method only has to turn one item into one sample. Any difference in the
numbers is therefore attributable to the method itself.

```
run_eval.py                CLI  ->  EvalConfig
  eval/config.py           every knob, hashed into a run fingerprint
  eval/data.py             loaders -> EvalItem -> segment()
  eval/backends.py         hf | system | mock   (one loaded at a time)
  eval/methods/            independently selectable methods
  eval/runner.py           sampling, voting, scoring, resume
  eval/metrics.py          CIs, McNemar, calibration, classification
  eval/report.py           console + markdown report
```

Backends are grouped and loaded one at a time, so comparing a bare-model
baseline against the PEFT-wrapped multi-agent system never needs two copies of a
7B model in VRAM.

---

## 2. Methods

Select with `--methods a,b,c`. Each is fully independent; run one or all.

| Method | Backend | What it is | Why it is in the set |
|---|---|---|---|
| `baseline-direct` | hf | Bare model, answer-only prompt, no reasoning | Absolute floor; shows how much reasoning buys |
| `baseline-cot` | hf | Bare model, reason-first CoT prompt | The honest floor — the default reference |
| `baseline-judger` | hf | Bare model with the repo's Judger prompt | Isolates the prompt from the pipeline |
| `baseline-loglik` | hf | Option log-likelihood scoring, no generation | Extraction-free; no parse failures possible |
| `text-mas` | system | `pipeline="hierarchical"`, every agent decodes text | Classic multi-agent control |
| `latent-mas` | system | true_latent **as originally shipped**: cache discarded, answer-first judger prompt | The system as it was |
| `latent-mas-kv` | system | KV handoff only, legacy prompt | Isolates the cache fix |
| `latent-mas-slora` | system | Switchable role adapters + KV handoff **+ reason-first judger prompt** | LatentMAS-SLoRA: the working configuration |
| `multi-lora` | system | Probe every resident adapter, merge top-k for this instance, one prompt + latent steps | The recommended architecture (see §7a) |
| `sequential-mas` | system | `pipeline="sequential"` chain-of-agents | Alternative topology |
| `router-only` | none | Semantic router domain classification, no generation | Router quality, measured separately and cheaply |

`latent-mas-paper` is accepted as an alias for `latent-mas-slora`, so older
commands and scripts keep working.

**Groups** (usable anywhere a name is):

| Group | Expands to |
|---|---|
| `all` | every method |
| `baselines` | direct, cot, judger, loglik |
| `mas` | text-mas, latent-mas, latent-mas-kv, sequential-mas |
| `latent` | latent-mas, latent-mas-kv, latent-mas-slora |
| `ladder` | baseline-cot, baseline-judger, latent-mas, latent-mas-kv, latent-mas-slora |
| `recommended` | baseline-cot, latent-mas-slora, multi-lora |
| `core` | baseline-cot, baseline-judger, text-mas, latent-mas, latent-mas-kv |

`baseline-loglik` skips non-multiple-choice items (they are recorded as
`UNKNOWN` with `skipped` in `extra`). `router-only` is scored against the
dataset's `domain` label, not against the answer.

---

## 3. Datasets

`--dataset <spec>`:

| Spec | Content | Task type |
|---|---|---|
| `sample` | `data/sample_data.json`, the repo's 5 questions | mcq |
| `local:PATH` | any local `.json` / `.jsonl` | inferred |
| `medqa` | MedQA-USMLE 4-option, test (1,273 items) | mcq |
| `medmcqa` | MedMCQA, validation | mcq |
| `pubmedqa` | PubMedQA `pqa_labeled`, yes/no/maybe | mcq |
| `mmlu` / `mmlu:anatomy` | MMLU, `:subject` picks a config | mcq |
| `mmlu_pro` | MMLU-Pro, 10 options (25% → 10% guess floor) | mcq |
| `arc` / `arc:ARC-Easy` | ARC-Challenge | mcq |
| `gsm8k` | GSM8K grade-school math | numeric |
| `math500` | MATH-500 | numeric |
| `gpqa` / `gpqa:gpqa_diamond` | GPQA (gated — needs `HF_TOKEN`) | mcq |
| `mix:medqa,gsm8k` | concatenation of any of the above | mixed |

`--split` overrides the split. Gated or rate-limited downloads need a token:

```bash
export HF_TOKEN=hf_...        # set it in your shell, never in a committed file
```

The runner logs in automatically when `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`)
is present.

**Local file format** — any of these shapes work:

```json
[
  {"id": 1, "domain": "medical",
   "question": "stem...\nA. one\nB. two\nC. three\nD. four",
   "gold_letter": "C"},
  {"question": "stem...", "choices": ["one","two"], "answer": "B", "domain": "math"},
  {"question": "What is 2+2?", "answer": "4", "task_type": "numeric"}
]
```

Choices embedded in the question text are parsed out automatically, which is
what makes option permutation possible on files like `data/sample_data.json`.

---

## 4. Data segmentation

The segment is a **pure function** of `(dataset, data_seed, fraction, offset,
limit, shuffle, stratify_by)`. Order of operations:

```
shuffle(seed) -> offset -> fraction -> limit
```

| Flag | Default | Meaning |
|---|---|---|
| `--fraction F` | `1.0` | Evaluate this fraction, e.g. `0.1` = 10%, `0.001` = 0.1%. A percentage works too: `0.1%`. Any value > 0 is valid — a fraction that rounds below one item still evaluates one item |
| `--fractions A,B,C` | – | Run one complete eval per fraction, then draw `scaling.png` |
| `--min-per-group N` | `1` | Items each stratum keeps at tiny fractions; `0` lets a stratum drop out |
| `--limit N` | none | Hard cap on item count, applied last |
| `--offset N` | `0` | Skip N items first (disjoint slices for a held-out set) |
| `--data-seed N` | `0` | Seed for the segment shuffle |
| `--no-shuffle` | off | Keep dataset order — slices become the first N items |
| `--stratify-by F` | `domain` | Keep per-group proportions; `none` disables |

**Fractions are nested.** For a fixed seed the 10% slice is a subset of the 50%
slice, which is a subset of 100%. A larger fraction only *adds* items, so a
scaling curve reflects added data rather than a swapped sample:

```bash
python run_eval.py --methods core --dataset medqa --fractions 0.1,0.5,1.0
# -> eval_runs/scan-frac0.1/, scan-frac0.5/, scan-frac1.0/
```

Stratification keeps a mixed set mixed: 10% of a medical+math+code mix stays
medical+math+code rather than collapsing onto whichever domain the shuffle put
first.

Some useful combinations:

```bash
--fraction 0.1                          # 10%, stratified, seeded
--fraction 0.001                        # 0.1% - 1 item per domain on MedQA
--fraction 0.1%                         # the same thing, written as a percentage
--fraction 0.001 --min-per-group 0      # 0.1% without the per-domain floor
--limit 50                              # exactly 50 items
--offset 200 --limit 200                # items 200-399: a disjoint second slice
--fraction 0.1 --data-seed 1            # a different 10% for a variance check
--no-shuffle --limit 20                 # the dataset's own first 20 (debugging)
--stratify-by source --dataset mix:medqa,gsm8k   # balance across sources
```

---

## 5. All CLI settings

### Discovery

| Flag | Default | Effect |
|---|---|---|
| `--list-methods` | – | Print methods and groups, exit |
| `--list-datasets` | – | Print dataset specs, exit |
| `--print-config` | – | Print the resolved config as JSON, exit |
| `--report-only RUN_DIR` | – | Rebuild reports from an existing run, exit (no model) |

### Model

| Flag | Default | Effect |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | Base model for every method |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32`, `4bit` |
| `--cache-dir` | `/home/caches` | HF cache location |

### Data

`--dataset --split --fraction --fractions --limit --offset --data-seed
--no-shuffle --stratify-by` — see §4.

### Methods

| Flag | Default | Effect |
|---|---|---|
| `--methods` | `baseline-cot` | Comma list of names or groups |
| `--set M.KEY=VAL` | – | Per-method override, repeatable (§6) |

### Decoding

| Flag | Default | Effect |
|---|---|---|
| `--max-new-tokens` | `512` | Generation budget for the answering agent |
| `--temperature` | `0.0` | `0.0` = greedy = deterministic |
| `--top-p` | `0.9` | Nucleus cutoff when sampling |
| `--seeds` | `0` | Comma list; every item is run once per seed |
| `--self-consistency K` | `1` | K samples per item, majority-voted |

`--self-consistency > 1` with `--temperature 0.0` produces K identical samples;
the CLI warns. Use `--temperature 0.7`.

### Multi-agent

| Flag | Default | Effect |
|---|---|---|
| `--latent-steps N` | `10` | Latent reasoning iterations per agent |
| `--agents A,B,C` | none | Force a fixed pipeline instead of routing |
| `--no-router` | off | Disable semantic routing (use the general pipeline) |
| `--no-adaptive-steps` | off | Do not vary latent steps by routed domain |
| `--loras a,b` | none | Registry LoRAs to load into the system backend |
| `--kv-handoff` / `--no-kv-handoff` | per method | Hand the latent cache to the decoder, or discard it |
| `--prompt-style` | `reason_first` | Judger prompt order; `reason_first` matches the reference |
| `--adapter-policy` | `logo` | How `multi-lora` composes adapters: `logo`, `merge`, `same`, `none` |
| `--top-k N` | `3` | Adapters merged per instance by `multi-lora` |

Default per-domain pipelines and latent steps:

| Domain | Pipeline | Steps |
|---|---|---|
| medical | Planner → MedicalExpert → Critic → Judger | 15 |
| math | Planner → MathExpert → Critic → Judger | 12 |
| code | Planner → CodeExpert → Critic → Judger | 10 |
| reasoning | Planner → Critic → Refiner → Judger | 12 |
| general | Planner → Critic → Refiner → Judger | 8 |

Registry LoRAs available to `--loras`: `medical_reasoner`, `medical_instruct`,
`math_instruct`, `coder_7b`, `reasoning_lora` (see `src/lora/adapter_manager.py`).

### Protocol

| Flag | Default | Effect |
|---|---|---|
| `--scoring` | `generate` | `generate` or `loglikelihood` (passed to methods that support it) |
| `--permute-options` | `none` | `none`, `cyclic` (n variants), `all` (n! variants) |

`--permute-options cyclic` multiplies runtime by the number of choices and
produces the `position_consistency` metric: the fraction of items whose chosen
*option* (not letter) survives moving the answer around.

### Output

| Flag | Default | Effect |
|---|---|---|
| `--out-dir` | `eval_runs` | Parent directory for runs |
| `--run-name` | timestamp+fingerprint | Run directory name |
| `--no-resume` | off | Ignore existing records instead of continuing them |
| `--no-generations` | off | Do not store model text (smaller records) |
| `--no-markdown` | off | Skip `report.md` |
| `--no-plots` | off | Skip the comparison figures |
| `--live-plot` | off | Refresh `live.png` while the eval runs |
| `--live-every N` | `5` | Records between live refreshes |
| `--scaling-from DIR` | – | Build `scaling.png` from every run under DIR, then exit |
| `--progress N` | `1` | Print a progress line every N items; `0` silences it |
| `-v, --verbose` | off | Per-item lines and full tracebacks |

### Analysis

| Flag | Default | Effect |
|---|---|---|
| `--bootstrap N` | `2000` | Bootstrap resamples; `0` disables |
| `--ci C` | `0.95` | Confidence level |
| `--compare-to M` | first baseline present | Reference method for paired tests |

### Execution

| Flag | Default | Effect |
|---|---|---|
| `--dry-run` | off | Mock backend: no weights, no GPU, full pipeline |
| `--config PATH` | – | Load a saved `config.json`, CLI args override it |

---

## 6. Per-method overrides

`--set method.key=value` configures one method without touching the others.
Values are parsed as JSON when possible, so numbers, booleans and lists work.

```bash
# same run, two latent depths, everything else identical
python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
    --set latent-mas-kv.latent_steps=0

# force a pipeline for one method only
--set latent-mas.agents='["Planner","Judger"]'

# turn routing off for one method only
--set text-mas.use_router=false

# score option text rather than the bare letter
--set baseline-loglik.loglik_mode=letter
```

| Key | Methods | Meaning |
|---|---|---|
| `latent_steps` | mas | Latent iterations; setting it also disables adaptive steps |
| `agents` | mas | Fixed agent list (JSON array) |
| `use_router` | mas | Enable/disable routing |
| `adaptive_latent_steps` | mas | Vary steps by routed domain |
| `loglik_mode` | baseline-loglik | `option` (default) or `letter` |
| `router_model` | router-only | Sentence-transformer name |
| `use_embeddings` | router-only | `false` = keyword-only routing |
| `confidence_threshold` | router-only | Below this, fall back to `general` |

---

## 7. Plots

Written to the run directory at the end of every run (`--no-plots` disables),
and refreshed during the run with `--live-plot`.

| File | Question it answers |
|---|---|
| `accuracy.png` | Which method is more accurate, and is the gap bigger than the interval? Bars with 95% Wilson intervals, reference method in grey |
| `efficiency.png` | What does a point of accuracy cost? Accuracy vs tokens per item, marker area ∝ median latency |
| `latency.png` | p50 / p95 seconds per item |
| `by_domain.png` | Where the difference comes from — one panel per domain |
| `deltas.png` | Paired difference vs the reference, with CI and McNemar p |
| `calibration.png` | Reliability diagram — does the vote share predict correctness? Needs `--self-consistency > 1` |
| `scaling.png` | Accuracy vs fraction of data. Written automatically after `--fractions`, or built later with `--scaling-from` |
| `live.png` | Running accuracy per method while the eval is still going |

Figures follow one rule: colour encodes the data's job, not the series index.
Magnitude comparisons use a single blue hue with the reference method in
emphasis grey; the paired-delta chart uses a blue/red diverging pair around
zero; only the genuinely multi-series line charts use the categorical ramp in
fixed slot order. Label positions are resolved against the *rendered* text
extents, so labels never collide however close two methods land.

```bash
# watch it happen, refreshing every 3 records
python run_eval.py --methods core --dataset medqa --fraction 0.1 --live-plot --live-every 3

# scaling curve across three slice sizes
python run_eval.py --methods core --dataset medqa --fractions 0.001,0.01,0.1

# rebuild the curve later from whatever runs exist
python run_eval.py --scaling-from eval_runs
```

Plot failures never abort a run: the figures are derived from
`records.jsonl`, and `--report-only` regenerates them.

---

## 7a. The ablation ladder and the composed method

**The ladder.** Each rung changes exactly one thing from the rung below, so a
paired McNemar test attributes the difference to that one change:

| rung | `kv_handoff` | `prompt_style` |
|---|---|---|
| `latent-mas` | ✗ | answer_first |
| `latent-mas-kv` | ✓ | answer_first |
| `latent-mas-slora` | ✓ | reason_first |

```bash
python run_eval.py --methods ladder --dataset medqa --fraction 0.1 --max-new-tokens 2048
```

Both knobs also exist in `src/` (`HierarchicalPipeline(kv_handoff=...)`,
`AgentConfig.judger(prompt_style=...)`), defaulting to the reference behaviour,
with the legacy behaviour reachable rather than deleted.

**`multi-lora`** composes adapters instead of moving state between them:

```
probe every resident adapter -> score (activation norm or entropy)
-> top-k -> weighted merge -> ONE weight set -> latent steps -> decode
```

One prompt, one weight set, one KV cache. This matters because a cache computed
under adapter A is not valid input for adapter B — LoRA modifies the Q/K/V
projections from the first token — so the sequential design's cache handoff
becomes unsound the moment adapters are real. Composition sidesteps it entirely.
Selection follows LoGo (arXiv 2511.07129) and is training-free. It shares
`baseline-cot`'s prompt on purpose, so `multi-lora` vs `baseline-cot` isolates
exactly one thing: adapter composition plus latent depth.

**It reports when it is a no-op.** Zero-initialised LoRA adapters are identity
functions: every adapter produces identical activations, selection degenerates
to a uniform merge, and the run measures the base model while looking like it
measured a mixture. When that happens the probe prints a warning and sets
`degenerate_probe: true` on every record. Until adapters are actually trained,
treat any `multi-lora` result as a base-model result.

---

## 8. Metrics

Per method, in `report.md` and `summary.json`:

| Metric | Meaning |
|---|---|
| `accuracy` | Exact match after voting |
| `ci` | Wilson score interval — well behaved at small n |
| `bootstrap_ci` | Percentile bootstrap over items |
| `parse_failure_rate` | Fraction where extraction found nothing. **Read this before the accuracy.** A high rate means you are measuring formatting, not reasoning |
| `strict_accuracy` | Accuracy counting only answers the model emitted in the requested `\boxed{}` format |
| `format_violation_rate` | Fraction where the model answered, but not in the requested format — e.g. `\boxed{A}` on a numeric item. **A gap between `accuracy` and `strict_accuracy` is a prompt bug, not a reasoning result** |
| `unknown_rate` | Fraction predicted `UNKNOWN` |
| `tokens.*` | Prompt / completion / total, mean and sum |
| `tokens.total_per_correct` | Tokens spent per correct answer — the efficiency metric that decides whether a pipeline is worth it |
| `latency_ms` | mean, p50, p95, total |
| `calibration.ece` / `.brier` | From the self-consistency vote share. Meaningful only with `--self-consistency > 1` |
| `by_domain` | Per-domain accuracy with intervals |
| `per_seed_accuracy` | Accuracy per seed — run-to-run spread is often larger than the effect being claimed |
| `position_consistency` | With `--permute-options`, fraction of items answered the same way regardless of where the answer sits |
| `extract_rules` | Which extraction rule fired, and how often |

### Strict vs flexible extraction

Following lm-evaluation-harness's GSM8K convention, every answer is scored twice:

- **strict** — the model emitted the format the prompt asked for (`rule = boxed`)
- **flexible** — the answer was recovered from prose after a malformed box

`parse_failure_rate` counts only items where *nothing* could be extracted.
It does not catch the more common failure: the model solves the problem, states
the answer correctly in prose, and then boxes the wrong kind of token. That
shows up in `format_violation_rate`.

This is not theoretical. In `eval_runs/20260905-195324-458ba3227aec`,
`latent-mas-slora` scored 76.9% against CoT's 96.2% with `parse fail 0.0%`. Four
of its six errors were `\boxed{A}` emitted on GSM8K items whose prose contained
the right number, because the Judger template said "(the option letter for
multiple choice)" on every task. The reasoning was fine; the format instruction
was wrong. The fix is per-task answer formats (`ANSWER_FORMAT` in
`src/agents/configs.py`); `format_violation_rate` is what makes the next
occurrence visible instead of silent.

Flexible recovery is a diagnostic, not a repair. Its last-number fallback
returns 2 for "3 loaves of bread cost $4 more than 2 bagels".

Paired comparisons against the reference method:

| Field | Meaning |
|---|---|
| `delta`, `ci_low`, `ci_high` | Accuracy difference with a **paired** bootstrap CI |
| `mcnemar.a_only` / `b_only` | Items won / lost against the reference |
| `mcnemar.p_value` | Exact binomial McNemar test |

McNemar is the correct test here because both methods answer the same items;
comparing two independent confidence intervals instead will hide real
differences. `router-only` reports `accuracy`, `macro_f1`, per-label
precision/recall and a confusion matrix instead.

---

## 9. Output files

```
eval_runs/<run-name>/
├── config.json      resolved settings + fingerprint (re-runnable with --config)
├── segment.json     exact item ids evaluated
├── records.jsonl    one row per item × method × seed × permutation
├── summary.json     all metrics and comparisons
├── report.txt       the console table
├── report.md        the report you keep
└── *.png            comparison figures (see §7)
```

`records.jsonl` is the source of truth — `summary.json`, `report.txt` and
`report.md` are all derived from it, and `--report-only <run_dir>` regenerates
them without touching a model. Each record holds votes, vote counts, token
counts, latency, the extraction rule, the routed domain, per-agent breakdown
and (unless `--no-generations`) the raw text.

Runs resume by default: re-running the same `--run-name` skips
`(method, seed, record_id)` triples already present, so an interrupted 1,273-item
eval continues where it stopped. If the settings changed since those records were
written, resume **refuses** rather than mixing two configurations into one
`records.jsonl` — use a new `--run-name`, or `--no-resume` to start over.

Progress prints one line per item by default:

```
  [baseline-cot s0]   27/127  medqa-786    A vs D       FAIL  acc  44.4%   12.7s    756 tok  eta 21m  no-parse
```

A multi-agent item takes 10-30s, so anything sparser is indistinguishable from a
hung process. `--progress 10` thins it out; `--progress 0` silences it. The
`no-parse` flag marks an item where answer extraction found nothing — if it
appears often, raise `--max-new-tokens` before trusting the accuracy.

---

## 10. Recipes

**Check the harness with no GPU** — seconds, exercises every code path:
```bash
python run_eval.py --methods all --dataset sample --dry-run
```

**Smallest real run** — one item per domain, both backends, all figures, a few
minutes on one 3090:
```bash
python run_eval.py --methods baseline-cot,baseline-judger,baseline-loglik,latent-mas,latent-mas-kv \
    --dataset medqa --fraction 0.001 --max-new-tokens 400 --live-plot
```

**The core comparison, 10% of MedQA, deterministic:**
```bash
python run_eval.py --methods core --dataset medqa --fraction 0.1 \
    --temperature 0.0 --max-new-tokens 512
```

**Does latent collaboration do anything?** (the mechanism ablation)
```bash
python run_eval.py --methods latent-mas,latent-mas-kv --dataset medqa --fraction 0.1 \
    --compare-to latent-mas
python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
    --run-name steps0 --set latent-mas-kv.latent_steps=0
```

**Latent-depth sweep:**
```bash
for s in 0 5 10 15 25; do
  python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
      --run-name steps-$s --set latent-mas-kv.latent_steps=$s
done
```

**Self-consistency with real vote diversity:**
```bash
python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
    --self-consistency 5 --temperature 0.7
```

**Variance check across seeds:**
```bash
python run_eval.py --methods baseline-cot,latent-mas --dataset medqa \
    --fraction 0.1 --seeds 0,1,2 --temperature 0.7
```

**Position-bias audit:**
```bash
python run_eval.py --methods baseline-cot,latent-mas --dataset medqa \
    --limit 100 --permute-options cyclic
```

**Extraction-free accuracy** (removes regex parsing from the measurement):
```bash
python run_eval.py --methods baseline-loglik --dataset medqa --fraction 1.0
```

**Router quality alone** — no generation, so run it on everything:
```bash
python run_eval.py --methods router-only --dataset mix:medqa,gsm8k --fraction 1.0
```

**Do the trained LoRAs help?**
```bash
python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
    --run-name no-lora
python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
    --run-name with-lora --loras medical_reasoner
```

**Data-scaling curve:**
```bash
python run_eval.py --methods core --dataset medqa --fractions 0.1,0.5,1.0
```

**Re-report an old run:**
```bash
python run_eval.py --report-only eval_runs/20260904-120000-ab12cd34ef56
```

---

## 11. Reproducibility

- `config.json` carries a `fingerprint` — a hash of every setting that affects
  results (output paths excluded). Same fingerprint + same model = same run.
- Re-run any past configuration with `--config eval_runs/<run>/config.json`,
  overriding individual flags on the command line.
- `--temperature 0.0` gives greedy decoding; `--seeds` fixes Python, torch and
  CUDA seeds before every sample.
- `segment.json` records the exact item ids, so a segment can be reconstructed
  even if the upstream dataset changes.
- Report a headline number as accuracy **with its interval and n**, and any
  method-vs-method claim with the McNemar p-value. On 5 items, the 95% interval
  around 60% runs roughly 23%–88% — n that small cannot separate two methods.

Rough sizing on one RTX 3090 (24GB, Qwen2.5-7B, bf16, 512 new tokens): a bare
baseline is ~4-5s per item, `latent-mas` ~10s, `latent-mas-kv` ~16s. So 10% of
MedQA (127 items) is roughly 10 minutes per baseline and 20-35 minutes per
multi-agent method; multiply by seeds, by self-consistency samples, and by the
number of choices if you permute options.

---

## 12. Known confounds in this repo

Two properties of `src/` change how the results should be read. The pipeline
measures both rather than hiding them.

**1. In `true_latent`, the latent state never reaches the answer.**
`src/pipelines/hierarchical.py` accumulates a KV cache in `LatentMemory`, but
the final `model.generate()` call ([hierarchical.py:292](../src/pipelines/hierarchical.py))
receives only `input_ids` and `attention_mask` — never `past_key_values`. The
Judger re-encodes its own prompt from scratch, so `latent-mas` is arithmetically
*the bare model with the Judger prompt*, plus the cost of the discarded latent
passes. `latent-mas-kv` runs the same latent loop and then decodes conditioned on
that cache. **`latent-mas` vs `latent-mas-kv` is therefore the ablation that says
whether latent collaboration does anything at all**, and `baseline-judger` is the
control that both must beat.

**2. The role LoRA adapters are untrained.**
`LoRASpec.to_peft_config()` ([configs.py:37](../src/agents/configs.py)) builds a
fresh `LoraConfig`; PEFT zero-initializes the B matrix, so each adapter is an
identity function. Unless `--loras` loads real weights from the registry, all
seven "specialists" are the same base model differing only by prompt. Any
S-LoRA specialization claim needs the `--loras` condition.

---

## 13. Extending

**A new method** — subclass `Method`, implement `sample()`, add it to `_ALL` in
`eval/methods/__init__.py`. It becomes `--methods <name>` with no other changes:

```python
class MyMethod(Method):
    name = "my-method"
    backend_kind = "hf"          # hf | system | mock | none
    description = "One line, shown by --list-methods."

    def sample(self, item, gen):
        prompt = self.backend.chat_prompt("You are...", item.question)
        out = self.backend.generate(prompt, gen, num_choices=item.num_choices)
        return self._finish(out, item)          # extraction + scoring handled for you
```

`self.args` holds this method's `--set` overrides merged over the global
defaults. Override `gold_of()` / `task_type_of()` if the method is scored
against something other than the answer (as `router-only` is).

**A new dataset** — add a branch to `load_hf()` in `eval/data.py` returning
`EvalItem`s, and a line in `HF_DATASETS`. Set `domain` so stratification,
per-domain breakdowns and router scoring work.

**Tests** — `venv/bin/python -m pytest tests/ -q` covers extraction rules,
segment nesting and determinism, permutation gold remapping, and the statistics.
