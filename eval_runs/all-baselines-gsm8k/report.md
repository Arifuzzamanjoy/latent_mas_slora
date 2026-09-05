# Eval report — `all-baselines-gsm8k`

## Configuration

| setting | value |
|---|---|
| `model` | `Qwen/Qwen2.5-7B-Instruct` |
| `dtype` | `bfloat16` |
| `device` | `cuda` |
| `dataset` | `gsm8k` |
| `split` | `None` |
| `fraction` | `0.02` |
| `limit` | `None` |
| `offset` | `0` |
| `data_seed` | `0` |
| `shuffle` | `True` |
| `stratify_by` | `domain` |
| `temperature` | `0.0` |
| `top_p` | `0.9` |
| `max_new_tokens` | `512` |
| `seeds` | `[0]` |
| `self_consistency` | `1` |
| `scoring` | `generate` |
| `permute_options` | `none` |
| `latent_steps` | `10` |
| `agents` | `None` |
| `use_router` | `True` |
| `adaptive_latent_steps` | `True` |
| `kv_handoff` | `True` |
| `loras` | `[]` |
| `bootstrap` | `2000` |
| `ci` | `0.95` |
| `fingerprint` | `002a181d87c6` |

Segment: **n=26**, domains `{'math': 26}`, sources `{'gsm8k': 26}`. Wall clock 2289.5s.

## Results

| method | n | accuracy | strict | 95% CI | parse fail | fmt viol | tokens/item | tokens/correct | latency p50 | latency p95 | ECE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-direct` | 26 | **23.1%** | 23.1% | [11.0%, 42.0%] | 0.0% | 0.0% | 114 | 494.5 | 165 | 251 | 0.7692 |
| `baseline-cot` | 26 | **96.2%** | 96.2% | [81.1%, 99.3%] | 0.0% | 0.0% | 395 | 411.2 | 5834 | 8463 | 0.0385 |
| `baseline-judger` | 26 | **34.6%** | 34.6% | [19.4%, 53.8%] | 0.0% | 0.0% | 420 | 1214.8 | 4838 | 9146 | 0.6538 |
| `text-mas` | 26 | **96.2%** | 96.2% | [81.1%, 99.3%] | 0.0% | 0.0% | 2418 | 2514.6 | 61809 | 77225 | 0.0385 |
| `latent-mas-slora` | 26 | **92.3%** | 92.3% | [75.9%, 97.9%] | 0.0% | 0.0% | 880 | 953.8 | 11419 | 15786 | 0.0769 |

## Paired comparisons vs `baseline-cot`

McNemar's exact test on the items both methods answered. `wins` = correct here and wrong for the reference.

| method | Δ accuracy | 95% CI of Δ | wins | losses | discordant | p |
|---|---:|---|---:|---:|---:|---:|
| `baseline-direct` | -73.1% | [-88.5%, -53.8%] | 0 | 19 | 19 | 4e-06 |
| `baseline-judger` | -61.5% | [-80.8%, -42.3%] | 0 | 16 | 16 | 3.1e-05 |
| `text-mas` | +0.0% | [+0.0%, +0.0%] | 0 | 0 | 0 | 1.0 |
| `latent-mas-slora` | -3.8% | [-11.5%, +0.0%] | 0 | 1 | 1 | 1.0 |

## Per-domain accuracy

| method | math |
|---|---:|
| `baseline-direct` | 23.1% (26) |
| `baseline-cot` | 96.2% (26) |
| `baseline-judger` | 34.6% (26) |
| `text-mas` | 96.2% (26) |
| `latent-mas-slora` | 92.3% (26) |

## Robustness

| method | per-seed accuracy | position consistency | unknown rate | errors |
|---|---|---:|---:|---:|
| `baseline-direct` | `{'0': 0.2308}` | - | 0.0% | 0 |
| `baseline-cot` | `{'0': 0.9615}` | - | 0.0% | 0 |
| `baseline-judger` | `{'0': 0.3462}` | - | 0.0% | 0 |
| `text-mas` | `{'0': 0.9615}` | - | 0.0% | 0 |
| `latent-mas-slora` | `{'0': 0.9231}` | - | 0.0% | 0 |

---

Records: `records.jsonl` (one row per item × method × seed × permutation). Config: `config.json`. Regenerate this report with `python run_eval.py --report-only <run_dir>`.
