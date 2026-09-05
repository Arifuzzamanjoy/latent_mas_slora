# Eval report — `postfix-gsm8k`

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
| `fingerprint` | `5f593521dc51` |

Segment: **n=26**, domains `{'math': 26}`, sources `{'gsm8k': 26}`. Wall clock 449.3s.

## Results

| method | n | accuracy | strict | 95% CI | parse fail | fmt viol | tokens/item | tokens/correct | latency p50 | latency p95 | ECE |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-cot` | 26 | **96.2%** | 96.2% | [81.1%, 99.3%] | 0.0% | 0.0% | 395 | 411.2 | 5711 | 8131 | 0.0385 |
| `latent-mas-slora` | 26 | **96.2%** | 96.2% | [81.1%, 99.3%] | 0.0% | 0.0% | 883 | 918.0 | 10391 | 14751 | 0.0385 |

## Paired comparisons vs `baseline-cot`

McNemar's exact test on the items both methods answered. `wins` = correct here and wrong for the reference.

| method | Δ accuracy | 95% CI of Δ | wins | losses | discordant | p |
|---|---:|---|---:|---:|---:|---:|
| `latent-mas-slora` | +0.0% | [+0.0%, +0.0%] | 0 | 0 | 0 | 1.0 |

## Per-domain accuracy

| method | math |
|---|---:|
| `baseline-cot` | 96.2% (26) |
| `latent-mas-slora` | 96.2% (26) |

## Robustness

| method | per-seed accuracy | position consistency | unknown rate | errors |
|---|---|---:|---:|---:|
| `baseline-cot` | `{'0': 0.9615}` | - | 0.0% | 0 |
| `latent-mas-slora` | `{'0': 0.9615}` | - | 0.0% | 0 |

---

Records: `records.jsonl` (one row per item × method × seed × permutation). Config: `config.json`. Regenerate this report with `python run_eval.py --report-only <run_dir>`.
