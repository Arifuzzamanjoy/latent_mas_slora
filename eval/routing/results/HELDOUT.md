# Held-out evaluation

The originally shipped thresholds were tuned in-sample on all 180 queries. This document redoes it properly: tune on a train split only, then report the resulting config on a holdout that was never used for tuning.

## The split

`split.json` — deterministic **140/40** train/holdout split of the 180 ids, seed 1234, stratified jointly by `bucket` and `expected_domain` (largest-remainder allocation, shuffled within each stratum). The split is **recorded in the file**, not regenerated at run time, so it is stable across runs and machines.

| | n | single | dual | OOD |
|---|---:|---:|---:|---:|
| train | 140 | 94 | 35 | 11 |
| holdout | 40 | 26 | 10 | 4 |

Reproduce with `--split train` / `--split holdout` (default remains `all`, so prior behaviour is unchanged).

## Train-tuned config

Sweep run on **train only** (140 queries), recorded in `staged/threshold_sweep_train.json`. Selection rule is the one already documented: lowest confident-and-wrong among configs whose accuracy is at least the `fast` baseline on the same train split (baseline: acc 55.7%, cw 15.7%).

**Train-tuned pick: `stage1_accept=0.7`, `stage2_accept=0.18`.**

This is **identical to the shipped defaults**. The in-sample tuning happened to select the same configuration that train-only tuning selects, so nothing had to change and no cascade was needed. That is a favourable accident, not evidence the original method was sound — the original sweep genuinely did see the whole set.

## Train vs holdout

| split | n | router | top-1 acc | confident-and-wrong | abstention |
|---|---:|---|---:|---:|---:|
| train | 140 | `fast` (baseline) | 55.7% | 15.7% (22/140) | 36.4% |
| train | 140 | `staged` | **61.4%** | **9.3%** (13/140) | 37.1% |
| holdout | 40 | `fast` (baseline) | 55.0% | 15.0% (6/40) | 40.0% |
| holdout | 40 | `staged` | **67.5%** | **5.0%** (2/40) | 37.5% |

## The gap (the honest generalisation estimate)

| metric | train | holdout | gap |
|---|---:|---:|---:|
| top-1 accuracy | 61.4% | 67.5% | +6.1 pp |
| confident-and-wrong | 9.3% | 5.0% | -4.3 pp |

The holdout is **better** than train on both metrics (+6.1 pp accuracy, -4.3 pp confident-and-wrong). Do not read that as the thresholds generalising unusually well. **The holdout is only 40 queries**, so confident-and-wrong there is 2/40 against 22/140 — a difference of a handful of queries. One or two flips move the holdout rate by 2.5 pp. The right conclusion is the weaker one: **the direction holds on data not used for tuning**, and the magnitude is not precisely estimated.

## Holdout vs baseline

On the untouched holdout, staged beats the `fast` baseline by **+12.5 pp** top-1 accuracy and **-10.0 pp** confident-and-wrong (6 → 2 silent errors out of 40). This is the number to quote when asked whether the improvement is real or an artifact of tuning.

## What this does and does not fix

- **Fixed:** thresholds are no longer selected using the queries they are scored on. The train/holdout boundary is recorded and reproducible.
- **Not fixed:** the holdout is 40 hand-written queries labelled by one person. A clean split of a biased sample is still a biased sample. Held-out performance on *this* set says nothing about performance on real production traffic.
- **Not fixed:** the grid was swept once. There is no repeated cross-validation, so the train numbers themselves carry selection optimism.

## Reproduce

```bash
python eval/routing/run_eval.py --router fast   --split train
python eval/routing/run_eval.py --router fast   --split holdout
python eval/routing/run_eval.py --router staged --split train
python eval/routing/run_eval.py --router staged --split holdout
```
