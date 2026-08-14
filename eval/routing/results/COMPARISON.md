# Routing comparison — four-way

Query set: `queries.jsonl` (n = 180). Every value is read from each router's `results.json` or from `cost_profile.json`; nothing is hand-entered.

## 1. Confident-and-wrong (headline)

Committed to a *specialist* domain (no abstention) and was wrong — a **silent** failure, since the routed domain selects which adapter/pipeline runs.

| router | confident-and-wrong | vs `fast` |
|---|---:|---:|
| `fast` | 15.6% (28/180) | — |
| `staged` | 8.3% (15/180) | -7.2 pp |
| `semantic` | 56.7% (102/180) | +41.1 pp |
| `advanced` | 14.4% (26/180) | -1.1 pp |

## 2. Top-1 accuracy

| router | overall | dual_signal | out_of_domain | single_signal |
|---|---:|---:|---:|---:|
| `fast` | **55.6%** | 40.0% | 100.0% | 55.8% |
| `staged` | **62.8%** | 37.8% | 100.0% | 67.5% |
| `semantic` | **42.8%** | 24.4% | 0.0% | 55.0% |
| `advanced` | **53.3%** | 26.7% | 100.0% | 57.5% |

## 3. Abstention rate

| router | abstention (routed to `general`) |
|---|---:|
| `fast` | 37.2% (67/180) |
| `staged` | 37.2% (67/180) |
| `semantic` | 0.6% (1/180) |
| `advanced` | 40.6% (73/180) |

## 4. Per-domain precision / recall

| domain | support | P fast | R fast | P staged | R staged | P semantic | R semantic | P advanced | R advanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 42 | 89.7% | 61.9% | 96.4% | 64.3% | 80.0% | 28.6% | 82.4% | 33.3% |
| math | 38 | 77.3% | 44.7% | 76.0% | 50.0% | 90.9% | 26.3% | 88.9% | 42.1% |
| medical | 29 | 73.3% | 75.9% | 91.3% | 72.4% | 63.6% | 96.6% | 75.9% | 75.9% |
| finance | 25 | 63.0% | 68.0% | 85.0% | 68.0% | 23.4% | 100.0% | 66.7% | 88.0% |
| reasoning | 31 | 60.0% | 9.7% | 82.4% | 45.2% | 100.0% | 6.5% | 70.0% | 22.6% |
| general | 15 | 22.4% | 100.0% | 22.4% | 100.0% | 0.0% | 0.0% | 20.5% | 100.0% |

## 5. Cost

| router | mean latency | cold start (import+construct+1st predict) | dependency weight | offline-capable |
|---|---:|---:|---|:--:|
| `fast` | 0.0571 ms | 31.9 ms | stdlib only (no third-party runtime deps) | yes |
| `staged` | 0.0695 ms | 41.4 ms | stdlib only (no third-party runtime deps) | yes |
| `semantic` | 4.5319 ms | — | — | — |
| `advanced` | 44.3693 ms | — | — | — |

## What actually changed

**The gain is entirely in `single_signal`: +15 fixed / −1 broken (net +14).**

**`dual_signal` REGRESSED: 40.0% → 37.8%, 3 fixed / 4 broken (net -1).** This is a regression. The staged router is *worse* at the ambiguous two-signal queries than the plain keyword router, because it abstains on several it previously happened to get right.

**The mechanism is `reasoning` recall: 9.7% → 45.2%.** The keyword router almost never emitted `reasoning` at all — the reasoning profile's keywords ("why", "how", "explain", "compare") are generic words that lose to more specific domain terms, so reasoning queries were absorbed by other domains. Stage 2's TF-IDF pass recovered them. **This, not dual-signal disambiguation, is where the improvement came from.** Anyone reading the headline as "staging resolves ambiguity" is reading it wrong.

### The 5 queries the staged router broke

| id | bucket | query | expected | `fast` | `staged` |
|---|---|---|---|---|---|
| q059 | single_signal | what causes iron deficiency anemia | `medical` | `medical` ✓ | `general` ✗ |
| q125 | dual_signal | calculate the insulin dose for an 80 kg patient at 0.5 uni… | `medical` | `medical` ✓ | `general` ✗ |
| q129 | dual_signal | what are the pros and cons of investing in bitcoin versus … | `finance` | `finance` ✓ | `reasoning` ✗ |
| q159 | dual_signal | is there a logical gap in the induction step of this proof | `math` | `math` ✓ | `general` ✗ |
| q163 | dual_signal | debug why my sql query returns duplicate patient rows | `code` | `code` ✓ | `general` ✗ |

4 of the 5 became abstentions rather than wrong specialist picks, so they cost accuracy but not confident-and-wrong. 1 became a different wrong specialist.

## Why confident-and-wrong is the metric that matters more here

Top-1 accuracy treats every error the same. In this system a routing error is not neutral: the routed domain decides which LoRA adapter loads and which agent pipeline runs. When the keyword router picks the wrong specialist with no hesitation, that is a **silent** failure — the wrong expert answers and nothing flags it. The staged router adds an explicit *unsure* outcome, so confident-and-wrong falls 15.6% → 8.3% (-7.2 pp) while accuracy moves +7.2 pp. A visible "not sure" can be escalated, logged, or sent to a default pipeline; a silent wrong route is discovered only when the answer is already wrong.

## Why staged, given semantic exists

Measured: semantic top-1 42.8% vs staged 62.8%; confident-and-wrong 56.7% vs 8.3%.
