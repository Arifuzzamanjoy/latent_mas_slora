# Routing comparison — baseline `fast` vs `staged`

Query set: `queries.jsonl` (n = 180). All values read directly from each router's `results.json`.

## 1. Confident-and-wrong (headline)

The rate at which the router commits to a *specialist* domain (no abstention) and is wrong. These are silent failures — a wrong adapter would load with no signal.

| | baseline `fast` | staged `staged` | delta |
|---|---:|---:|---:|
| **confident-and-wrong** | 15.6% (28/180) | 8.3% (15/180) | **-7.2 pp** |

## 2. Top-1 accuracy

| bucket | baseline `fast` | staged `staged` | delta |
|---|---:|---:|---:|
| **overall** | 55.6% | 62.8% | **+7.2 pp** |
| dual_signal | 40.0% | 37.8% | -2.2 pp |
| out_of_domain | 100.0% | 100.0% | +0.0 pp |
| single_signal | 55.8% | 67.5% | +11.7 pp |

## 3. Abstention rate

| | baseline `fast` | staged `staged` | delta |
|---|---:|---:|---:|
| routed to `general` | 37.2% (67/180) | 37.2% (67/180) | +0.0 pp |

## 4. Per-domain precision / recall

| domain | support | P fast | P staged | R fast | R staged |
|---|---:|---:|---:|---:|---:|
| code | 42 | 89.7% | 96.4% | 61.9% | 64.3% |
| math | 38 | 77.3% | 76.0% | 44.7% | 50.0% |
| medical | 29 | 73.3% | 91.3% | 75.9% | 72.4% |
| finance | 25 | 63.0% | 85.0% | 68.0% | 68.0% |
| reasoning | 31 | 60.0% | 82.4% | 9.7% | 45.2% |
| general | 15 | 22.4% | 22.4% | 100.0% | 100.0% |

## 5. Latency (mean ms/query)

| baseline `fast` | staged `staged` |
|---:|---:|
| 0.058 | 0.086 |

## Why confident-and-wrong is the metric that matters more here

Top-1 accuracy treats every error the same. But in this system a routing error is not neutral: the routed domain decides which LoRA adapter gets loaded and which agent pipeline runs. When the baseline keyword router picks the *wrong specialist* with no hesitation, that is a **silent** failure — the pipeline confidently serves a query from the wrong expert and nothing in the system flags it. The staged router adds an explicit *unsure* outcome: when neither the cheap keyword pass nor the costlier semantic pass clears its floor, it routes to `general`/abstain instead of forcing a specialist. The effect is that confident-and-wrong falls from 15.6% to 8.3% (-7.2 pp): 13 queries that were previously mis-served with confidence are now either corrected by the second stage or turned into visible abstentions. Overall top-1 accuracy moves +7.2 pp at the same time, so the reduction in silent failures did not come at the cost of accuracy. A silent wrong route is worse than a visible 'I'm not sure' — the latter can be escalated, logged, or sent to a default pipeline; the former is discovered only when the answer is already wrong.
