# Routing eval report — router = `fast`

- Query set: `/sessions/jolly-busy-dijkstra/mnt/latent_mas_slora/eval/routing/queries.jsonl`  (n = 140)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 15.7%** (22/140) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **55.7%** (78/140)
- Abstention rate: 36.4% (51/140) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 35 | 15 | 42.9% |
| out_of_domain | 11 | 11 | 100.0% |
| single_signal | 94 | 52 | 55.3% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 32 | 91.7% | 68.8% | 22 | 2 | 10 |
| math | 30 | 76.5% | 43.3% | 13 | 4 | 17 |
| medical | 23 | 69.6% | 69.6% | 16 | 7 | 7 |
| finance | 20 | 61.9% | 65.0% | 13 | 8 | 7 |
| reasoning | 24 | 75.0% | 12.5% | 3 | 1 | 21 |
| general | 11 | 21.6% | 100.0% | 11 | 40 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 22 | 1 | 2 | 3 | 0 | 4 |
| **math** | 1 | 13 | 3 | 3 | 1 | 9 |
| **medical** | 0 | 0 | 16 | 0 | 0 | 7 |
| **finance** | 0 | 2 | 0 | 13 | 0 | 5 |
| **reasoning** | 1 | 1 | 2 | 2 | 3 | 15 |
| **general** | 0 | 0 | 0 | 0 | 0 | 11 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 0.086 ms | median: 0.068 ms | p95: 0.209 ms | max: 0.473 ms

## Errors

62 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
