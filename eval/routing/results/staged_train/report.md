# Routing eval report — router = `staged`

- Query set: `/sessions/jolly-busy-dijkstra/mnt/latent_mas_slora/eval/routing/queries.jsonl`  (n = 140)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 9.3%** (13/140) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **61.4%** (86/140)
- Abstention rate: 37.1% (52/140) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 35 | 12 | 34.3% |
| out_of_domain | 11 | 11 | 100.0% |
| single_signal | 94 | 63 | 67.0% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 32 | 95.7% | 68.8% | 22 | 1 | 10 |
| math | 30 | 73.7% | 46.7% | 14 | 5 | 16 |
| medical | 23 | 88.9% | 69.6% | 16 | 2 | 7 |
| finance | 20 | 81.2% | 65.0% | 13 | 3 | 7 |
| reasoning | 24 | 83.3% | 41.7% | 10 | 2 | 14 |
| general | 11 | 21.2% | 100.0% | 11 | 41 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 22 | 3 | 0 | 0 | 0 | 7 |
| **math** | 1 | 14 | 1 | 3 | 1 | 10 |
| **medical** | 0 | 0 | 16 | 0 | 0 | 7 |
| **finance** | 0 | 2 | 0 | 13 | 1 | 4 |
| **reasoning** | 0 | 0 | 1 | 0 | 10 | 13 |
| **general** | 0 | 0 | 0 | 0 | 0 | 11 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 0.075 ms | median: 0.072 ms | p95: 0.118 ms | max: 0.154 ms

## Errors

54 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
