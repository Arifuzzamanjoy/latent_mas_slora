# Routing eval report — router = `staged`

- Query set: `/sessions/jolly-busy-dijkstra/mnt/latent_mas_slora/eval/routing/queries.jsonl`  (n = 180)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 8.3%** (15/180) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **62.8%** (113/180)
- Abstention rate: 37.2% (67/180) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 45 | 17 | 37.8% |
| out_of_domain | 15 | 15 | 100.0% |
| single_signal | 120 | 81 | 67.5% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 42 | 96.4% | 64.3% | 27 | 1 | 15 |
| math | 38 | 76.0% | 50.0% | 19 | 6 | 19 |
| medical | 29 | 91.3% | 72.4% | 21 | 2 | 8 |
| finance | 25 | 85.0% | 68.0% | 17 | 3 | 8 |
| reasoning | 31 | 82.4% | 45.2% | 14 | 3 | 17 |
| general | 15 | 22.4% | 100.0% | 15 | 52 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 27 | 4 | 0 | 0 | 1 | 10 |
| **math** | 1 | 19 | 1 | 3 | 1 | 13 |
| **medical** | 0 | 0 | 21 | 0 | 0 | 8 |
| **finance** | 0 | 2 | 0 | 17 | 1 | 5 |
| **reasoning** | 0 | 0 | 1 | 0 | 14 | 16 |
| **general** | 0 | 0 | 0 | 0 | 0 | 15 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 0.075 ms | median: 0.066 ms | p95: 0.126 ms | max: 0.385 ms

## Errors

67 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
