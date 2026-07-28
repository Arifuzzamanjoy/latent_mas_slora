# Routing eval report — router = `fast`

- Query set: `/sessions/jolly-busy-dijkstra/mnt/latent_mas_slora/eval/routing/queries.jsonl`  (n = 180)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 15.6%** (28/180) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **55.6%** (100/180)
- Abstention rate: 37.2% (67/180) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 45 | 18 | 40.0% |
| out_of_domain | 15 | 15 | 100.0% |
| single_signal | 120 | 67 | 55.8% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 42 | 89.7% | 61.9% | 26 | 3 | 16 |
| math | 38 | 77.3% | 44.7% | 17 | 5 | 21 |
| medical | 29 | 73.3% | 75.9% | 22 | 8 | 7 |
| finance | 25 | 63.0% | 68.0% | 17 | 10 | 8 |
| reasoning | 31 | 60.0% | 9.7% | 3 | 2 | 28 |
| general | 15 | 22.4% | 100.0% | 15 | 52 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 26 | 1 | 2 | 5 | 1 | 7 |
| **math** | 1 | 17 | 3 | 3 | 1 | 13 |
| **medical** | 0 | 0 | 22 | 0 | 0 | 7 |
| **finance** | 0 | 2 | 0 | 17 | 0 | 6 |
| **reasoning** | 2 | 2 | 3 | 2 | 3 | 19 |
| **general** | 0 | 0 | 0 | 0 | 0 | 15 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 0.058 ms | median: 0.056 ms | p95: 0.091 ms | max: 0.131 ms

## Errors

80 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
