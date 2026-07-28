# Routing eval report — router = `staged`

- Query set: `/sessions/jolly-busy-dijkstra/mnt/latent_mas_slora/eval/routing/queries.jsonl`  (n = 40)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 5.0%** (2/40) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **67.5%** (27/40)
- Abstention rate: 37.5% (15/40) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 10 | 5 | 50.0% |
| out_of_domain | 4 | 4 | 100.0% |
| single_signal | 26 | 18 | 69.2% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 10 | 100.0% | 50.0% | 5 | 0 | 5 |
| math | 8 | 83.3% | 62.5% | 5 | 1 | 3 |
| medical | 6 | 100.0% | 83.3% | 5 | 0 | 1 |
| finance | 5 | 100.0% | 80.0% | 4 | 0 | 1 |
| reasoning | 7 | 80.0% | 57.1% | 4 | 1 | 3 |
| general | 4 | 26.7% | 100.0% | 4 | 11 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 5 | 1 | 0 | 0 | 1 | 3 |
| **math** | 0 | 5 | 0 | 0 | 0 | 3 |
| **medical** | 0 | 0 | 5 | 0 | 0 | 1 |
| **finance** | 0 | 0 | 0 | 4 | 0 | 1 |
| **reasoning** | 0 | 0 | 0 | 0 | 4 | 3 |
| **general** | 0 | 0 | 0 | 0 | 0 | 4 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 0.073 ms | median: 0.066 ms | p95: 0.110 ms | max: 0.142 ms

## Errors

13 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
