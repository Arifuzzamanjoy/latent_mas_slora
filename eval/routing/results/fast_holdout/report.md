# Routing eval report — router = `fast`

- Query set: `/sessions/jolly-busy-dijkstra/mnt/latent_mas_slora/eval/routing/queries.jsonl`  (n = 40)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 15.0%** (6/40) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **55.0%** (22/40)
- Abstention rate: 40.0% (16/40) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 10 | 3 | 30.0% |
| out_of_domain | 4 | 4 | 100.0% |
| single_signal | 26 | 15 | 57.7% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 10 | 80.0% | 40.0% | 4 | 1 | 6 |
| math | 8 | 80.0% | 50.0% | 4 | 1 | 4 |
| medical | 6 | 85.7% | 100.0% | 6 | 1 | 0 |
| finance | 5 | 66.7% | 80.0% | 4 | 2 | 1 |
| reasoning | 7 | 0.0% | 0.0% | 0 | 1 | 7 |
| general | 4 | 25.0% | 100.0% | 4 | 12 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 4 | 0 | 0 | 2 | 1 | 3 |
| **math** | 0 | 4 | 0 | 0 | 0 | 4 |
| **medical** | 0 | 0 | 6 | 0 | 0 | 0 |
| **finance** | 0 | 0 | 0 | 4 | 0 | 1 |
| **reasoning** | 1 | 1 | 1 | 0 | 0 | 4 |
| **general** | 0 | 0 | 0 | 0 | 0 | 4 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 0.060 ms | median: 0.057 ms | p95: 0.084 ms | max: 0.094 ms

## Errors

18 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
