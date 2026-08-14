# Routing eval report — router = `semantic`

- Query set: `/workspace/latent_mas_slora/eval/routing/queries.jsonl`  (n = 180)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 56.7%** (102/180) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **42.8%** (77/180)
- Abstention rate: 0.6% (1/180) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 45 | 11 | 24.4% |
| out_of_domain | 15 | 0 | 0.0% |
| single_signal | 120 | 66 | 55.0% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 42 | 80.0% | 28.6% | 12 | 3 | 30 |
| math | 38 | 90.9% | 26.3% | 10 | 1 | 28 |
| medical | 29 | 63.6% | 96.6% | 28 | 16 | 1 |
| finance | 25 | 23.4% | 100.0% | 25 | 82 | 0 |
| reasoning | 31 | 100.0% | 6.5% | 2 | 0 | 29 |
| general | 15 | 0.0% | 0.0% | 0 | 1 | 15 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 12 | 0 | 4 | 26 | 0 | 0 |
| **math** | 1 | 10 | 5 | 22 | 0 | 0 |
| **medical** | 0 | 0 | 28 | 1 | 0 | 0 |
| **finance** | 0 | 0 | 0 | 25 | 0 | 0 |
| **reasoning** | 2 | 1 | 6 | 19 | 2 | 1 |
| **general** | 0 | 0 | 1 | 14 | 0 | 0 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 4.532 ms | median: 4.507 ms | p95: 4.653 ms | max: 5.345 ms

## Errors

103 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
