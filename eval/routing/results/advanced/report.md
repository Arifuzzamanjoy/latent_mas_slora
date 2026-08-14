# Routing eval report — router = `advanced`

- Query set: `/workspace/latent_mas_slora/eval/routing/queries.jsonl`  (n = 180)
- Every number below is emitted from this run into `results.json` in the same directory; nothing is hand-entered.

## Headline

- **CONFIDENT-AND-WRONG rate: 14.4%** (26/180) — committed to a specialist domain, did not abstain, and was wrong. These are the silent failures.
- Top-1 accuracy (overall): **53.3%** (96/180)
- Abstention rate: 40.6% (73/180) — routed to `general` instead of a specialist.

## Top-1 accuracy by bucket

| bucket | n | correct | accuracy |
|---|---:|---:|---:|
| dual_signal | 45 | 12 | 26.7% |
| out_of_domain | 15 | 15 | 100.0% |
| single_signal | 120 | 69 | 57.5% |

## Per-domain precision / recall

| domain | support | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| code | 42 | 82.4% | 33.3% | 14 | 3 | 28 |
| math | 38 | 88.9% | 42.1% | 16 | 2 | 22 |
| medical | 29 | 75.9% | 75.9% | 22 | 7 | 7 |
| finance | 25 | 66.7% | 88.0% | 22 | 11 | 3 |
| reasoning | 31 | 70.0% | 22.6% | 7 | 3 | 24 |
| general | 15 | 20.5% | 100.0% | 15 | 58 | 0 |

## Confusion matrix (rows = expected, cols = predicted)

| exp \ pred | code | math | medical | finance | reasoning | general |
|---|---:|---:|---:|---:|---:|---:|
| **code** | 14 | 1 | 3 | 5 | 1 | 18 |
| **math** | 1 | 16 | 3 | 4 | 1 | 13 |
| **medical** | 0 | 1 | 22 | 0 | 0 | 6 |
| **finance** | 0 | 0 | 0 | 22 | 1 | 2 |
| **reasoning** | 2 | 0 | 1 | 2 | 7 | 19 |
| **general** | 0 | 0 | 0 | 0 | 0 | 15 |

See `confusion_matrix.png` for the screen-share-friendly version.

## Latency (per query, wall clock)

- mean: 44.369 ms | median: 9.009 ms | p95: 92.003 ms | max: 93.978 ms

## Errors

84 misrouted queries listed in `errors.csv` (expected vs predicted vs confidence).
