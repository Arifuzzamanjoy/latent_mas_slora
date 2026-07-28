# Routing comparison — four-way

Query set: `queries.jsonl` (n = 180). Every value is read from each router's `results.json` or from `cost_profile.json`; nothing is hand-entered.

> **semantic, advanced could not be run in this environment.** They are reported as NOT MEASURED below, with the exact reason. No values were substituted or estimated for them.

## 1. Confident-and-wrong (headline)

Committed to a *specialist* domain (no abstention) and was wrong — a **silent** failure, since the routed domain selects which adapter/pipeline runs.

| router | confident-and-wrong | vs `fast` |
|---|---:|---:|
| `fast` | 15.6% (28/180) | — |
| `staged` | 8.3% (15/180) | -7.2 pp |
| `semantic` | NOT MEASURED | — |
| `advanced` | NOT MEASURED | — |

## 2. Top-1 accuracy

| router | overall | dual_signal | out_of_domain | single_signal |
|---|---:|---:|---:|---:|
| `fast` | **55.6%** | 40.0% | 100.0% | 55.8% |
| `staged` | **62.8%** | 37.8% | 100.0% | 67.5% |
| `semantic` | NOT MEASURED | — | — | — |
| `advanced` | NOT MEASURED | — | — | — |

## 3. Abstention rate

| router | abstention (routed to `general`) |
|---|---:|
| `fast` | 37.2% (67/180) |
| `staged` | 37.2% (67/180) |
| `semantic` | NOT MEASURED |
| `advanced` | NOT MEASURED |

## 4. Per-domain precision / recall

| domain | support | P fast | R fast | P staged | R staged |
|---|---:|---:|---:|---:|---:|
| code | 42 | 89.7% | 61.9% | 96.4% | 64.3% |
| math | 38 | 77.3% | 44.7% | 76.0% | 50.0% |
| medical | 29 | 73.3% | 75.9% | 91.3% | 72.4% |
| finance | 25 | 63.0% | 68.0% | 85.0% | 68.0% |
| reasoning | 31 | 60.0% | 9.7% | 82.4% | 45.2% |
| general | 15 | 22.4% | 100.0% | 22.4% | 100.0% |

## 5. Cost

| router | mean latency | cold start (import+construct+1st predict) | dependency weight | offline-capable |
|---|---:|---:|---|:--:|
| `fast` | 0.0571 ms | 31.9 ms | stdlib only (no third-party runtime deps) | yes |
| `staged` | 0.0695 ms | 41.4 ms | stdlib only (no third-party runtime deps) | yes |
| `semantic` | NOT MEASURED | NOT MEASURED | torch wheel 526.6 MB + sentence-transformers + transformers | **no** (downloads model on first init) |
| `advanced` | NOT MEASURED | NOT MEASURED | torch wheel 526.6 MB + sentence-transformers + transformers | **no** (downloads model on first init) |

`semantic`, `advanced` could not be measured here. Measured facts behind that:

- torch wheel is **526.6 MB** (`pip download of torch-2.13.0-cp310-cp310-manylinux_2_28_x86_64.whl`), and observed pypi throughput was ~1.1 MB/s.
- `403 Forbidden` — https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2
- `403 Forbidden` — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
- `403 Forbidden` — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors

Because the model weights, `config.json`, **and** the hub metadata endpoint are all blocked, `SentenceTransformer('all-MiniLM-L6-v2')` cannot initialise, so neither neural router can produce a prediction here. Their accuracy is **unknown**, not zero and not assumed.

`run_eval.py` also refuses to *silently* degrade: both `SemanticRouter` and `AdvancedHybridRouter` catch `ImportError` internally and fall back to keyword-only scoring, which would otherwise be reported as a 'semantic' result. The harness now asserts the encoder and domain centroids actually loaded and raises `_NeuralNotLoaded` if not.

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

**This is the honest answer: I do not know, because I could not run them here.** The embedding routers were never benchmarked — `huggingface.co` is blocked in this environment, so the model cannot be fetched (§5). It is entirely possible the semantic router beats the staged router on accuracy; a fair reading is that this comparison is **incomplete**, and the staged-vs-fast result should not be presented as "staged is the best router".

What can be defended without those numbers is narrower and is about **cost and gateability**, not quality: the staged router has no third-party runtime dependency, starts in tens of milliseconds, routes in well under a millisecond, and runs fully offline — so it can gate every pull request on a standard CPU runner in seconds. The neural routers need a ~527 MB torch wheel plus a model download, which is a different class of CI dependency. That is an argument about what is cheap to *gate on*, not an argument that staging produces better routing. Running `--router semantic` and `--router advanced` on a networked machine is the obvious next step and would settle it.
