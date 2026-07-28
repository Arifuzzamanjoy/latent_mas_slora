# Routing evaluation

A reproducible, CPU-runnable evaluation of the query router in this repo, plus a
staged router that reduces **silent** routing failures. Every number in every
artifact here is emitted by an actual run of the code into a `results.json`;
nothing is hand-entered or estimated.

## What this measures

The router classifies an incoming query into one of **6 domains**
(`code, math, medical, finance, reasoning, general`) defined in
`src/routing/domain_profiles.py`. This harness measures how well it does that on a
hand-labelled set of 180 queries, with emphasis on one metric:

- **Confident-and-wrong** (the headline): the router committed to a *specialist*
  domain (did not abstain) and was wrong. This is a **silent** failure — in the
  full system the routed domain selects which adapter/pipeline runs, so a confident
  wrong route serves the query from the wrong expert with no signal that anything
  went wrong.
- **Top-1 accuracy** — overall and split by bucket (single-signal / dual-signal /
  out-of-domain).
- **Abstention rate** — how often it routes to `general` instead of forcing a
  specialist pick. (`general` doubles as the routers' fallback/abstain token.)
- **Per-domain precision / recall**, full **6×6 confusion matrix**, and
  **mean latency**.

## Reproduce in one command

```bash
# baseline (existing FastRouter) and staged, then the side-by-side comparison:
python eval/routing/run_eval.py --router fast
python eval/routing/run_eval.py --router staged
python eval/routing/compare.py            # writes results/COMPARISON.md
```

Each `run_eval.py` run writes to `eval/routing/results/<router>/`:
`results.json`, `report.md`, `confusion_matrix.png`, `errors.csv`.
Runs are **deterministic** (fixed seed, sorted iteration) and **offline** for the
`fast` and `staged` routers (no network, no torch, no model download).

Unit tests for the staged router's threshold / escalation / unsure logic:

```bash
python tests/test_staged_router.py        # runs with or without pytest
```

## The staged router

`src/routing/staged_router.py` is a **new wrapper**; it does not modify or change
the default behaviour of any existing router.

1. **Stage 1 (cheap):** the existing keyword `FastRouter`. If it commits to a
   specialist with confidence ≥ `stage1_accept`, return it.
2. **Stage 2 (costlier):** a local TF-IDF cosine matcher over each domain's
   exemplar prompts + keywords (pure `numpy`-free stdlib, offline). If its best
   specialist scores ≥ `stage2_accept`, return it.
3. **Unsure:** if neither stage clears its floor, return `general` — an explicit
   abstain instead of forcing a pick.

All thresholds live in one `StagedConfig` block at the top of the file.

## Headline result (measured)

| metric | baseline `fast` | staged | delta |
|---|---:|---:|---:|
| confident-and-wrong | 15.6% (28/180) | 8.3% (15/180) | **−7.2 pp** |
| top-1 accuracy | 55.6% | 62.8% | +7.2 pp |

Numbers trace to `results/fast/results.json`, `results/staged/results.json`, and
`results/COMPARISON.md`.

## How the labelled set was built

`queries.jsonl` — 180 lines, each:

```json
{"id":"q001","query":"...","expected_domain":"code",
 "bucket":"single_signal","signals":["code"],"note":"why this label"}
```

- **120 single_signal** — unambiguous, 24 each evenly across the five specialist
  domains (`code, math, medical, finance, reasoning`).
- **45 dual_signal** — genuinely carry two domain signals; `signals` lists both;
  `expected_domain` is the one a careful human would pick and `note` says why.
- **15 out_of_domain** — nothing fits; `expected_domain = "general"`.

Phrasing is deliberately varied, including terse/sloppy queries. There is no
near-duplicate padding; each line is meant to be defensible on its own.

`expected_domain` is labelled against the **6-value `Domain` enum**, i.e. what the
routers actually emit — this is a **domain-routing** eval, not adapter selection.
For the out-of-domain bucket we use the existing `general` enum member (no separate
`"none"` label was added).

## How routing relates to adapters (architecture)

This is deliberately kept straight, in plain language:

- The router classifies a query into **one of 6 domains**. That is all it does.
- **Domain-to-adapter binding is a separate concern**, resolved at deploy time. It
  is *not* implemented as part of this eval, and this eval does not score it.
- The **four adapters currently loadable** from `data/lora_registry.json`
  (`medical_vl`, `reward_vl`, `comics_vl`, `point_detect_vl`) are **general-purpose
  vision adapters** for `Qwen2.5-VL-7B` — three of the four have domain `general`.
  So **today the mapping is not one-domain-one-specialist**: only `medical` has a
  genuinely matching specialist adapter. The GPU validation in `RUNPOD.md` exercises
  adapter *hot-swap mechanics* across all four; it does not claim a specialist per
  domain.

## LIMITATIONS

Read this before trusting any number above.

- **The queries are hand-written, not sampled from production traffic.** They
  reflect what the author imagines users type, which is not the same as the real
  distribution — real traffic has typos, multi-turn context, non-English, and a
  long tail this set does not contain. Absolute accuracy here should not be read as
  production accuracy.
- **Single labeller, no inter-annotator agreement.** One person (the author) wrote
  and labelled all 180 queries. There is no second annotator and therefore no
  measure of labelling reliability. The dual-signal "correct" domain in particular
  is a judgement call; a different careful labeller would disagree on some.
- **Set size is small: 180 queries** (120/45/15). Per-domain and per-bucket cells
  are small enough that a few flips move the percentages by whole points. Treat
  these as directional, not precise.
- **The staged thresholds were chosen in-sample.** `stage1_accept=0.70`,
  `stage2_accept=0.18` were picked by a sweep over *this same* 180-query set
  (recorded in `results/staged/threshold_sweep.json`). They are **not held out**.
  Reported staged numbers are in-sample; a real deployment must recalibrate on a
  separate validation split, and out-of-sample performance will be somewhat worse.
- **Neural embedding routers were not evaluated in this environment.** The
  `SemanticRouter`/`AdvancedHybridRouter` need `torch` and download
  `all-MiniLM-L6-v2` from Hugging Face, which is unreachable in the offline
  CI/eval environment used here. The harness supports `--router semantic/advanced`
  and will report a clean failure (never a fabricated number) if the model cannot
  load. Consequently the **stage-2 "semantic" pass is lexical TF-IDF, not neural
  embeddings** — a genuine but weaker signal. The staging/abstain *logic* is
  independent of stage-2's implementation; swapping in the embedding router as
  stage 2 on a networked machine would reuse the same thresholds and escalation.
- **`general` does double duty** as both the out-of-domain label and the abstain
  token. Abstention is therefore defined as *predicted == general*; for a true OOD
  query, abstaining is also the correct answer. This is intentional but worth
  knowing when reading the abstention and OOD-accuracy numbers together.

## What I would change with real traffic

Sample queries from production logs and label a held-out test set from them; get at
least a second labeller and report inter-annotator agreement (e.g. Cohen's κ) on
the dual-signal cases; grow the set by 5–10× so per-domain cells are stable;
calibrate thresholds on a train split and report only held-out numbers; and add the
neural embedding router as stage 2 with cached embeddings so the "expensive" pass
is the real thing rather than a lexical stand-in.

## Files

| Path | What |
|---|---|
| `queries.jsonl` | 180 labelled queries |
| `run_eval.py` | eval harness (`--router`, `--out`) |
| `compare.py` | baseline vs staged → `results/COMPARISON.md` |
| `ci_gate.py` | pass/fail gate used by CI |
| `results/<router>/` | `results.json`, `report.md`, `confusion_matrix.png`, `errors.csv` |
| `results/staged/threshold_sweep.json` | the in-sample threshold sweep |
| `runpod_validate.sh`, `runpod_driver.py`, `RUNPOD.md` | GPU full-stack validation |
| `../../src/routing/staged_router.py` | the staged router (new) |
| `../../tests/test_staged_router.py` | unit tests |
| `../../.github/workflows/routing-eval.yml` | CI gate |
