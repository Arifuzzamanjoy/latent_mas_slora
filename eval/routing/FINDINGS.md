# Findings — defects the routing eval surfaced

Notes on four defects in the existing routing code that building this evaluation exposed.
None of them are fixed in this branch; the last section explains why, and which one I would
ship first.

Every figure below is either verifiable from source (file:line given) or read from a
`results.json` in `results/`. One exception is flagged loudly in F2.

---

## F1 — Duplicate `Domain` enum; cross-module equality is silently `False`

**What it is.** Two separate `Domain` enums exist with identical members:

- `src/routing/domain_profiles.py:12` — `class Domain(Enum)`
- `src/routing/fast_router.py:19` — `class Domain(Enum)`

`src/routing/__init__.py:12` re-exports the `domain_profiles` one
(`from .domain_profiles import Domain, DomainProfile, DOMAIN_PROFILES`), while `FastRouter`
returns members of its own local copy. A caller doing `from src.routing import Domain` and
comparing against a `FastRouter` result is comparing two unrelated classes.

**Evidence.** Verbatim REPL output (both modules loaded directly by file path to avoid the
torch import in `src/routing/__init__.py`; class identity is the same either way):

```
>>> A = domain_profiles.Domain ; B = fast_router.Domain
>>> A is B
False
>>> A.CODE == B.CODE
False
>>> A.CODE is B.CODE
False
>>> A.CODE == 'code'
False
>>> B.CODE == 'code'
False
>>> A.CODE.value == B.CODE.value
True
>>> isinstance(A.CODE, str)
False
>>> [d.value for d in A] == [d.value for d in B]
True
>>> A.__module__, B.__module__
domain_profiles fast_router
```

Both directions fail: enum-to-enum across modules is `False`, and enum-to-string is also
`False` because these are plain `Enum`, not `str, Enum`. The `.value` strings are identical,
and the member lists are identical — which is exactly what makes it easy to miss.

**Why it is silent.** Nothing has broken because every consumer happens to normalise through
`.value` before comparing:

- `eval/routing/run_eval.py` — `FastAdapter.predict` returns `res.domain.value`; the staged
  adapter returns `res.domain`, already a `str`.
- `src/routing/staged_router.py:93` — `ABSTAIN = "general"`, a plain string, and
  `StagedResult.domain` is typed `str`.
- Comparisons against expected labels are string-to-string throughout.

So this is **latent, not dormant-by-design**. No code comments the equality hazard, no test
covers it, and nothing enforces the `.value` convention. The first caller that writes the
natural-looking `if router.route(q)[0] == Domain.MEDICAL:` against a `FastRouter` result gets
a branch that is silently always `False` — no exception, no warning, just a rule that never
fires. A wrong-but-plausible routing decision is precisely the failure class this eval was
built to make visible, and this one would not show up in the metrics at all, because the
harness never takes that path.

**Blast radius.** Any consumer of `FastRouter` outside the eval harness. `src/system.py`
imports `Domain` from `.routing` (the `domain_profiles` one) and separately calls
`self._fast_router.route(question)` at `system.py:407`, then compares `domain != Domain.GENERAL`
at `system.py:434` — the two enums meeting is exactly the shape described above. Also affects
anything using `Domain` as a dict key across module boundaries, since the hashes differ.

**Fix.** Delete the enum in `fast_router.py` and import from `domain_profiles`. The
docstring rationale for the duplicate — `fast_router.py:2,5`, *"Zero ML Dependencies / No torch,
no transformers, no sentence-transformers"* — does not hold: `domain_profiles.py` imports only
`enum`, `dataclasses` and `typing` (verified, `domain_profiles.py:7-9`). Importing it costs
nothing. If a hard guarantee is wanted, make the shared enum `class Domain(str, Enum)` so
string comparison also works, and add a test asserting there is exactly one `Domain` class in
the package.

**Why not fixed here.** It modifies a baseline router in the middle of a before/after
comparison. `fast_router.py` is the measured baseline; editing it invalidates
`results/fast/`, the four-way table, and the train/holdout numbers that were all produced
against the current code.

---

## F2 — `SemanticRouter` default `confidence_threshold` is miscalibrated

**What it is.** `src/routing/semantic_router.py:171` — `get_best_domain(self, prompt,
confidence_threshold: float = 0.20)`. The abstain mechanism exists and works; the *default
value* sits below the score distribution the router actually produces, so it effectively
never triggers and the router commits on almost everything.

**Evidence — code (verifiable here).** The score pipeline makes 0.20 close to meaningless:

- `semantic_router.py:121` — cosine similarity is rescaled `(similarity + 1) / 2`, mapping
  `[-1, 1]` to `[0, 1]`. Sentence-embedding cosines between short English texts are rarely
  negative, so *every* domain receives a substantial floor score even when unrelated.
- `route()` then normalises across domains: `total = sum(...); normalized = [(d, s / total) ...]`.
  With **6 domains** the uniform-share baseline is **1/6 = 0.1667** (verified by enumerating
  `Domain`). A threshold of **0.20** is only ~0.033 above pure uniform.
- Domain weights are applied before normalisation and are uneven
  (`finance 1.3, medical 1.2, code 1.0, math 1.0, reasoning 0.8, general 0.5`), which alone
  pushes the top-weighted domain past 0.20 on a near-tie.

So the top-1 normalised score clears 0.20 for nearly any input where one domain edges ahead.
The threshold is not doing the job its name implies.

**Evidence — metrics: NOT VERIFIABLE IN THIS TREE.**

> The abstention / out-of-domain / confident-and-wrong figures for this router were reported
> to me as **abstention 0.6% (1/180), out_of_domain accuracy 0.0% (0/15), confident-and-wrong
> 56.7%**, citing `results/semantic`. **I could not verify them.** `results/semantic/` exists
> but is **empty** — it was created by `run_eval.py`'s `out_dir.mkdir()` before the router
> failed to initialise, and contains no `results.json`. The semantic router has never
> completed a run in this environment: every Hugging Face endpoint needed for
> `all-MiniLM-L6-v2` returns `403 Forbidden` (see `results/COMPARISON.md` §5 and
> `results/cost_profile.json`). Those numbers are recorded here as **reported, unverified**,
> and are deliberately not restated as measured fact anywhere else in this repo.
>
> To make them citable, run on a networked machine:
> ```bash
> python eval/routing/run_eval.py --router semantic
> ```
> then re-read `results/semantic/results.json`. If the numbers hold, they are consistent with
> the code analysis above: near-zero abstention and 0% out-of-domain accuracy is exactly the
> signature of a threshold sitting below the score floor.

**Why it is silent.** A threshold that never fires produces no error and no log line — it
produces confident answers. Downstream, `system.py:434` gates adapter loading on
`confidence > 0.20 and domain != Domain.GENERAL`, so a router that always clears 0.20 always
looks decisive. Without an abstention metric in the eval, this is invisible: top-1 accuracy
alone cannot distinguish "correctly confident" from "never declines".

**Distinction that matters.** This is a **calibration defect, not an embedding-quality
defect.** Nothing here shows the embeddings are bad — the ranking they produce may well be
better than the keyword router's. What is broken is the decision rule sitting on top of the
scores. Fixing it is a matter of choosing the operating point, not of changing the model.

**Fix.** Calibrate the threshold against the actual score distribution rather than picking a
round number: sweep it on the train split the way `results/staged/threshold_sweep_train.json`
does, select on confident-and-wrong subject to an accuracy floor, and record the chosen value
with its provenance. Consider normalising with a temperature-scaled softmax instead of
sum-normalisation so the scores are comparable across queries, and set the default per-domain
as `advanced_router.py` already attempts.

**Why not fixed here.** Same reason as F1 — it edits a baseline router. It is also untestable
in this environment: with the model unreachable I cannot measure the before state, so I would
be tuning blind and could not produce before/after evidence.

---

## F3 — The `reasoning` profile is effectively unreachable in the keyword router

**What it is.** `FastRouter` almost never emits `reasoning`. Measured on the 180-query set:

| | value | source |
|---|---:|---|
| reasoning recall | **9.7%** (3/31) | `results/fast/results.json` → `per_domain.reasoning` |
| reasoning true positives | 3 | same |
| false negatives | 28 | same |
| total predictions of `reasoning` | 5 | confusion matrix column |

Where the 31 reasoning queries actually went (`results/fast` confusion matrix row):
`general 19, medical 3, code 2, math 2, finance 2, reasoning 3`.

**Evidence — the mechanism.** `fast_router.py:131-144`:

```
matches = pattern.findall(query)
scores[domain] = len(matches)        # raw count of keyword hits
total = sum(scores.values())
if total == 0: -> Domain.GENERAL, confidence 0.5
best_domain = max(scores, key=scores.get)
confidence = scores[best_domain] / total
```

Scoring is an unweighted count of matched keywords, so the domain with the **largest and most
specific vocabulary wins**. Keyword counts in `DOMAIN_KEYWORDS` (verified by enumeration):

| domain | keywords |
|---|---:|
| medical | 82 |
| finance | 52 |
| code | 51 |
| math | 38 |
| **reasoning** | **19** |

The reasoning list is both the smallest and the most generic — `"why", "how", "explain",
"analyze", "compare", "evaluate", "problem", "solution"`. Two failure paths follow:

1. **Loses head-to-head.** A query like *"explain the reasoning error in using past fund
   returns to predict future returns"* matches one or two generic reasoning terms but several
   finance terms, so finance wins on raw count.
2. **Matches nothing at all.** `Domain.GENERAL` is **not a key in `DOMAIN_KEYWORDS`**
   (verified: keys are `medical, finance, code, math, reasoning`), so `general` is reachable
   *only* via the `total == 0` branch. **19 of the 31 reasoning queries matched zero keywords
   in any domain** and were forced to `general` by that branch — not chosen, defaulted to.

That second number is the core of it: reasoning queries are phrased in ordinary English that
the keyword tables simply do not cover.

**Why it is silent.** Overall accuracy hides it. A domain can sit near-zero recall while the
aggregate looks acceptable, and because the misses land on `general` — which is also the
abstain token (F4) — they look like cautious behaviour rather than a blind spot. It took
per-domain recall in the confusion matrix to see it.

**Blast radius.** Any query routed on reasoning grounds is misrouted or defaulted. This is
also the single largest contributor to the staged router's improvement: recall goes 9.7% →
45.2% (`results/staged`), which `results/COMPARISON.md` identifies as the actual mechanism
behind the headline number.

**Fix.** Raw match counts are the wrong scoring function — use IDF-style weighting so common
terms count less, which is what the staged router's stage 2 does. Failing that, the reasoning
profile needs vocabulary that discriminates (`syllogism, fallacy, premise, valid, infer,
assumption, counterexample`) and generic connectives should be down-weighted rather than
counted. Longer term, a keyword table is the wrong tool for a domain defined by *form of
argument* rather than subject vocabulary.

**Why not fixed here.** Editing `DOMAIN_KEYWORDS` changes the measured baseline, and the
reasoning gap is the main effect the staged-router comparison demonstrates. Fixing the
baseline and keeping the old comparison would be dishonest; fixing it properly means re-running
everything, which belongs in its own PR.

---

## F4 — `ABSTAIN` reuses the `general` enum member

**What it is.** There is no distinct "declined to route" outcome. `staged_router.py:93` sets
`ABSTAIN = "general"`, the same value as `domain_profiles.py:19` `GENERAL = "general"`. A
caller receiving `general` cannot tell whether the router judged the query *genuinely general*
or *declined to commit*.

**Evidence.** Measured on `results/staged/results.json`:

| | value |
|---|---:|
| `general` precision | **22.4%** (0.2239) |
| true positives | 15 |
| false positives | 52 |
| `general` recall | 100% |
| abstentions | 67 / 180 |

15 of the 67 predictions of `general` are genuine out-of-domain queries; the other 52 are
abstentions on queries that belonged to a specialist. Precision of 22.4% is not a routing
failure — it is the arithmetic consequence of two different outcomes sharing one label.

**Why it is silent.** It degrades an interpretation, not an execution. Everything runs; the
metric is simply not answerable from the output. It also makes `out_of_domain` accuracy
partly tautological — the correct label for an OOD query *is* `general`, and abstaining also
yields `general`, so a router that abstained on everything would score 100% on that bucket
(this is documented in `README.md`). Any downstream policy that wants to treat "unsure"
differently from "general" — escalate, ask a clarifying question, log for review — cannot be
written against this return value.

**Blast radius.** Anything wanting to act on uncertainty. Note the staged router *does* carry
the distinction internally: `StagedResult.method` is `staged:unsure` vs `staged:stage1` /
`staged:stage2`, and `stage` is `0` for abstain. The information exists; it is discarded the
moment a caller looks only at `.domain`.

**Fix.** Either add a distinct `UNSURE` member to the shared enum, or — less invasive — add an
explicit `abstained: bool` on the result object and keep `domain` as-is.

**Migration cost.** A new enum member is the breaking option: every `Domain`-keyed dict and
exhaustive `if/elif` chain over domains needs a branch, including
`system.py:_get_model_compatible_adapter()` (whose dict is keyed by domain) and
`DOMAIN_PROFILES` itself, which would need a profile for a member that has no exemplar
prompts. Any consumer treating `general` as "use the default pipeline" would silently change
behaviour for the 52 abstained queries. The `abstained` flag is backwards compatible — old
callers keep working and read `general`, new callers check the flag — and is what I would
actually do first, with the enum member as a later, deliberate break.

---

## Why none of these are fixed in this branch

All four live in the code the evaluation *measures*. Changing any of them invalidates the
evidence this branch exists to provide:

- **The before/after comparison.** `results/fast/` is the measured baseline. F1 and F3 edit
  `fast_router.py` directly; F2 edits `semantic_router.py`. Once the baseline changes, the
  four-way table in `results/COMPARISON.md` describes code that no longer exists.
- **The held-out split.** `results/HELDOUT.md` reports a train-tuned config on an untouched
  holdout. Changing routing behaviour makes the train tuning stale, and re-tuning on the same
  split after seeing the holdout would quietly destroy the separation the split was created
  to establish.
- **The CI gate.** `.github/workflows/routing-eval.yml` pins `--min-accuracy 0.58` and
  `--max-confident-wrong 0.11`, with a comment recording the run they came from. Any of these
  fixes moves the measured numbers, so the thresholds would be gating against a run that no
  longer describes the code.

The right shape is a follow-up PR per fix, each re-running the eval and attaching before/after
as evidence — which is precisely what this harness is for. That is the argument for having
built it: these are now four measurable changes rather than four judgement calls.

**Which I would ship first: F3.** It has the largest measured effect (reasoning recall 9.7%,
28 false negatives out of 31 — the biggest single hole in the confusion matrix), the fix is
contained to a scoring function and a keyword list, and the eval already proves the gap is
recoverable, since stage 2 recovers it to 45.2% without touching the model. It is also the
only one of the four with a directly measurable success criterion already in place.

**F1 second** — it is the cheapest fix and the highest-severity *latent* bug: it is currently
harmless only by convention, and the convention is undocumented and untested. **F4 third**, as
the backwards-compatible `abstained` flag, since it unblocks any real uncertainty-handling
policy. **F2 last**, not because it matters least — a 56.7% confident-and-wrong rate would
make it the most severe of the four if confirmed — but because it cannot be measured in this
environment at all, and tuning a threshold without a before-measurement is guessing.
