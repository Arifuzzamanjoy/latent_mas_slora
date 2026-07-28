# Generalizing this method to other LLM applications

These are notes on how the evaluation approach in this directory transfers to production
LLM apps in other domains — construction, tourism, aviation, e-commerce. The specifics here
are about domain routing, but almost none of the method depends on that.

## Start from traffic, not imagination

The weakest part of this repo's eval is that I wrote the 180 queries myself. That was the
only option for a public repo with no users, and it is stated plainly in the README's
LIMITATIONS. In a real app you do not have that excuse: you have logs.

Sample a golden set from actual production traffic. Stratify it the way the traffic is
actually distributed, not the way you wish it were — if 60% of your e-commerce queries are
order-status lookups, the golden set is 60% order-status, including the badly spelled ones
and the ones where the user pasted a tracking number with no other words. Freeze it, version
it, and keep it in the repo. Resample it periodically, because traffic drifts; a set frozen
in March stops describing August.

Two labellers minimum, with disagreement measured and reported. This repo has one labeller
and no inter-annotator agreement, which is a real weakness. Where two annotators disagree,
that is usually not annotator sloppiness — it is a genuinely ambiguous case, and those are
exactly the cases worth designing for. Keep them; do not discard them to make the number
look better.

## Build the failure taxonomy before choosing a metric

The most useful thing I did here was decide, before writing any metric, that not all errors
are equal — a wrong-but-confident route is worse than an abstention, because the wrong-but-
confident one is silent. That single distinction is what made `confident-and-wrong` the
headline instead of top-1 accuracy, and it changed which router looked better.

Do that first, per app. Sit with real failures, sort them into kinds, then ask which kinds
are expensive. Only then define the metric. A metric chosen before you understand the
failure modes will optimise for the wrong thing very efficiently.

## "Wrong" means different things per app

This is where the four domains diverge sharply, and why one shared metric across all of them
would be a mistake.

**Aviation.** Regulatory and safety-adjacent. A confidently wrong answer about a procedure or
a limitation is the worst outcome by a wide margin, and it is not linearly worse — it is
categorically worse. Abstention is cheap here; being wrong is not. Set thresholds so the
system declines readily, and measure the confident-error rate as the primary number, with
coverage as the constraint you are willing to trade.

**Construction.** Similar asymmetry (spec compliance, tolerances, load figures) but with a
twist: much of the ground truth lives in documents — drawings, spec sheets, revisions. A
large share of "model errors" are actually retrieval errors against a stale or wrong-revision
document. Evaluate retrieval separately from generation, or you will keep tuning the
generator to fix a retrieval bug.

**Tourism.** Much more tolerant. A mediocre restaurant recommendation costs almost nothing;
users self-correct. But there is a hard subset — visas, entry requirements, vaccination
rules, refund policies — where wrong is expensive and occasionally legally significant. Do
not average those together. Split the eval by consequence class and report them separately;
a single blended accuracy number hides the only part that matters.

**E-commerce.** The errors that hurt are the ones tied to money and state: wrong price, wrong
stock, wrong order, wrong return window. These are also the most checkable, because there is
a database that knows the answer. Lean on that (see programmatic checks below).

The common move: **define the consequence classes per app, and report the metric per class.**
One headline number across all traffic is how expensive failures get averaged into
invisibility.

## Abstention as a design default, not a fallback

The staged router's real contribution here is not accuracy — it is that "unsure" is a
first-class outcome rather than an error path. That transfers directly and is usually the
highest-leverage change in a production LLM app.

Give the system somewhere to go when it is not confident: ask a clarifying question, escalate
to a human, fall back to a narrower deterministic flow, or return a hedged answer that says
what it does not know. Then measure how often it takes that path and whether taking it was
right. Watch the tradeoff honestly — in this repo, abstaining more turned four dual-signal
queries from correct into abstentions. Abstention is not free; it costs coverage. The point
is that you now *see* the cost instead of eating it silently.

## Gate it in CI, and grow a regression corpus

The pattern in `.github/workflows/routing-eval.yml` transfers as-is:

- Run the eval on every pull request.
- **Fail the build** on a threshold — accuracy floor, confident-error ceiling. A metric
  nobody enforces becomes decoration within about two sprints.
- Upload the results directory as a build artifact so you can diff runs and see *which*
  cases changed, not just that the aggregate moved.
- Keep the gate cheap and deterministic. Ours is CPU-only, offline, no model download, and
  finishes in seconds — that is why it can run on every push. If your eval needs a GPU and
  five minutes, it will get skipped, disabled, or moved to nightly and then ignored. Where
  the real system needs a GPU, split it: cheap deterministic gate on every PR, full-stack
  validation on demand (this repo separates those into the CI job and the RunPod script).

Then make the corpus grow. **Every time a client reports a bug, that exact input becomes a
test case with the correct label attached.** This is the highest-value habit on the list.
Over a year the regression corpus becomes the most accurate description of what your users
actually do and where the system actually breaks — far more valuable than any set written up
front, including a well-sampled one.

## What this repo's eval does NOT cover

Being precise about this, because it is the main limit of everything above.

This eval scores a **classification step with a clean label**. Every query has one defensible
correct domain, so "correct" is a string comparison and metrics like precision, recall and a
confusion matrix apply directly. That is a genuinely easy case, and it is the case I have
demonstrated end to end.

**Evaluating generated output quality is a substantially harder problem, and none of it is
demonstrated here.** There is no single correct answer to compare against, quality is
multi-dimensional (correct, grounded, complete, appropriately hedged, correctly toned), and
the dimensions trade off. What I would actually do, marked for what it is:

- **Rubric-based LLM-as-judge, calibrated against humans.** Write an explicit per-dimension
  rubric, have humans score a few hundred outputs, then check the judge reproduces those
  human scores before trusting it anywhere. The calibration set is the part people skip and
  it is the part that makes the judge meaningful. Recalibrate when you change judge model or
  prompt — both silently shift the scale.
- **Pairwise comparison for regressions.** Absolute quality scores are noisy and drift.
  "Is B better or worse than A on this input" is a much more stable question, and regression
  detection is usually the actual thing you need.
- **Programmatic checks wherever ground truth exists.** These are worth more than any judge
  because they are cheap and exact. Did the cited document ID exist and contain the claim
  (construction, aviation)? Does the quoted price match the database (e-commerce)? Is the
  cited regulation real and current (aviation)? Does the itinerary respect the dates given
  (tourism)? Every one of these you can automate is one you do not need a judge for.
- **Human review on a sampled slice, permanently.** Not as a launch gate — as an ongoing
  sample, weighted toward the high-consequence classes.

To be explicit about provenance: **the routing eval, the abstention design, and the CI gate
are proven in this repo, with numbers that reproduce.** Everything in this section is
proposed, not demonstrated. I would rather say that than imply this repo shows more than it
does.
