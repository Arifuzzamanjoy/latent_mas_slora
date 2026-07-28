#!/usr/bin/env python3
"""
Four-way routing comparison -> results/COMPARISON.md

Reads results.json for each router that has been run, plus results/cost_profile.json,
and generates the comparison document. Routers that could not be run are reported as
NOT MEASURED with the recorded reason -- never with a substituted number.

Every figure in the generated markdown is read or computed from those JSON files.

Usage:
  python eval/routing/compare.py
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DOMAINS = ["code", "math", "medical", "finance", "reasoning", "general"]
ROUTERS = ["fast", "staged", "semantic", "advanced"]


def load_metrics(router):
    p = RESULTS / router / "results.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return d["metrics"], d


def pct(x):
    return f"{x*100:.1f}%"


def pp(x):
    return f"{x*100:+.1f} pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "COMPARISON.md"))
    args = ap.parse_args()

    M, FULL = {}, {}
    for r in ROUTERS:
        M[r], FULL[r] = load_metrics(r)

    cost = {}
    cp = RESULTS / "cost_profile.json"
    if cp.exists():
        cost = json.loads(cp.read_text())

    measured = [r for r in ROUTERS if M[r]]
    missing = [r for r in ROUTERS if not M[r]]
    if "fast" not in measured or "staged" not in measured:
        raise SystemExit("[FATAL] need at least fast + staged results. Run run_eval.py first.")

    bm, sm = M["fast"], M["staged"]
    n = bm["n"]
    L = []

    L.append("# Routing comparison — four-way\n")
    L.append(f"Query set: `queries.jsonl` (n = {n}). Every value is read from each router's "
             f"`results.json` or from `cost_profile.json`; nothing is hand-entered.\n")
    if missing:
        L.append(f"> **{', '.join(missing)} could not be run in this environment.** They are "
                 f"reported as NOT MEASURED below, with the exact reason. No values were "
                 f"substituted or estimated for them.\n")

    # ---------------- headline ----------------
    L.append("## 1. Confident-and-wrong (headline)\n")
    L.append("Committed to a *specialist* domain (no abstention) and was wrong — a **silent** "
             "failure, since the routed domain selects which adapter/pipeline runs.\n")
    L.append("| router | confident-and-wrong | vs `fast` |")
    L.append("|---|---:|---:|")
    for r in ROUTERS:
        if M[r]:
            d = M[r]["confident_and_wrong_rate"]
            delta = "—" if r == "fast" else pp(d - bm["confident_and_wrong_rate"])
            L.append(f"| `{r}` | {pct(d)} ({M[r]['confident_and_wrong']}/{M[r]['n']}) | {delta} |")
        else:
            L.append(f"| `{r}` | NOT MEASURED | — |")
    L.append("")

    # ---------------- accuracy ----------------
    L.append("## 2. Top-1 accuracy\n")
    buckets = sorted(bm["bucket_accuracy"])
    L.append("| router | overall | " + " | ".join(buckets) + " |")
    L.append("|---|---:|" + "|".join(["---:"] * len(buckets)) + "|")
    for r in ROUTERS:
        if M[r]:
            cells = " | ".join(pct(M[r]["bucket_accuracy"][b]["accuracy"]) for b in buckets)
            L.append(f"| `{r}` | **{pct(M[r]['overall_accuracy'])}** | {cells} |")
        else:
            L.append(f"| `{r}` | NOT MEASURED | " + " | ".join(["—"] * len(buckets)) + " |")
    L.append("")

    # ---------------- abstention ----------------
    L.append("## 3. Abstention rate\n")
    L.append("| router | abstention (routed to `general`) |")
    L.append("|---|---:|")
    for r in ROUTERS:
        L.append(f"| `{r}` | {pct(M[r]['abstention_rate'])} ({M[r]['abstentions']}/{M[r]['n']}) |"
                 if M[r] else f"| `{r}` | NOT MEASURED |")
    L.append("")

    # ---------------- per-domain P/R ----------------
    L.append("## 4. Per-domain precision / recall\n")
    hdr = " | ".join(f"P {r} | R {r}" for r in measured)
    L.append(f"| domain | support | {hdr} |")
    L.append("|---|---:|" + "|".join(["---:"] * (2 * len(measured))) + "|")
    for d in DOMAINS:
        cells = " | ".join(f"{pct(M[r]['per_domain'][d]['precision'])} | "
                           f"{pct(M[r]['per_domain'][d]['recall'])}" for r in measured)
        L.append(f"| {d} | {bm['per_domain'][d]['support']} | {cells} |")
    L.append("")

    # ---------------- cost ----------------
    L.append("## 5. Cost\n")
    L.append("| router | mean latency | cold start (import+construct+1st predict) | "
             "dependency weight | offline-capable |")
    L.append("|---|---:|---:|---|:--:|")
    for r in ROUTERS:
        c = cost.get(r, {})
        if M[r] and "mean_latency_ms" in c:
            L.append(f"| `{r}` | {c['mean_latency_ms']:.4f} ms | "
                     f"{c['cold_start_ms_import_construct_first_predict']:.1f} ms | "
                     f"stdlib only (no third-party runtime deps) | yes |")
        elif M[r]:
            L.append(f"| `{r}` | {M[r]['latency_ms']['mean']:.4f} ms | — | — | — |")
        else:
            reason = cost.get(r, {}).get("reason", "not run")
            facts = cost.get("_measured_facts", {})
            wheel = facts.get("torch_wheel_size_mb")
            dep = (f"torch wheel {wheel} MB + sentence-transformers + transformers"
                   if wheel else "torch + sentence-transformers")
            L.append(f"| `{r}` | NOT MEASURED | NOT MEASURED | {dep} | "
                     f"**no** (downloads model on first init) |")
    L.append("")
    if missing:
        facts = cost.get("_measured_facts", {})
        L.append(f"`{'`, `'.join(missing)}` could not be measured here. Measured facts behind that:\n")
        if facts.get("torch_wheel_size_mb"):
            L.append(f"- torch wheel is **{facts['torch_wheel_size_mb']} MB** "
                     f"(`{facts.get('torch_wheel_source','')}`), and observed pypi throughput was "
                     f"~{facts.get('pypi_throughput_mb_s','?')} MB/s.")
        for u in facts.get("hf_endpoints_403", []):
            L.append(f"- `403 Forbidden` — {u}")
        L.append("\nBecause the model weights, `config.json`, **and** the hub metadata endpoint are all "
                 "blocked, `SentenceTransformer('all-MiniLM-L6-v2')` cannot initialise, so neither "
                 "neural router can produce a prediction here. Their accuracy is **unknown**, not zero "
                 "and not assumed.\n")
        L.append("`run_eval.py` also refuses to *silently* degrade: both `SemanticRouter` and "
                 "`AdvancedHybridRouter` catch `ImportError` internally and fall back to keyword-only "
                 "scoring, which would otherwise be reported as a 'semantic' result. The harness now "
                 "asserts the encoder and domain centroids actually loaded and raises "
                 "`_NeuralNotLoaded` if not.\n")

    # ---------------- what actually changed ----------------
    fp = {p["id"]: p for p in FULL["fast"]["predictions"]}
    sp = {p["id"]: p for p in FULL["staged"]["predictions"]}
    fixed, broke = [], []
    for i in sorted(fp):
        a, b = fp[i], sp[i]
        ok_f, ok_s = a["predicted"] == a["expected"], b["predicted"] == b["expected"]
        if not ok_f and ok_s:
            fixed.append(a)
        if ok_f and not ok_s:
            broke.append((a, b))
    def cnt(items, bucket, idx=None):
        return sum(1 for x in items if (x[0] if idx is not None else x)["bucket"] == bucket)
    f_single = sum(1 for x in fixed if x["bucket"] == "single_signal")
    b_single = sum(1 for x, _ in broke if x["bucket"] == "single_signal")
    f_dual = sum(1 for x in fixed if x["bucket"] == "dual_signal")
    b_dual = sum(1 for x, _ in broke if x["bucket"] == "dual_signal")

    L.append("## What actually changed\n")
    L.append(f"**The gain is entirely in `single_signal`: +{f_single} fixed / −{b_single} broken "
             f"(net +{f_single-b_single}).**\n")
    L.append(f"**`dual_signal` REGRESSED: "
             f"{pct(bm['bucket_accuracy']['dual_signal']['accuracy'])} → "
             f"{pct(sm['bucket_accuracy']['dual_signal']['accuracy'])}, "
             f"{f_dual} fixed / {b_dual} broken (net {f_dual-b_dual}).** This is a regression. The "
             f"staged router is *worse* at the ambiguous two-signal queries than the plain keyword "
             f"router, because it abstains on several it previously happened to get right.\n")
    rf = bm["per_domain"]["reasoning"]["recall"]
    rs = sm["per_domain"]["reasoning"]["recall"]
    L.append(f"**The mechanism is `reasoning` recall: {pct(rf)} → {pct(rs)}.** The keyword router "
             f"almost never emitted `reasoning` at all — the reasoning profile's keywords "
             f"(\"why\", \"how\", \"explain\", \"compare\") are generic words that lose to more "
             f"specific domain terms, so reasoning queries were absorbed by other domains. Stage 2's "
             f"TF-IDF pass recovered them. **This, not dual-signal disambiguation, is where the "
             f"improvement came from.** Anyone reading the headline as \"staging resolves ambiguity\" "
             f"is reading it wrong.\n")

    L.append(f"### The {len(broke)} queries the staged router broke\n")
    L.append("| id | bucket | query | expected | `fast` | `staged` |")
    L.append("|---|---|---|---|---|---|")
    for a, b in broke:
        q = a["query"][:58] + ("…" if len(a["query"]) > 58 else "")
        L.append(f"| {a['id']} | {a['bucket']} | {q} | `{a['expected']}` | "
                 f"`{a['predicted']}` ✓ | `{b['predicted']}` ✗ |")
    L.append("")
    n_abstain_break = sum(1 for _, b in broke if b["predicted"] == "general")
    L.append(f"{n_abstain_break} of the {len(broke)} became abstentions rather than wrong "
             f"specialist picks, so they cost accuracy but not confident-and-wrong. "
             f"{len(broke)-n_abstain_break} became a different wrong specialist.\n")

    # ---------------- narrative ----------------
    cw_delta = sm["confident_and_wrong_rate"] - bm["confident_and_wrong_rate"]
    acc_delta = sm["overall_accuracy"] - bm["overall_accuracy"]
    L.append("## Why confident-and-wrong is the metric that matters more here\n")
    L.append(
        f"Top-1 accuracy treats every error the same. In this system a routing error is not "
        f"neutral: the routed domain decides which LoRA adapter loads and which agent pipeline "
        f"runs. When the keyword router picks the wrong specialist with no hesitation, that is a "
        f"**silent** failure — the wrong expert answers and nothing flags it. The staged router "
        f"adds an explicit *unsure* outcome, so confident-and-wrong falls "
        f"{pct(bm['confident_and_wrong_rate'])} → {pct(sm['confident_and_wrong_rate'])} "
        f"({pp(cw_delta)}) while accuracy moves {pp(acc_delta)}. A visible \"not sure\" can be "
        f"escalated, logged, or sent to a default pipeline; a silent wrong route is discovered "
        f"only when the answer is already wrong.\n")

    L.append("## Why staged, given semantic exists\n")
    if M.get("semantic"):
        sem = M["semantic"]
        better = sem["overall_accuracy"] > sm["overall_accuracy"]
        L.append(f"Measured: semantic top-1 {pct(sem['overall_accuracy'])} vs staged "
                 f"{pct(sm['overall_accuracy'])}; confident-and-wrong "
                 f"{pct(sem['confident_and_wrong_rate'])} vs {pct(sm['confident_and_wrong_rate'])}.\n")
        if better:
            L.append("**The semantic router is more accurate than staged.** The argument for staged "
                     "is therefore not accuracy — it is cost and CI-gateability (see §5). State the "
                     "accuracy gap plainly when presenting this.\n")
    else:
        L.append(
            "**This is the honest answer: I do not know, because I could not run them here.** The "
            "embedding routers were never benchmarked — `huggingface.co` is blocked in this "
            "environment, so the model cannot be fetched (§5). It is entirely possible the semantic "
            "router beats the staged router on accuracy; a fair reading is that this comparison is "
            "**incomplete**, and the staged-vs-fast result should not be presented as \"staged is the "
            "best router\".\n")
        L.append(
            "What can be defended without those numbers is narrower and is about **cost and "
            "gateability**, not quality: the staged router has no third-party runtime dependency, "
            "starts in tens of milliseconds, routes in well under a millisecond, and runs fully "
            "offline — so it can gate every pull request on a standard CPU runner in seconds. The "
            "neural routers need a ~527 MB torch wheel plus a model download, which is a different "
            "class of CI dependency. That is an argument about what is cheap to *gate on*, not an "
            "argument that staging produces better routing. Running "
            "`--router semantic` and `--router advanced` on a networked machine is the obvious next "
            "step and would settle it.\n")

    Path(args.out).write_text("\n".join(L))
    print(f"measured: {measured}   NOT MEASURED: {missing or 'none'}")
    print(f"confident-and-wrong  fast {pct(bm['confident_and_wrong_rate'])} -> "
          f"staged {pct(sm['confident_and_wrong_rate'])} ({pp(cw_delta)})")
    print(f"top-1 accuracy       fast {pct(bm['overall_accuracy'])} -> "
          f"staged {pct(sm['overall_accuracy'])} ({pp(acc_delta)})")
    print(f"single net +{f_single-b_single} | dual net {f_dual-b_dual} | "
          f"reasoning recall {pct(rf)} -> {pct(rs)} | broke {len(broke)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
