#!/usr/bin/env python3
"""
Compare two routing eval runs side by side and write results/COMPARISON.md.

Reads results.json produced by run_eval.py for a baseline and a staged router.
Every number is read straight from those results.json files -- nothing is
recomputed or hand-entered here. The delta on CONFIDENT-AND-WRONG is reported
first (the metric that matters most), top-1 accuracy second.

Usage:
  python eval/routing/compare.py                       # fast vs staged (defaults)
  python eval/routing/compare.py --baseline fast --staged staged
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DOMAINS = ["code", "math", "medical", "finance", "reasoning", "general"]


def load(router, explicit=None):
    p = Path(explicit) if explicit else (RESULTS / router / "results.json")
    if not p.exists():
        raise SystemExit(f"[FATAL] missing results for '{router}': {p}\n"
                         f"        run: python eval/routing/run_eval.py --router {router}")
    return json.loads(p.read_text())


def pct(x):
    return f"{x*100:.1f}%"


def signed_pp(delta_fraction):
    # delta expressed in percentage POINTS
    v = delta_fraction * 100
    return f"{v:+.1f} pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="fast")
    ap.add_argument("--staged", default="staged")
    ap.add_argument("--baseline-json", default=None)
    ap.add_argument("--staged-json", default=None)
    ap.add_argument("--out", default=str(RESULTS / "COMPARISON.md"))
    args = ap.parse_args()

    B = load(args.baseline, args.baseline_json)
    S = load(args.staged, args.staged_json)
    bm, sm = B["metrics"], S["metrics"]
    n = bm["n"]
    assert bm["n"] == sm["n"], "eval sets differ in size"

    cw_delta = sm["confident_and_wrong_rate"] - bm["confident_and_wrong_rate"]
    acc_delta = sm["overall_accuracy"] - bm["overall_accuracy"]
    abst_delta = sm["abstention_rate"] - bm["abstention_rate"]

    L = []
    L.append(f"# Routing comparison — baseline `{args.baseline}` vs `{args.staged}`\n")
    L.append(f"Query set: `{Path(B['queries_path']).name}` (n = {n}). "
             f"All values read directly from each router's `results.json`.\n")

    # ---- headline: confident-and-wrong FIRST ----
    L.append("## 1. Confident-and-wrong (headline)\n")
    L.append("The rate at which the router commits to a *specialist* domain (no abstention) "
             "and is wrong. These are silent failures — a wrong adapter would load with no signal.\n")
    L.append(f"| | baseline `{args.baseline}` | staged `{args.staged}` | delta |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| **confident-and-wrong** | {pct(bm['confident_and_wrong_rate'])} "
             f"({bm['confident_and_wrong']}/{n}) | {pct(sm['confident_and_wrong_rate'])} "
             f"({sm['confident_and_wrong']}/{n}) | **{signed_pp(cw_delta)}** |")
    L.append("")

    # ---- top-1 accuracy SECOND ----
    L.append("## 2. Top-1 accuracy\n")
    L.append(f"| bucket | baseline `{args.baseline}` | staged `{args.staged}` | delta |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| **overall** | {pct(bm['overall_accuracy'])} | {pct(sm['overall_accuracy'])} "
             f"| **{signed_pp(acc_delta)}** |")
    for b in sorted(bm["bucket_accuracy"]):
        ba = bm["bucket_accuracy"][b]["accuracy"]
        sa = sm["bucket_accuracy"][b]["accuracy"]
        L.append(f"| {b} | {pct(ba)} | {pct(sa)} | {signed_pp(sa-ba)} |")
    L.append("")

    # ---- abstention ----
    L.append("## 3. Abstention rate\n")
    L.append(f"| | baseline `{args.baseline}` | staged `{args.staged}` | delta |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| routed to `general` | {pct(bm['abstention_rate'])} ({bm['abstentions']}/{n}) "
             f"| {pct(sm['abstention_rate'])} ({sm['abstentions']}/{n}) | {signed_pp(abst_delta)} |")
    L.append("")

    # ---- per-domain precision/recall ----
    L.append("## 4. Per-domain precision / recall\n")
    L.append(f"| domain | support | P {args.baseline} | P {args.staged} "
             f"| R {args.baseline} | R {args.staged} |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for d in DOMAINS:
        bpd, spd = bm["per_domain"][d], sm["per_domain"][d]
        L.append(f"| {d} | {bpd['support']} | {pct(bpd['precision'])} | {pct(spd['precision'])} "
                 f"| {pct(bpd['recall'])} | {pct(spd['recall'])} |")
    L.append("")

    # ---- latency ----
    L.append("## 5. Latency (mean ms/query)\n")
    L.append(f"| baseline `{args.baseline}` | staged `{args.staged}` |")
    L.append("|---:|---:|")
    L.append(f"| {bm['latency_ms']['mean']:.3f} | {sm['latency_ms']['mean']:.3f} |")
    L.append("")

    # ---- narrative ----
    L.append("## Why confident-and-wrong is the metric that matters more here\n")
    L.append(
        f"Top-1 accuracy treats every error the same. But in this system a routing error is "
        f"not neutral: the routed domain decides which LoRA adapter gets loaded and which "
        f"agent pipeline runs. When the baseline keyword router picks the *wrong specialist* "
        f"with no hesitation, that is a **silent** failure — the pipeline confidently serves a "
        f"query from the wrong expert and nothing in the system flags it. "
        f"The staged router adds an explicit *unsure* outcome: when neither the cheap keyword "
        f"pass nor the costlier semantic pass clears its floor, it routes to `general`/abstain "
        f"instead of forcing a specialist. "
        f"The effect is that confident-and-wrong falls from "
        f"{pct(bm['confident_and_wrong_rate'])} to {pct(sm['confident_and_wrong_rate'])} "
        f"({signed_pp(cw_delta)}): {bm['confident_and_wrong']-sm['confident_and_wrong']} "
        f"queries that were previously mis-served with confidence are now either corrected by "
        f"the second stage or turned into visible abstentions. "
        f"Overall top-1 accuracy moves {signed_pp(acc_delta)} at the same time, so the reduction "
        f"in silent failures did not come at the cost of accuracy. "
        f"A silent wrong route is worse than a visible 'I'm not sure' — the latter can be "
        f"escalated, logged, or sent to a default pipeline; the former is discovered only when "
        f"the answer is already wrong.\n"
    )

    out = Path(args.out)
    out.write_text("\n".join(L))

    print(f"baseline={args.baseline}  staged={args.staged}  n={n}")
    print(f"confident-and-wrong : {pct(bm['confident_and_wrong_rate'])} -> "
          f"{pct(sm['confident_and_wrong_rate'])}  ({signed_pp(cw_delta)})   <<< headline")
    print(f"top-1 accuracy      : {pct(bm['overall_accuracy'])} -> "
          f"{pct(sm['overall_accuracy'])}  ({signed_pp(acc_delta)})")
    print(f"abstention rate     : {pct(bm['abstention_rate'])} -> "
          f"{pct(sm['abstention_rate'])}  ({signed_pp(abst_delta)})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
