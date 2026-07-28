#!/usr/bin/env python3
"""
Routing evaluation harness.

Runs a labelled query set (queries.jsonl) through a chosen router and reports:
  - top-1 accuracy (overall + per bucket)
  - full 6x6 confusion matrix (expected vs predicted, incl. 'general')
  - abstention rate           (predicted == 'general' -> declined to pick a specialist)
  - CONFIDENT-AND-WRONG rate  (committed to a specialist, no abstention, wrong)  << HEADLINE
  - per-domain precision / recall
  - mean latency per query

Design constraints:
  - Deterministic: fixed seed, queries iterated in sorted id order, no randomness in routers.
  - CPU-only / offline for the 'fast' and 'staged' routers (no network, no torch).
  - 'semantic'/'advanced' are supported but require torch + network to fetch the
    embedding model; if that fails this harness reports the failure and exits non-zero.
    It NEVER substitutes a fabricated number for a failed run.

Usage:
  python eval/routing/run_eval.py --router staged
  python eval/routing/run_eval.py --router fast --out eval/routing/results/fast
"""
import argparse, json, os, sys, time, csv, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

DOMAINS = ["code", "math", "medical", "finance", "reasoning", "general"]
ABSTAIN_LABEL = "general"   # 'general' doubles as the routers' abstain/fallback token
SEED = 1234


# ----------------------------------------------------------------------------
# Router adapters: uniform .predict(query) -> (domain:str, confidence:float, method:str)
# ----------------------------------------------------------------------------
def _load_module_from_file(mod_name, filename):
    """Load a self-contained module directly by file path.

    Deliberately bypasses src/routing/__init__.py (which imports the torch-based
    routers). The 'fast' and 'staged' routers have no torch/network dependency,
    so loading them this way keeps the harness CPU-only and offline.
    """
    import importlib.util
    path = REPO / "src" / "routing" / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class FastAdapter:
    name = "fast"
    def __init__(self):
        FastRouter = _load_module_from_file("fast_router_standalone", "fast_router.py").FastRouter
        self._r = FastRouter()
    def predict(self, q):
        res = self._r.route_detailed(q)
        return res.domain.value, float(res.confidence), res.method


class StagedAdapter:
    name = "staged"
    def __init__(self):
        StagedRouter = _load_module_from_file("staged_router_standalone", "staged_router.py").StagedRouter
        self._r = StagedRouter()
    def predict(self, q):
        res = self._r.route(q)
        return res.domain, float(res.confidence), res.method


class _NeuralNotLoaded(RuntimeError):
    """Raised when a neural router silently degraded to keyword-only scoring."""


class SemanticAdapter:
    """Existing SemanticRouter (unmodified). Requires torch + embedding model download."""
    name = "semantic"
    def __init__(self):
        from src.routing.semantic_router import SemanticRouter
        self._r = SemanticRouter()
        # Force lazy init, then REFUSE to run if the embedding model did not load.
        # SemanticRouter catches ImportError and falls back to keyword-only scoring
        # (self._model = None -> _semantic_score returns 0.0 for every domain). Without
        # this guard the harness would happily emit keyword-only numbers labelled
        # 'semantic'. That would be a fabricated result, so we hard-fail instead.
        self._r._lazy_init()
        if getattr(self._r, "_model", None) is None:
            raise _NeuralNotLoaded(
                "SemanticRouter fell back to keyword-only scoring: the embedding model "
                "did not load (sentence-transformers missing, or the model could not be "
                "downloaded). Refusing to report keyword-only numbers as 'semantic'.")
        if not getattr(self._r, "_domain_embeddings", None):
            raise _NeuralNotLoaded(
                "SemanticRouter has no domain centroids; embeddings were not computed. "
                "Refusing to report a degraded run as 'semantic'.")

    def predict(self, q):
        domain, conf = self._r.get_best_domain(q)
        return domain.value, float(conf), "semantic"


class AdvancedAdapter:
    """Existing AdvancedHybridRouter (unmodified). Requires torch + embedding model download."""
    name = "advanced"
    def __init__(self):
        from src.routing.advanced_router import AdvancedHybridRouter
        self._r = AdvancedHybridRouter()
        # Same guard as SemanticAdapter: AdvancedHybridRouter also catches ImportError
        # and continues with self._encoder = None, scoring on keyword+meta signals only.
        self._r._lazy_init()
        if getattr(self._r, "_encoder", None) is None:
            raise _NeuralNotLoaded(
                "AdvancedHybridRouter fell back to keyword+meta scoring: the embedding "
                "model did not load. Refusing to report a degraded run as 'advanced'.")
        if not getattr(self._r, "_domain_centroids", None):
            raise _NeuralNotLoaded(
                "AdvancedHybridRouter has no domain centroids; embeddings were not "
                "computed. Refusing to report a degraded run as 'advanced'.")

    def predict(self, q):
        res = self._r.route(q)
        return res.domain.value, float(res.confidence), res.method


ADAPTERS = {
    "fast": FastAdapter,
    "staged": StagedAdapter,
    "semantic": SemanticAdapter,
    "advanced": AdvancedAdapter,
}


def build_router(name):
    if name not in ADAPTERS:
        raise SystemExit(f"Unknown router '{name}'. Choices: {sorted(ADAPTERS)}")
    try:
        return ADAPTERS[name]()
    except Exception as e:
        # Do NOT fabricate results for a router that cannot load.
        raise SystemExit(
            f"[FATAL] Router '{name}' could not be initialised in this environment:\n"
            f"        {type(e).__name__}: {e}\n"
            f"        (semantic/advanced need torch + network access to download the "
            f"embedding model. No numbers were produced; nothing was fabricated.)"
        )


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_queries(path, split="all", split_path=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["id"])   # deterministic order

    if split != "all":
        sp = Path(split_path) if split_path else (HERE / "split.json")
        if not sp.exists():
            raise SystemExit(f"[FATAL] --split {split} requested but {sp} not found.")
        spec = json.loads(sp.read_text())
        key = {"train": "train_ids", "holdout": "holdout_ids"}[split]
        keep = set(spec[key])
        rows = [r for r in rows if r["id"] in keep]
        if len(rows) != len(keep):
            raise SystemExit(f"[FATAL] split '{split}' lists {len(keep)} ids but matched "
                             f"{len(rows)} queries.")
    return rows


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(records):
    n = len(records)
    correct = sum(1 for r in records if r["predicted"] == r["expected"])
    overall_acc = correct / n if n else 0.0

    by_bucket = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in records:
        b = by_bucket[r["bucket"]]
        b["n"] += 1
        b["correct"] += int(r["predicted"] == r["expected"])
    bucket_acc = {
        b: {"n": v["n"], "correct": v["correct"],
            "accuracy": (v["correct"] / v["n"] if v["n"] else 0.0)}
        for b, v in sorted(by_bucket.items())
    }

    abstentions = sum(1 for r in records if r["predicted"] == ABSTAIN_LABEL)
    abstention_rate = abstentions / n if n else 0.0

    # HEADLINE: committed to a specialist (not abstained), and wrong.
    conf_wrong = [r for r in records
                  if r["predicted"] != ABSTAIN_LABEL and r["predicted"] != r["expected"]]
    confident_and_wrong = len(conf_wrong)
    confident_and_wrong_rate = confident_and_wrong / n if n else 0.0

    # Confusion matrix (rows=expected, cols=predicted)
    idx = {d: i for i, d in enumerate(DOMAINS)}
    cm = np.zeros((len(DOMAINS), len(DOMAINS)), dtype=int)
    for r in records:
        cm[idx[r["expected"]], idx[r["predicted"]]] += 1

    # Per-domain precision / recall
    per_domain = {}
    for d in DOMAINS:
        tp = sum(1 for r in records if r["predicted"] == d and r["expected"] == d)
        fp = sum(1 for r in records if r["predicted"] == d and r["expected"] != d)
        fn = sum(1 for r in records if r["predicted"] != d and r["expected"] == d)
        support = sum(1 for r in records if r["expected"] == d)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_domain[d] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp,
                         "fn": fn, "support": support}

    latencies = [r["latency_ms"] for r in records]
    return {
        "n": n,
        "overall_accuracy": overall_acc,
        "correct": correct,
        "bucket_accuracy": bucket_acc,
        "abstentions": abstentions,
        "abstention_rate": abstention_rate,
        "confident_and_wrong": confident_and_wrong,
        "confident_and_wrong_rate": confident_and_wrong_rate,
        "per_domain": per_domain,
        "confusion_matrix": {"labels": DOMAINS, "rows_expected_cols_predicted": cm.tolist()},
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "median": float(np.median(latencies)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "max": float(np.max(latencies)) if latencies else 0.0,
        },
    }


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def write_confusion_png(cm, labels, router, out_path, accuracy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(cm, cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("query count", rotation=270, labelpad=15)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("PREDICTED domain", fontsize=12, fontweight="bold")
    ax.set_ylabel("EXPECTED domain", fontsize=12, fontweight="bold")
    ax.set_title(f"Routing confusion matrix — router='{router}'  "
                 f"(top-1 acc={accuracy:.1%}, n={cm.sum()})",
                 fontsize=12, fontweight="bold", pad=14)

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=12,
                    fontweight="bold" if i == j else "normal")
    # gridlines
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_errors_csv(records, out_path):
    errs = [r for r in records if r["predicted"] != r["expected"]]
    errs.sort(key=lambda r: r["id"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "bucket", "expected", "predicted", "confidence",
                    "abstained", "confident_and_wrong", "method", "query", "note"])
        for r in errs:
            abstained = r["predicted"] == ABSTAIN_LABEL
            w.writerow([r["id"], r["bucket"], r["expected"], r["predicted"],
                        f"{r['confidence']:.4f}", int(abstained),
                        int((not abstained) and r["predicted"] != r["expected"]),
                        r["method"], r["query"], r.get("note", "")])
    return len(errs)


def write_report_md(router, metrics, out_path, n_errors, queries_path):
    m = metrics
    L = []
    L.append(f"# Routing eval report — router = `{router}`\n")
    L.append(f"- Query set: `{queries_path}`  (n = {m['n']})")
    L.append(f"- Every number below is emitted from this run into `results.json` "
             f"in the same directory; nothing is hand-entered.\n")

    L.append("## Headline\n")
    L.append(f"- **CONFIDENT-AND-WRONG rate: {m['confident_and_wrong_rate']:.1%}** "
             f"({m['confident_and_wrong']}/{m['n']}) "
             f"— committed to a specialist domain, did not abstain, and was wrong. "
             f"These are the silent failures.")
    L.append(f"- Top-1 accuracy (overall): **{m['overall_accuracy']:.1%}** "
             f"({m['correct']}/{m['n']})")
    L.append(f"- Abstention rate: {m['abstention_rate']:.1%} "
             f"({m['abstentions']}/{m['n']}) — routed to `general` instead of a specialist.\n")

    L.append("## Top-1 accuracy by bucket\n")
    L.append("| bucket | n | correct | accuracy |")
    L.append("|---|---:|---:|---:|")
    for b, v in m["bucket_accuracy"].items():
        L.append(f"| {b} | {v['n']} | {v['correct']} | {v['accuracy']:.1%} |")
    L.append("")

    L.append("## Per-domain precision / recall\n")
    L.append("| domain | support | precision | recall | tp | fp | fn |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in DOMAINS:
        pd = m["per_domain"][d]
        L.append(f"| {d} | {pd['support']} | {pd['precision']:.1%} | {pd['recall']:.1%} "
                 f"| {pd['tp']} | {pd['fp']} | {pd['fn']} |")
    L.append("")

    L.append("## Confusion matrix (rows = expected, cols = predicted)\n")
    labels = m["confusion_matrix"]["labels"]
    cm = m["confusion_matrix"]["rows_expected_cols_predicted"]
    L.append("| exp \\ pred | " + " | ".join(labels) + " |")
    L.append("|---|" + "|".join(["---:"] * len(labels)) + "|")
    for i, d in enumerate(labels):
        L.append(f"| **{d}** | " + " | ".join(str(x) for x in cm[i]) + " |")
    L.append("\nSee `confusion_matrix.png` for the screen-share-friendly version.\n")

    L.append("## Latency (per query, wall clock)\n")
    lat = m["latency_ms"]
    L.append(f"- mean: {lat['mean']:.3f} ms | median: {lat['median']:.3f} ms "
             f"| p95: {lat['p95']:.3f} ms | max: {lat['max']:.3f} ms")
    L.append(f"\n## Errors\n\n{n_errors} misrouted queries listed in `errors.csv` "
             f"(expected vs predicted vs confidence).\n")
    with open(out_path, "w") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Routing evaluation harness")
    ap.add_argument("--router", required=True, choices=sorted(ADAPTERS),
                    help="which router to evaluate")
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--split", default="all", choices=["all", "train", "holdout"],
                    help="evaluate on the full set (default), or the train/holdout "
                         "partition recorded in split.json")
    ap.add_argument("--split-file", default=None, help="override path to split.json")
    ap.add_argument("--out", default=None,
                    help="output dir (default: eval/routing/results/<router>[_<split>])")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    default_name = args.router if args.split == "all" else f"{args.router}_{args.split}"
    out_dir = Path(args.out) if args.out else (HERE / "results" / default_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries, args.split, args.split_file)
    router = build_router(args.router)

    # warm-up (exclude cold-start cost like lazy init from timed latency)
    try:
        router.predict("warmup query about python code")
    except Exception:
        pass

    records = []
    for q in queries:
        t0 = time.perf_counter()
        pred, conf, method = router.predict(q["query"])
        dt = (time.perf_counter() - t0) * 1000.0
        records.append({
            "id": q["id"], "query": q["query"], "bucket": q["bucket"],
            "expected": q["expected_domain"], "predicted": pred,
            "confidence": conf, "method": method, "latency_ms": dt,
            "note": q.get("note", ""),
        })

    metrics = compute_metrics(records)

    results = {
        "router": args.router,
        "queries_path": str(args.queries),
        "split": args.split,
        "seed": SEED,
        "n": metrics["n"],
        "metrics": metrics,
        "predictions": records,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    n_err = write_errors_csv(records, out_dir / "errors.csv")
    write_confusion_png(metrics["confusion_matrix"]["rows_expected_cols_predicted"],
                        metrics["confusion_matrix"]["labels"], args.router,
                        out_dir / "confusion_matrix.png", metrics["overall_accuracy"])
    write_report_md(args.router, metrics, out_dir / "report.md", n_err, args.queries)

    # console summary
    print(f"\n=== router='{args.router}'  split='{args.split}'  n={metrics['n']} ===")
    print(f"top-1 accuracy (overall)   : {metrics['overall_accuracy']:.1%} "
          f"({metrics['correct']}/{metrics['n']})")
    for b, v in metrics["bucket_accuracy"].items():
        print(f"  {b:16s}         : {v['accuracy']:.1%} ({v['correct']}/{v['n']})")
    print(f"abstention rate            : {metrics['abstention_rate']:.1%} "
          f"({metrics['abstentions']}/{metrics['n']})")
    print(f"CONFIDENT-AND-WRONG rate   : {metrics['confident_and_wrong_rate']:.1%} "
          f"({metrics['confident_and_wrong']}/{metrics['n']})   <<< headline")
    print(f"mean latency               : {metrics['latency_ms']['mean']:.3f} ms/query")
    print(f"outputs written to         : {out_dir}")


if __name__ == "__main__":
    main()
