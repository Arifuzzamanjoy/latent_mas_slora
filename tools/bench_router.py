#!/usr/bin/env python3
"""
Score the semantic router against dataset domain labels.

The router picks the agent pipeline, so a router that cannot tell math from code
silently changes what every mas method runs. This measures that directly, with
no model weights and no GPU:

    venv/bin/python tools/bench_router.py
    venv/bin/python tools/bench_router.py --n 300 --temperature 0.2

Two caveats on the number it prints. There is no code dataset in eval/data.py,
so `code` and `general` routing are not exercised — this covers math, medical
and reasoning only. And the gold labels are the ones eval/data.py assigns
(`arc` -> reasoning, `gsm8k` -> math), which is a repo convention rather than
ground truth about which pipeline answers best.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.data import load_dataset  # noqa: E402
from src.routing import SemanticRouter  # noqa: E402

CORPUS = (("gsm8k", "math"), ("medqa", "medical"), ("arc", "reasoning"))


def build_corpus(n, cache_dir, seed=0):
    data = []
    for spec, gold in CORPUS:
        items = load_dataset(spec, cache_dir=cache_dir)
        random.Random(seed).shuffle(items)
        data += [(i.question, gold) for i in items[:n]]
    return data


def evaluate(router, data):
    preds = [(*router.get_best_domain(q), g) for q, g in data]
    preds = [(d.value, c, g) for d, c, g in preds]
    n = len(preds)
    acc = sum(p == g for p, _, g in preds) / n

    domains = sorted({g for *_, g in preds} | {p for p, *_ in preds})
    f1s = []
    for d in domains:
        tp = sum(1 for p, _, g in preds if p == d and g == d)
        fp = sum(1 for p, _, g in preds if p == d and g != d)
        fn = sum(1 for p, _, g in preds if p != d and g == d)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    confs = sorted(c for _, c, _ in preds)
    print(f"\n  items {n}   accuracy {acc:.1%}   macro-F1 {sum(f1s) / len(f1s):.3f}")
    print(
        f"  confidence  min {confs[0]:.3f}   p05 {confs[n // 20]:.3f}   "
        f"median {confs[n // 2]:.3f}   max {confs[-1]:.3f}"
    )

    cm = defaultdict(Counter)
    for p, _, g in preds:
        cm[g][p] += 1
    print("\n  " + "gold / pred".ljust(14) + "".join(f"{d:>11}" for d in domains))
    for g in sorted(cm):
        print("  " + g.ljust(14) + "".join(f"{cm[g][d]:>11}" for d in domains))

    # Calibration: confidence is only useful if it predicts correctness.
    bins = defaultdict(list)
    for p, c, g in preds:
        bins[min(int(c * 5), 4)].append(p == g)
    print("\n  calibration")
    for b in sorted(bins):
        rows = bins[b]
        print(
            f"    conf [{b / 5:.1f}-{(b + 1) / 5:.1f})  n={len(rows):>4}  acc {sum(rows) / len(rows):.0%}"
        )
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150, help="items per dataset")
    ap.add_argument("--cache-dir", default="/home/caches")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--keyword-weight", type=float, default=None)
    a = ap.parse_args()

    kwargs = {}
    if a.temperature is not None:
        kwargs["temperature"] = a.temperature
    if a.keyword_weight is not None:
        kwargs["keyword_weight"] = a.keyword_weight

    data = build_corpus(a.n, a.cache_dir)
    print(f"corpus: {dict(Counter(g for _, g in data))}")
    evaluate(SemanticRouter(**kwargs), data)


if __name__ == "__main__":
    main()
