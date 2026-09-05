"""
Metrics and statistics.

Accuracy alone cannot separate two methods on a few hundred items, so every
headline number carries an interval, and every method-vs-method claim uses a
paired test on the items both methods actually saw.
"""

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

# ─── Intervals ───────────────────────────────────────────────────────────────


def wilson_interval(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval - well behaved at small n and near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963985 if abs(conf - 0.95) < 1e-6 else _z_for(conf)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _z_for(conf: float) -> float:
    # inverse normal CDF (Acklam), adequate for reporting
    p = 1 - (1 - conf) / 2
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def bootstrap_interval(
    correct: Sequence[bool], n_boot: int = 2000, conf: float = 0.95, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap over items."""
    n = len(correct)
    if n == 0 or n_boot <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    vals = [1.0 if c else 0.0 for c in correct]
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - conf) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + conf) / 2 * n_boot))]
    return (lo, hi)


# ─── Paired tests ────────────────────────────────────────────────────────────


def mcnemar(
    a_correct: Sequence[bool], b_correct: Sequence[bool], exact: bool = True
) -> Dict[str, Any]:
    """
    Exact McNemar test on paired item outcomes.

    This is the right test when two methods answer the same questions: it looks
    only at the items where they disagree. Comparing two independent-sample
    confidence intervals instead will call real differences insignificant.
    """
    assert len(a_correct) == len(b_correct), "paired test needs aligned items"
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)  # a wins
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)  # b wins
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0, "test": "none"}

    if exact and n <= 200:
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
        p = min(1.0, 2 * tail)
        test = "exact_binomial"
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / n
        p = math.erfc(math.sqrt(chi2 / 2))
        test = "chi2_continuity_corrected"

    return {"a_only": b01, "b_only": b10, "discordant": n, "p_value": round(p, 6), "test": test}


def paired_bootstrap_delta(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    n_boot: int = 2000,
    conf: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Bootstrap CI for (acc_a - acc_b), resampling items jointly."""
    n = len(a_correct)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_correct[i] for i in idx) / n
        db = sum(b_correct[i] for i in idx) / n
        deltas.append(da - db)
    deltas.sort()
    return {
        "delta": sum(a_correct) / n - sum(b_correct) / n,
        "ci_low": deltas[int((1 - conf) / 2 * n_boot)],
        "ci_high": deltas[min(n_boot - 1, int((1 + conf) / 2 * n_boot))],
    }


# ─── Calibration ─────────────────────────────────────────────────────────────


def calibration(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> Dict[str, Any]:
    """
    ECE and Brier score.

    With self-consistency the vote share is a usable confidence signal: if 3/3
    samples agree the method should be right more often than when it is 2/3.
    A method that is no better calibrated than chance is voting noise.
    """
    n = len(confidences)
    if n == 0:
        return {"ece": None, "brier": None, "bins": []}

    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(confidences, correct)) / n

    buckets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(confidences, correct):
        b = min(bins - 1, int(c * bins))
        buckets[b].append((c, ok))

    ece = 0.0
    rows = []
    for b in sorted(buckets):
        pts = buckets[b]
        avg_conf = sum(c for c, _ in pts) / len(pts)
        acc = sum(1 for _, ok in pts if ok) / len(pts)
        ece += len(pts) / n * abs(avg_conf - acc)
        rows.append(
            {
                "bin": b,
                "n": len(pts),
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(acc, 4),
            }
        )

    return {"ece": round(ece, 4), "brier": round(brier, 4), "bins": rows}


# ─── Classification (for router-only) ────────────────────────────────────────


def classification_report(preds: Sequence[str], golds: Sequence[str]) -> Dict[str, Any]:
    labels = sorted(set(golds) | set(preds))
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for p, g in zip(preds, golds):
        matrix[g][p] += 1

    per_label = {}
    f1s = []
    for lab in labels:
        tp = matrix[lab][lab]
        fp = sum(matrix[g][lab] for g in labels if g != lab)
        fn = sum(matrix[lab][p] for p in labels if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(matrix[lab].values())
        per_label[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1s.append(f1)

    acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds) if golds else 0.0
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_label": per_label,
        "confusion": matrix,
        "labels": labels,
    }


# ─── Aggregation ─────────────────────────────────────────────────────────────


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def summarize_records(
    records: List[Dict[str, Any]], bootstrap: int = 2000, conf: float = 0.95, seed: int = 0
) -> Dict[str, Any]:
    """Aggregate per-item records for one method into a metrics block."""
    if not records:
        return {"n": 0}

    correct = [bool(r["correct"]) for r in records]
    n, k = len(correct), sum(correct)
    acc = k / n

    lat = [r["latency_ms"] for r in records]
    prompt_toks = [r["prompt_tokens"] for r in records]
    comp_toks = [r["completion_tokens"] for r in records]
    tot_toks = [r["total_tokens"] for r in records]

    confs = [r.get("vote_confidence", 1.0) for r in records]

    by_domain: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r.get("domain", "?")].append(r)
    for dom, rows in sorted(groups.items()):
        kk = sum(1 for r in rows if r["correct"])
        lo, hi = wilson_interval(kk, len(rows), conf)
        by_domain[dom] = {
            "n": len(rows),
            "correct": kk,
            "accuracy": round(kk / len(rows), 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        }

    wl, wh = wilson_interval(k, n, conf)
    bl, bh = bootstrap_interval(correct, bootstrap, conf, seed)

    total_completion = sum(comp_toks)
    total_all = sum(tot_toks)

    return {
        "n": n,
        "correct": k,
        "accuracy": round(acc, 4),
        "ci": {"method": "wilson", "low": round(wl, 4), "high": round(wh, 4), "level": conf},
        "bootstrap_ci": {"low": round(bl, 4), "high": round(bh, 4), "n_boot": bootstrap},
        "parse_failure_rate": round(sum(1 for r in records if r.get("extract_failed")) / n, 4),
        # lm-evaluation-harness convention: "strict" counts only answers the
        # model emitted in the requested \boxed{} format, "accuracy" above is the
        # flexible variant that also accepts an answer stated in prose. A gap
        # between them is a prompt problem, not a reasoning result.
        "strict_accuracy": round(
            sum(1 for r in records if r["correct"] and r.get("extract_strict", True)) / n, 4
        ),
        "format_violation_rate": round(sum(1 for r in records if r.get("format_violation")) / n, 4),
        "unknown_rate": round(sum(1 for r in records if r.get("pred") == "UNKNOWN") / n, 4),
        "tokens": {
            "prompt_total": sum(prompt_toks),
            "completion_total": total_completion,
            "total": total_all,
            "prompt_mean": round(sum(prompt_toks) / n, 1),
            "completion_mean": round(total_completion / n, 1),
            "total_mean": round(total_all / n, 1),
            "completion_per_correct": round(total_completion / k, 1) if k else None,
            "total_per_correct": round(total_all / k, 1) if k else None,
        },
        "latency_ms": {
            "mean": round(sum(lat) / n, 1),
            "p50": percentile(lat, 0.50),
            "p95": percentile(lat, 0.95),
            "total_s": round(sum(lat) / 1000, 1),
        },
        "calibration": calibration(confs, correct),
        "by_domain": by_domain,
        "extract_rules": dict(Counter(r.get("extract_rule", "?") for r in records)),
    }
