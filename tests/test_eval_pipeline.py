"""
Tests for the eval pipeline itself.

These cover the parts that silently corrupt results if they break: answer
extraction, segment nesting, permutation gold remapping, and the statistics.
Run with:  venv/bin/python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from eval.data import EvalItem, segment, load_dataset
from eval.extract import extract_mcq, extract_numeric, is_correct, majority_vote, UNKNOWN
from eval.metrics import mcnemar, wilson_interval, calibration, classification_report
from eval.methods import METHOD_REGISTRY, expand_methods
from eval.runner import permutations_for


# ─── extraction ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected,rule", [
    ("reasoning...\n\\boxed{C}", "C", "boxed"),
    ("\\boxed{B} then \\boxed{D}", "D", "boxed"),          # last box wins
    ("The final answer is (B).", "B", "answer_is"),
    ("Therefore the correct option is: A", "A", "option_is"),
    ("The answer depends on the context.\nB", "B", "last_line"),
    ("Choice D is correct.", "D", "is_correct"),
    ("[Self-Consistency Vote: A]\n\nblah", "A", "sc_header"),
    ("...ends with\nC", "C", "last_line"),
])
def test_mcq_rules(text, expected, rule):
    r = extract_mcq(text)
    assert (r.answer, r.rule) == (expected, rule)


def test_mcq_failure_is_reported_not_guessed():
    r = extract_mcq("I am not sure about this one.")
    assert r.answer == UNKNOWN and r.failed


def test_mcq_respects_choice_count():
    # 10-option datasets must be able to return E-J
    assert extract_mcq("\\boxed{H}", num_choices=10).answer == "H"
    assert extract_mcq("\\boxed{H}", num_choices=4).answer == UNKNOWN


@pytest.mark.parametrize("text,expected", [
    ("so the answer is \\boxed{42}", "42"),
    ("#### 1,024", "1024"),
    ("final answer: $18.00", "18"),
])
def test_numeric(text, expected):
    assert extract_numeric(text).answer == expected


def test_voting_ignores_unknown_but_survives_all_unknown():
    assert majority_vote(["A", UNKNOWN, "A"])[0] == "A"
    assert majority_vote([UNKNOWN, UNKNOWN])[0] == UNKNOWN
    _, _, agreement = majority_vote(["A", "B", "A"])
    assert agreement == pytest.approx(2 / 3)


def test_unknown_never_scores_correct():
    assert not is_correct(UNKNOWN, "UNKNOWN", "mcq")


# ─── segmentation ────────────────────────────────────────────────────────────

def _items(n=100):
    return [EvalItem(id=str(i), question=f"q{i}", gold="A",
                     choices=["a", "b", "c", "d"],
                     domain=["medical", "math", "code"][i % 3]) for i in range(n)]


def test_fractions_are_nested():
    pool = _items()
    small = {i.id for i in segment(pool, fraction=0.1, seed=7)}
    mid = {i.id for i in segment(pool, fraction=0.5, seed=7)}
    full = {i.id for i in segment(pool, fraction=1.0, seed=7)}
    assert small < mid < full, "a larger fraction must only add items"


def test_segment_is_deterministic_per_seed():
    pool = _items()
    a = [i.id for i in segment(pool, fraction=0.3, seed=1)]
    b = [i.id for i in segment(pool, fraction=0.3, seed=1)]
    c = [i.id for i in segment(pool, fraction=0.3, seed=2)]
    assert a == b and a != c


def test_stratification_preserves_domain_mix():
    pool = _items(99)
    seg = segment(pool, fraction=0.3, seed=0, stratify_by="domain")
    counts = {}
    for i in seg:
        counts[i.domain] = counts.get(i.domain, 0) + 1
    assert len(counts) == 3 and max(counts.values()) - min(counts.values()) <= 1


def test_limit_and_offset():
    pool = _items()
    assert len(segment(pool, limit=7)) == 7
    assert segment(pool, offset=10, shuffle=False)[0].id == "10"


def test_sample_dataset_loads():
    items = load_dataset("sample")
    assert len(items) == 5
    assert all(i.gold in "ABCD" and len(i.choices) == 4 for i in items)


# ─── permutation ─────────────────────────────────────────────────────────────

def test_permutation_moves_the_gold_letter():
    item = EvalItem(id="x", question="stem\nA. w\nB. x\nC. y\nD. z", gold="C",
                    choices=["w", "x", "y", "z"], metadata={"stem": "stem"})
    perms = permutations_for(item, "cyclic")
    assert len(perms) == 4
    for p in perms:
        v = item.rendered(p)
        # the gold text must still be the gold letter after permutation
        assert v.choices["ABCD".index(v.gold)] == "y"


def test_permutation_none_is_identity():
    item = _items(1)[0]
    assert permutations_for(item, "none") == [None]


# ─── metrics ─────────────────────────────────────────────────────────────────

def test_mcnemar_uses_only_discordant_pairs():
    a = [True] * 10 + [True, True, False]
    b = [True] * 10 + [False, False, True]
    r = mcnemar(a, b)
    assert r["a_only"] == 2 and r["b_only"] == 1 and r["discordant"] == 3


def test_mcnemar_identical_methods_is_not_significant():
    a = [True, False] * 20
    assert mcnemar(a, list(a))["p_value"] == 1.0


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(3, 5)
    assert lo < 0.6 < hi and 0.0 <= lo and hi <= 1.0


def test_wilson_narrows_with_n():
    small = wilson_interval(6, 10)
    large = wilson_interval(600, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_calibration_perfect_confidence():
    c = calibration([1.0, 1.0, 1.0], [True, True, True])
    assert c["ece"] == 0.0 and c["brier"] == 0.0


def test_classification_report_macro_f1():
    r = classification_report(["a", "b", "a"], ["a", "b", "b"])
    assert r["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["confusion"]["b"]["a"] == 1


# ─── registry ────────────────────────────────────────────────────────────────

def test_groups_expand_and_dedupe():
    got = expand_methods(["baselines", "baseline-cot"])
    assert got[0] == "baseline-direct" and got.count("baseline-cot") == 1


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        expand_methods(["not-a-method"])


def test_every_method_declares_a_backend():
    for name, cls in METHOD_REGISTRY.items():
        assert cls.backend_kind in {"hf", "system", "mock", "none"}, name
        assert cls.description, name


# ─── tiny fractions ──────────────────────────────────────────────────────────

from eval.config import parse_fraction


@pytest.mark.parametrize("raw,expected", [
    ("1.0", 1.0), ("0.5", 0.5), ("0.01", 0.01), ("0.001", 0.001),
    ("0.0001", 0.0001), ("50%", 0.5), ("1%", 0.01), ("0.1%", 0.001),
])
def test_parse_fraction(raw, expected):
    assert parse_fraction(raw) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", "200%"])
def test_parse_fraction_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        parse_fraction(bad)


def test_tiny_fraction_still_returns_items():
    pool = _items(1273)                       # MedQA-sized
    assert len(segment(pool, fraction=0.001, stratify_by=None)) == 1
    assert len(segment(pool, fraction=0.0001, stratify_by=None)) == 1


def test_tiny_fraction_keeps_one_per_domain_by_default():
    pool = _items(1273)                       # 3 domains
    assert len(segment(pool, fraction=0.001)) == 3
    assert len(segment(pool, fraction=0.001, min_per_group=0)) == 0


def test_tiny_fractions_are_still_nested():
    pool = _items(1273)
    a = {i.id for i in segment(pool, fraction=0.001, seed=3)}
    b = {i.id for i in segment(pool, fraction=0.01, seed=3)}
    c = {i.id for i in segment(pool, fraction=0.1, seed=3)}
    assert a < b < c


# ─── plots ───────────────────────────────────────────────────────────────────

def test_plots_are_written(tmp_path):
    from eval.plots import make_all
    summary = {
        "segment": {"spec": "unit", "n": 20, "by_domain": {"math": 10, "code": 10}},
        "reference_method": "baseline-cot",
        "methods": {
            "baseline-cot": {"n": 20, "accuracy": 0.5,
                             "ci": {"low": 0.3, "high": 0.7},
                             "tokens": {"total_mean": 300, "total_per_correct": 600},
                             "latency_ms": {"p50": 1000, "p95": 2000, "mean": 1200},
                             "calibration": {"ece": 0.1, "bins": [
                                 {"bin": 5, "n": 10, "avg_confidence": 0.55, "accuracy": 0.5},
                                 {"bin": 9, "n": 10, "avg_confidence": 0.95, "accuracy": 0.9}]},
                             "by_domain": {"math": {"n": 10, "accuracy": 0.6},
                                           "code": {"n": 10, "accuracy": 0.4}}},
            "latent-mas": {"n": 20, "accuracy": 0.65,
                           "ci": {"low": 0.45, "high": 0.85},
                           "tokens": {"total_mean": 900, "total_per_correct": 1400},
                           "latency_ms": {"p50": 9000, "p95": 12000, "mean": 9500},
                           "calibration": {"ece": 0.2, "bins": [
                               {"bin": 5, "n": 10, "avg_confidence": 0.55, "accuracy": 0.4},
                               {"bin": 9, "n": 10, "avg_confidence": 0.95, "accuracy": 0.8}]},
                           "by_domain": {"math": {"n": 10, "accuracy": 0.7},
                                         "code": {"n": 10, "accuracy": 0.6}}},
        },
        "comparisons": {"latent-mas": {"delta": 0.15, "ci_low": -0.05, "ci_high": 0.35,
                                       "mcnemar": {"p_value": 0.03, "a_only": 5, "b_only": 1}}},
    }
    made = make_all(summary, tmp_path)
    names = {Path(p).name for p in made}
    assert {"accuracy.png", "efficiency.png", "latency.png", "by_domain.png",
            "deltas.png", "calibration.png"} <= names
    assert all(Path(p).stat().st_size > 5000 for p in made)


def test_live_plot_is_written(tmp_path):
    from eval.plots import plot_live
    recs = [{"method": "m1", "correct": i % 2 == 0} for i in range(10)]
    recs += [{"method": "m2", "correct": i % 3 == 0} for i in range(10)]
    p = plot_live(recs, tmp_path, total=10)
    assert p and Path(p).exists() and Path(p).stat().st_size > 5000
