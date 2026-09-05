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

from eval.data import EvalItem, load_dataset, segment
from eval.extract import UNKNOWN, extract_mcq, extract_numeric, is_correct, majority_vote
from eval.methods import METHOD_REGISTRY, expand_methods
from eval.metrics import calibration, classification_report, mcnemar, wilson_interval
from eval.runner import permutations_for

# ─── extraction ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected,rule",
    [
        ("reasoning...\n\\boxed{C}", "C", "boxed"),
        ("\\boxed{B} then \\boxed{D}", "D", "boxed"),  # last box wins
        ("The final answer is (B).", "B", "answer_is"),
        ("Therefore the correct option is: A", "A", "option_is"),
        ("The answer depends on the context.\nB", "B", "last_line"),
        ("Choice D is correct.", "D", "is_correct"),
        ("[Self-Consistency Vote: A]\n\nblah", "A", "sc_header"),
        ("...ends with\nC", "C", "last_line"),
    ],
)
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


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so the answer is \\boxed{42}", "42"),
        ("#### 1,024", "1024"),
        ("final answer: $18.00", "18"),
    ],
)
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
    return [
        EvalItem(
            id=str(i),
            question=f"q{i}",
            gold="A",
            choices=["a", "b", "c", "d"],
            domain=["medical", "math", "code"][i % 3],
        )
        for i in range(n)
    ]


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
    item = EvalItem(
        id="x",
        question="stem\nA. w\nB. x\nC. y\nD. z",
        gold="C",
        choices=["w", "x", "y", "z"],
        metadata={"stem": "stem"},
    )
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.0", 1.0),
        ("0.5", 0.5),
        ("0.01", 0.01),
        ("0.001", 0.001),
        ("0.0001", 0.0001),
        ("50%", 0.5),
        ("1%", 0.01),
        ("0.1%", 0.001),
    ],
)
def test_parse_fraction(raw, expected):
    assert parse_fraction(raw) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", "200%"])
def test_parse_fraction_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        parse_fraction(bad)


def test_tiny_fraction_still_returns_items():
    pool = _items(1273)  # MedQA-sized
    assert len(segment(pool, fraction=0.001, stratify_by=None)) == 1
    assert len(segment(pool, fraction=0.0001, stratify_by=None)) == 1


def test_tiny_fraction_keeps_one_per_domain_by_default():
    pool = _items(1273)  # 3 domains
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
            "baseline-cot": {
                "n": 20,
                "accuracy": 0.5,
                "ci": {"low": 0.3, "high": 0.7},
                "tokens": {"total_mean": 300, "total_per_correct": 600},
                "latency_ms": {"p50": 1000, "p95": 2000, "mean": 1200},
                "calibration": {
                    "ece": 0.1,
                    "bins": [
                        {"bin": 5, "n": 10, "avg_confidence": 0.55, "accuracy": 0.5},
                        {"bin": 9, "n": 10, "avg_confidence": 0.95, "accuracy": 0.9},
                    ],
                },
                "by_domain": {
                    "math": {"n": 10, "accuracy": 0.6},
                    "code": {"n": 10, "accuracy": 0.4},
                },
            },
            "latent-mas": {
                "n": 20,
                "accuracy": 0.65,
                "ci": {"low": 0.45, "high": 0.85},
                "tokens": {"total_mean": 900, "total_per_correct": 1400},
                "latency_ms": {"p50": 9000, "p95": 12000, "mean": 9500},
                "calibration": {
                    "ece": 0.2,
                    "bins": [
                        {"bin": 5, "n": 10, "avg_confidence": 0.55, "accuracy": 0.4},
                        {"bin": 9, "n": 10, "avg_confidence": 0.95, "accuracy": 0.8},
                    ],
                },
                "by_domain": {
                    "math": {"n": 10, "accuracy": 0.7},
                    "code": {"n": 10, "accuracy": 0.6},
                },
            },
        },
        "comparisons": {
            "latent-mas": {
                "delta": 0.15,
                "ci_low": -0.05,
                "ci_high": 0.35,
                "mcnemar": {"p_value": 0.03, "a_only": 5, "b_only": 1},
            }
        },
    }
    made = make_all(summary, tmp_path)
    names = {Path(p).name for p in made}
    assert {
        "accuracy.png",
        "efficiency.png",
        "latency.png",
        "by_domain.png",
        "deltas.png",
        "calibration.png",
    } <= names
    assert all(Path(p).stat().st_size > 5000 for p in made)


def test_live_plot_is_written(tmp_path):
    from eval.plots import plot_live

    recs = [{"method": "m1", "correct": i % 2 == 0} for i in range(10)]
    recs += [{"method": "m2", "correct": i % 3 == 0} for i in range(10)]
    p = plot_live(recs, tmp_path, total=10)
    assert p and Path(p).exists() and Path(p).stat().st_size > 5000


# ─── id uniqueness (regression: mix: collapsed 150 items to 96) ──────────────

from eval.data import _ensure_unique_ids, _slug


def test_ensure_unique_ids_disambiguates():
    items = [EvalItem(id="x", question="q", gold="A") for _ in range(3)]
    items.append(EvalItem(id="y", question="q", gold="A"))
    out = _ensure_unique_ids(items)
    assert [i.id for i in out] == ["x", "x#2", "x#3", "y"]


def test_ensure_unique_ids_leaves_unique_ids_alone():
    items = [EvalItem(id=f"i{k}", question="q", gold="A") for k in range(5)]
    assert [i.id for i in _ensure_unique_ids(items)] == ["i0", "i1", "i2", "i3", "i4"]


def test_slug_is_id_safe():
    assert _slug("mmlu:college_mathematics") == "mmlu-college_mathematics"
    assert _slug("mix:a,b") == "mix-a-b"


def test_local_file_with_duplicate_ids_is_disambiguated(tmp_path):
    import json as _json

    f = tmp_path / "dup.json"
    f.write_text(
        _json.dumps(
            [
                {"id": 1, "question": "q1\nA. a\nB. b", "gold_letter": "A"},
                {"id": 1, "question": "q2\nA. a\nB. b", "gold_letter": "B"},
            ]
        )
    )
    items = load_dataset(f"local:{f}")
    assert len({i.id for i in items}) == 2, "duplicate ids must not collapse"


def test_duplicate_ids_would_drop_items_from_a_segment():
    """The failure mode the fix prevents: dedup keys collapse a segment."""
    colliding = [EvalItem(id="same", question=f"q{k}", gold="A") for k in range(10)]
    keys = {(("m", 0, i.id)) for i in colliding}
    assert len(keys) == 1  # 10 items, 1 dedup key -> 9 lost
    fixed = _ensure_unique_ids(colliding)
    assert len({("m", 0, i.id) for i in fixed}) == 10


# ─── multi-lora composition ──────────────────────────────────────────────────

from eval.methods.multilora import MultiLoRA


def test_degenerate_probe_detects_identical_adapters():
    """Zero-init LoRA is an identity function: every adapter scores the same."""
    same = [("a", 305.123), ("b", 305.123), ("c", 305.123)]
    assert MultiLoRA.is_degenerate(same)


def test_degenerate_probe_passes_distinguishable_adapters():
    assert not MultiLoRA.is_degenerate([("a", 300.0), ("b", 312.0), ("c", 295.0)])
    assert not MultiLoRA.is_degenerate([("a", 1.0)])  # single adapter: n/a


def test_compose_takes_top_k_with_normalized_weights():
    m = MultiLoRA.__new__(MultiLoRA)
    m.top_k = 2
    names, weights = m._compose([("a", 1.0), ("b", 9.0), ("c", 5.0)])
    assert names == ["b", "c"]  # ranked, truncated
    assert weights[0] > weights[1]  # higher score, higher weight
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


def test_compose_handles_negative_scores():
    """entropy mode yields negative scores; weights must stay non-negative."""
    m = MultiLoRA.__new__(MultiLoRA)
    m.top_k = 3
    _, weights = m._compose([("a", -5.0), ("b", -1.0), ("c", -3.0)])
    assert all(w >= 0 for w in weights) and abs(sum(weights) - 1.0) < 1e-6


def test_ablation_ladder_changes_one_thing_per_rung():
    """latent-mas -> latent-mas-kv -> latent-mas-slora must differ by one knob."""
    from eval.methods import METHOD_REGISTRY

    rungs = ["latent-mas", "latent-mas-kv", "latent-mas-slora"]
    d = [METHOD_REGISTRY[r].defaults for r in rungs]
    assert d[0] == {"kv_handoff": False, "prompt_style": "answer_first"}
    assert d[1] == {"kv_handoff": True, "prompt_style": "answer_first"}
    assert d[2] == {"kv_handoff": True, "prompt_style": "reason_first"}
    for lo, hi in zip(d, d[1:]):
        assert sum(lo[k] != hi[k] for k in lo) == 1, "each rung changes exactly one knob"


def test_judger_prompt_styles_differ_as_intended():
    pytest.importorskip("torch", reason="src package imports torch at package level")
    from src.agents.configs import AgentConfig

    rf = AgentConfig.judger()
    af = AgentConfig.judger(prompt_style="answer_first")
    assert rf.prompt_style == "reason_first"  # new default
    # the assertion used to look for wording this template has never contained
    assert "State your final answer FIRST" in af.user_prompt_template
    assert "State your final answer FIRST" not in rf.user_prompt_template
    assert "A, B, C, or D" not in rf.system_prompt  # works for numeric too


def test_multilora_candidates_include_externally_loaded_adapters():
    """--loras adapters must be able to participate, not just sit in memory."""
    from eval.methods.multilora import MultiLoRA

    class _Pool:
        def list_agents(self):
            return ["Planner"]

        def get(self, n):
            return type("C", (), {"adapter_name": "planner_lora"})()

    class _Model:
        peft_config = {"planner_lora": 1, "reasoning_lora": 1, "_logo_mix": 1}

    m = MultiLoRA.__new__(MultiLoRA)
    m.system = type("S", (), {"_pool": _Pool(), "model": _Model()})()
    got = m._candidates()
    assert "reasoning_lora" in got, "externally loaded adapter must be reachable"
    assert "_logo_mix" not in got, "the merge target must never be a candidate"


def test_compose_never_picks_an_identity_adapter_over_a_real_one():
    """Regression: top_k=1 used to select an untrained adapter by tie-break."""
    from eval.methods.multilora import MultiLoRA

    m = MultiLoRA.__new__(MultiLoRA)
    m.top_k = 1
    names, _ = m._compose([("planner_lora", 0.0), ("medical_lora", 0.0), ("reasoning_lora", 149.9)])
    assert names == ["reasoning_lora"]


def test_compose_ties_break_deterministically_by_name():
    from eval.methods.multilora import MultiLoRA

    m = MultiLoRA.__new__(MultiLoRA)
    m.top_k = 2
    a, _ = m._compose([("b_lora", 5.0), ("a_lora", 5.0), ("c_lora", 5.0)])
    b, _ = m._compose([("c_lora", 5.0), ("b_lora", 5.0), ("a_lora", 5.0)])
    assert a == b == ["a_lora", "b_lora"], "order must not depend on registration order"


def test_all_zero_scores_are_degenerate():
    from eval.methods.multilora import MultiLoRA

    assert MultiLoRA.is_degenerate([("a", 0.0), ("b", 0.0), ("c", 0.0)])


def test_compose_gives_every_selected_adapter_nonzero_weight():
    """Regression: min-subtraction zeroed the lowest survivor, making top_k -> top_k-1."""
    from eval.methods.multilora import MultiLoRA

    m = MultiLoRA.__new__(MultiLoRA)
    m.top_k = 3
    names, w = m._compose([("a", 10.0), ("b", 6.0), ("c", 2.0)])
    assert len(names) == 3
    assert all(x > 0.0 for x in w), f"every selected adapter must contribute: {w}"
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[1] > w[2]  # weight follows score


def test_compose_uniform_when_every_score_is_zero():
    from eval.methods.multilora import MultiLoRA

    m = MultiLoRA.__new__(MultiLoRA)
    m.top_k = 2
    _, w = m._compose([("a", 0.0), ("b", 0.0)])
    assert w == [0.5, 0.5]


def test_old_method_name_still_resolves():
    """latent-mas-paper was renamed; existing commands must keep working."""
    assert expand_methods(["latent-mas-paper"]) == ["latent-mas-slora"]
    assert "latent-mas-slora" in expand_methods(["ladder"])
    assert "latent-mas-paper" not in expand_methods(["ladder"])
