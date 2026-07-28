"""
Unit tests for the staged router (threshold logic, escalation, unsure path).

Runs with pytest OR as a plain script (`python tests/test_staged_router.py`) so
CI needs no extra dependency. The staged router is loaded directly by file path
to avoid executing src/routing/__init__.py (which imports the torch-based routers);
these tests are CPU-only and offline.
"""
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / "src" / "routing" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


sr = _load("staged_router_under_test", "staged_router.py")
StagedRouter = sr.StagedRouter
StagedConfig = sr.StagedConfig


# --- Stage 1 (threshold) logic -------------------------------------------------
def test_stage1_accepts_strong_single_domain_keyword():
    r = StagedRouter()  # defaults
    res = r.route("how do I reverse a string in python")
    assert res.domain == "code"
    assert res.stage == 1
    assert res.method == "staged:stage1"
    assert res.confidence >= r.config.stage1_accept


def test_stage1_confidence_below_threshold_does_not_accept_at_stage1():
    # Impossible stage-1 bar forces every query to escalate past stage 1.
    r = StagedRouter(StagedConfig(stage1_accept=1.01, stage2_accept=0.0))
    res = r.route("how do I reverse a string in python")
    assert res.stage != 1  # did not accept at stage 1


# --- Escalation path -----------------------------------------------------------
def test_escalation_to_stage2_returns_specialist():
    # Never accept at stage 1; accept any positive stage-2 similarity.
    r = StagedRouter(StagedConfig(stage1_accept=1.01, stage2_accept=0.0))
    res = r.route("how do I reverse a string in python")
    assert res.stage == 2
    assert res.method == "staged:stage2"
    assert res.domain == "code"
    assert res.stage2_conf > 0.0


# --- Unsure / abstain path -----------------------------------------------------
def test_unsure_when_both_stages_below_floor():
    # Both bars impossible -> must abstain to 'general'.
    r = StagedRouter(StagedConfig(stage1_accept=1.01, stage2_accept=1.01))
    res = r.route("solve x squared minus 5x plus 6 equals 0")
    assert res.domain == "general"
    assert res.method == "staged:unsure"
    assert res.stage == 0


def test_out_of_domain_query_abstains_with_defaults():
    r = StagedRouter()
    res = r.route("whats the capital of france")
    assert res.domain == "general"
    assert res.method == "staged:unsure"


# --- Result shape / backwards-compat ------------------------------------------
def test_result_exposes_confidence_and_serializes():
    r = StagedRouter()
    res = r.route("what are the symptoms of appendicitis")
    d = res.to_dict()
    for key in ("domain", "confidence", "method", "stage", "stage2_conf"):
        assert key in d
    assert isinstance(res.confidence, float)


def test_get_best_domain_backwards_compatible_tuple():
    r = StagedRouter()
    dom, conf = r.get_best_domain("btc price today")
    assert dom.value == "finance"
    assert isinstance(conf, float)


# --- Determinism ---------------------------------------------------------------
def test_routing_is_deterministic():
    r = StagedRouter()
    q = "explain how ethereum gas fees work"
    a = r.route(q)
    b = r.route(q)
    assert (a.domain, a.method, round(a.confidence, 9)) == \
           (b.domain, b.method, round(b.confidence, 9))


def test_existing_router_default_behaviour_untouched():
    # Sanity: the staged wrapper does not require or alter the baseline FastRouter.
    fr = _load("fast_router_check", "fast_router.py")
    dom, conf = fr.FastRouter().route("write a python function")
    assert dom.value == "code"


if __name__ == "__main__":
    # Minimal runner so this passes in CI without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
