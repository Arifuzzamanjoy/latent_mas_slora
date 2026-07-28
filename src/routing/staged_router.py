"""
Staged Router - confidence-gated two-stage routing with an explicit abstain path.

This is a NEW wrapper. It does NOT modify or change the default behaviour of any
existing router; the baseline routers remain exactly as they were for before/after
comparison.

Motivation
----------
A single-shot keyword router commits to a domain on every query, including
ambiguous or out-of-domain ones. Wrong-but-confident routes are SILENT failures:
the wrong adapter would be loaded and nobody sees a signal. Staged routing adds
(a) a cheap first pass, (b) a more expensive second pass only when the first is
not confident, and (c) an explicit "unsure" outcome when neither pass clears a
floor -- turning some silent, confident-wrong routes into visible abstentions.

Stages
------
  Stage 1 (cheap):  keyword FastRouter (~micro-seconds, zero ML deps).
                    Accept if it commits to a specialist with confidence >= STAGE1_ACCEPT.
  Stage 2 (costlier): a local TF-IDF cosine matcher over each domain's exemplar
                    prompts + keywords (pure numpy, no network, no torch, no model
                    download). Accept its best specialist if cosine >= STAGE2_ACCEPT.
  Unsure:           if stage 2 is also below the floor -> return 'general'
                    (the abstain / "don't force a pick" outcome).

NOTE ON THE SEMANTIC STAGE: stage 2 is a *lexical* semantic matcher (TF-IDF), not
a neural sentence-embedding model. It is used here because the neural embedding
model (all-MiniLM-L6-v2) cannot be fetched in the offline CI/eval environment.
The staging/abstain LOGIC is independent of the stage-2 implementation: swap in
the embedding router as stage 2 on a networked machine and the same thresholds and
escalation path apply. Do not overstate stage 2 as neural.

All thresholds live in STAGED_CONFIG below (no scattered magic numbers).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- imports that work both as a package module and as a stand-alone file -----
# (loading as a package would execute src/routing/__init__.py, which imports the
#  torch-based routers; the eval harness loads this file directly to stay offline.)
try:  # normal package import
    from .domain_profiles import Domain, DOMAIN_PROFILES
    from .fast_router import FastRouter
except ImportError:  # stand-alone import by file path
    import importlib.util
    import pathlib
    _HERE = pathlib.Path(__file__).resolve().parent

    def _load(mod_name, filename):
        spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _dp = _load("_staged_domain_profiles", "domain_profiles.py")
    Domain, DOMAIN_PROFILES = _dp.Domain, _dp.DOMAIN_PROFILES
    FastRouter = _load("_staged_fast_router", "fast_router.py").FastRouter


# =============================================================================
# CONFIG  (all thresholds in one place)
# =============================================================================
@dataclass(frozen=True)
class StagedConfig:
    # Stage 1 (keyword) confidence needed to accept without escalating.
    # FastRouter confidence = (winning-domain keyword hits) / (all keyword hits),
    # i.e. how dominant the winning domain is among matched keywords.
    stage1_accept: float = 0.70

    # Stage 2 (TF-IDF cosine) similarity needed to accept a specialist.
    stage2_accept: float = 0.18

    # Specialist domains stage 2 chooses among. 'general' is never a stage-2
    # target; it is only produced by the abstain path.
    specialists: Tuple[str, ...] = ("code", "math", "medical", "finance", "reasoning")

    # Provenance of the default thresholds (be honest in artifacts):
    thresholds_provenance: str = (
        "Chosen by an in-sample sweep over eval/routing/queries.jsonl (see "
        "results/staged/threshold_sweep.json). NOT held-out; a production "
        "deployment must recalibrate on a separate validation set."
    )


STAGED_CONFIG = StagedConfig()

ABSTAIN = "general"


# =============================================================================
# Result type
# =============================================================================
@dataclass
class StagedResult:
    domain: str                 # 'code'|'math'|'medical'|'finance'|'reasoning'|'general'
    confidence: float           # confidence of the ACCEPTED stage (or stage-2 conf when unsure)
    method: str                 # 'staged:stage1' | 'staged:stage2' | 'staged:unsure'
    stage: int                  # 1, 2, or 0 (unsure)
    stage1_domain: str = ""
    stage1_conf: float = 0.0
    stage2_domain: str = ""
    stage2_conf: float = 0.0
    all_stage2_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            "domain": self.domain, "confidence": self.confidence, "method": self.method,
            "stage": self.stage, "stage1_domain": self.stage1_domain,
            "stage1_conf": self.stage1_conf, "stage2_domain": self.stage2_domain,
            "stage2_conf": self.stage2_conf, "all_stage2_scores": self.all_stage2_scores,
        }


# =============================================================================
# Pure-numpy-free TF-IDF over fixed domain corpora (deterministic, offline)
# =============================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Non-discriminative English function words. Dropped from BOTH the domain corpora
# and the query so that out-of-domain queries made only of common words score ~0
# (and therefore abstain) instead of matching a domain by stopword overlap.
_STOPWORDS = frozenset("""
a an the of to in on for and or but with my your our i we you it its this that these
those is are was were be been being do does did done how what whats which who whom
whose when where why can could should would will shall may might must me us them they
he she his her as at by from into over under about after before between vs versus if
then than so such not no yes get got give show tell explain please help need want
""".split())


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) >= 2 and t not in _STOPWORDS]


class _DomainTfidf:
    """Lexical-semantic matcher: cosine similarity between a query and each
    specialist domain's corpus (exemplar prompts + keywords). Deterministic."""

    def __init__(self, specialists: Tuple[str, ...]):
        self.specialists = list(specialists)
        # Build one document per specialist domain from DOMAIN_PROFILES.
        docs: Dict[str, List[str]] = {}
        for dstr in self.specialists:
            dom = Domain(dstr)
            profile = DOMAIN_PROFILES[dom]
            toks: List[str] = []
            for p in profile.exemplar_prompts:
                toks += _tokenize(p)
            for kw in profile.keywords:
                toks += _tokenize(kw)
            docs[dstr] = toks

        # Vocabulary + document frequency.
        vocab = {}
        for dstr in self.specialists:
            for t in set(docs[dstr]):
                vocab[t] = vocab.get(t, 0) + 1  # df
        self.vocab = {t: i for i, t in enumerate(sorted(vocab))}
        N = len(self.specialists)
        # standard idf = log(N/df); a term present in every domain doc (df==N)
        # gets idf 0 and cannot drive similarity (kills stopword/common-word matches).
        self.idf = {t: math.log(N / df) for t, df in vocab.items()}

        # Precompute L2-normalised tf-idf vectors per domain (as sparse dicts).
        self.domain_vecs: Dict[str, Dict[int, float]] = {}
        for dstr in self.specialists:
            self.domain_vecs[dstr] = self._vectorize(docs[dstr])

    def _vectorize(self, tokens: List[str]) -> Dict[int, float]:
        if not tokens:
            return {}
        counts: Dict[int, int] = {}
        for t in tokens:
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        length = len(tokens)
        # tf-idf
        vec: Dict[int, float] = {}
        # map vocab index back to token idf: build reverse once
        inv = getattr(self, "_inv", None)
        if inv is None:
            inv = {i: t for t, i in self.vocab.items()}
            self._inv = inv
        for j, c in counts.items():
            vec[j] = (c / length) * self.idf[inv[j]]
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for j in vec:
                vec[j] /= norm
        return vec

    @staticmethod
    def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        # iterate smaller dict
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(j, 0.0) for j, v in a.items())

    def score(self, query: str) -> Dict[str, float]:
        qv = self._vectorize(_tokenize(query))
        return {d: self._cosine(qv, self.domain_vecs[d]) for d in self.specialists}


# =============================================================================
# Staged router
# =============================================================================
class StagedRouter:
    def __init__(self, config: StagedConfig = STAGED_CONFIG):
        self.config = config
        self._fast = FastRouter()
        self._tfidf = _DomainTfidf(config.specialists)

    # -- stage 2 as a standalone call (also usable to swap in a neural stage) --
    def _stage2(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        scores = self._tfidf.score(query)
        best = max(scores, key=scores.get)
        return best, float(scores[best]), {k: float(v) for k, v in scores.items()}

    def route(self, query: str) -> StagedResult:
        cfg = self.config

        # ---- Stage 1: cheap keyword pass ----
        s1 = self._fast.route_detailed(query)
        s1_domain, s1_conf = s1.domain.value, float(s1.confidence)

        if s1_domain != ABSTAIN and s1_conf >= cfg.stage1_accept:
            return StagedResult(
                domain=s1_domain, confidence=s1_conf, method="staged:stage1", stage=1,
                stage1_domain=s1_domain, stage1_conf=s1_conf,
            )

        # ---- Stage 2: escalate to the costlier semantic pass ----
        s2_domain, s2_conf, s2_scores = self._stage2(query)

        if s2_conf >= cfg.stage2_accept:
            return StagedResult(
                domain=s2_domain, confidence=s2_conf, method="staged:stage2", stage=2,
                stage1_domain=s1_domain, stage1_conf=s1_conf,
                stage2_domain=s2_domain, stage2_conf=s2_conf, all_stage2_scores=s2_scores,
            )

        # ---- Unsure: below the floor on both stages -> abstain ----
        return StagedResult(
            domain=ABSTAIN, confidence=s2_conf, method="staged:unsure", stage=0,
            stage1_domain=s1_domain, stage1_conf=s1_conf,
            stage2_domain=s2_domain, stage2_conf=s2_conf, all_stage2_scores=s2_scores,
        )

    # Backwards-compatible convenience API (mirrors existing routers).
    def get_best_domain(self, prompt: str, confidence_threshold: float = 0.0
                        ) -> Tuple[Domain, float]:
        r = self.route(prompt)
        return Domain(r.domain), r.confidence


# convenience singleton
_staged: Optional[StagedRouter] = None


def get_staged_router() -> StagedRouter:
    global _staged
    if _staged is None:
        _staged = StagedRouter()
    return _staged


def staged_route(prompt: str) -> Tuple[str, float]:
    r = get_staged_router().route(prompt)
    return r.domain, r.confidence


if __name__ == "__main__":
    r = StagedRouter()
    for q in ["how do I reverse a string in python",
              "btc price today",
              "whats the capital of france",
              "is it logically sound to buy a stock because it went up yesterday"]:
        res = r.route(q)
        print(f"{q[:45]:47s} -> {res.domain:9s} {res.method:15s} conf={res.confidence:.3f}")
