"""
Semantic Router - State-of-the-Art Prompt-Based Routing

Uses embedding similarity + keyword boosting for intelligent
domain/pipeline selection.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import torch

from .domain_profiles import DOMAIN_PROFILES, Domain, DomainProfile


class SemanticRouter:
    """
    State-of-the-art embedding-based router for LoRA/pipeline selection.

    Combines:
    1. Sentence embeddings for semantic similarity
    2. Keyword boosting for domain-specific terms
    3. Negative keyword penalties
    4. Confidence calibration

    Example:
        router = SemanticRouter()
        domain, confidence = router.route("Write a Python function")
        # domain=Domain.CODE, confidence=0.85
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        temperature: float = 0.10,
        keyword_weight: float = 0.15,
    ):
        self.model_name = model_name
        self.use_embeddings = use_embeddings
        # Softmax temperature over centred scores. Cosine gaps between domains
        # are O(0.05), so the temperature has to be of that order for the
        # posterior to be anything but uniform. Swept over {0.02 .. 0.40} on 450
        # labelled items: accuracy is flat from 0.02 to 0.10 (96.2-96.9%) and
        # falls off by 0.40, while calibration keeps improving as it rises.
        # 0.10 is the point where confidence is still informative - accuracy by
        # confidence bin runs 64% / 86% / 99% / 99% - rather than saturated at
        # 1.000 for nearly every item.
        self.temperature = temperature
        # Keyword evidence is a nudge, not a vote. It used to be 40% of a
        # normalized blend, which let it override the semantic ranking outright.
        self.keyword_weight = keyword_weight
        self._model = None
        self._domain_embeddings: Dict[Domain, torch.Tensor] = {}
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazy initialization of embedding model"""
        if self._initialized:
            return

        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                print("[Router] Loading embedding model...")
                self._model = SentenceTransformer(self.model_name)
                self._precompute_domain_embeddings()
                print("[Router] Semantic router ready")
            except ImportError:
                print("[Router] sentence-transformers not found, using keyword-only")
                self._model = None

        self._initialized = True

    def _precompute_domain_embeddings(self) -> None:
        """Pre-compute domain centroids from exemplar prompts"""
        if self._model is None:
            return

        for domain, profile in DOMAIN_PROFILES.items():
            if not profile.exemplar_prompts:
                continue

            embeddings = self._model.encode(
                profile.exemplar_prompts,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            # Domain centroid = mean of exemplars
            self._domain_embeddings[domain] = embeddings.mean(dim=0)

    def _embed(self, text: str) -> torch.Tensor:
        return self._model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

    def _semantic_score(self, text: str, domain: Domain) -> float:
        """
        Raw cosine similarity to the domain centroid, in [-1, 1].

        This used to return (cos + 1) / 2. The shift is what made the router
        uninformative: sentence-embedding cosines against these centroids live in
        roughly [0.0, 0.35], so adding a constant 0.5 to every domain and then
        normalizing across five of them compressed every confidence into a
        0.22-0.28 band. Differences survive centering (see route()); they do not
        survive a constant offset.
        """
        self._lazy_init()
        if self._model is None or domain not in self._domain_embeddings:
            return 0.0

        return torch.nn.functional.cosine_similarity(
            self._embed(text).unsqueeze(0),
            self._domain_embeddings[domain].unsqueeze(0),
        ).item()

    @staticmethod
    def _matches(keyword: str, text_lower: str) -> bool:
        """
        Whole-word match for alphanumeric keywords, substring for symbolic ones.

        Substring matching made "sin" fire on "using", "iv" on "give"/"five",
        "log" on "biology" and "tan" on "important". Symbolic keywords ("def ",
        "print(", "()", "^2") have no word boundaries to anchor to, so they stay
        substring matches.
        """
        kw = keyword.lower()
        if not kw[0].isalnum() or not kw[-1].isalnum():
            return kw in text_lower
        return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", text_lower) is not None

    def _keyword_score(self, text: str, profile: DomainProfile) -> float:
        """
        Bounded keyword evidence in [-1, 1].

        Saturating rather than linear: three domain terms is strong evidence,
        thirty is not ten times stronger, and an unbounded count let a long
        question dominate the semantic signal entirely.
        """
        text_lower = text.lower()

        positive = sum(1 for kw in profile.keywords if self._matches(kw, text_lower))
        negative = sum(1 for kw in profile.negative_keywords if self._matches(kw, text_lower))

        return math.tanh(positive / 3.0) - math.tanh(negative / 2.0)

    def route(
        self,
        prompt: str,
        top_k: int = 3,
    ) -> List[Tuple[Domain, float]]:
        """
        Route prompt to matching domains with confidence scores.

        Args:
            prompt: Input prompt
            top_k: Number of top domains to return

        Returns:
            List of (domain, confidence) tuples, sorted by confidence
        """
        self._lazy_init()

        # One embedding for the prompt, reused against every centroid.
        text_emb = self._embed(prompt) if self._model is not None else None

        raw: Dict[Domain, float] = {}
        for domain, profile in DOMAIN_PROFILES.items():
            if text_emb is not None and domain in self._domain_embeddings:
                semantic = torch.nn.functional.cosine_similarity(
                    text_emb.unsqueeze(0), self._domain_embeddings[domain].unsqueeze(0)
                ).item()
            else:
                semantic = 0.0
            raw[domain] = semantic + self.keyword_weight * self._keyword_score(prompt, profile)

        # Centre before softmax. Only differences between domains carry
        # information; the absolute cosine level is a property of the embedding
        # model, not of the prompt, and leaving it in flattens every posterior.
        mean = sum(raw.values()) / len(raw)
        logits = {d: (v - mean) / self.temperature for d, v in raw.items()}

        # profile.weight is a prior over domains, so it belongs in log space.
        for d, profile in DOMAIN_PROFILES.items():
            logits[d] += math.log(max(profile.weight, 1e-6))

        top = max(logits.values())
        exp = {d: math.exp(v - top) for d, v in logits.items()}
        total = sum(exp.values())
        posterior = sorted(((d, v / total) for d, v in exp.items()), key=lambda x: -x[1])

        return posterior[:top_k]

    def get_best_domain(
        self,
        prompt: str,
        confidence_threshold: float = 0.15,
    ) -> Tuple[Domain, float]:
        """
        Get single best domain for prompt.

        Falls back to GENERAL when the posterior is too flat to justify a
        specialized pipeline.

        There used to be a second keyword re-ranking pass here, guarded by
        `confidence < 0.30`. Because route() normalized post-shifted cosines
        across five domains, confidence could not reach 0.30 - the guard fired on
        199 of 200 GSM8K items, so the keyword score silently decided every
        route. With a calibrated posterior the guard is unnecessary: keyword
        evidence is already folded into the score, once, with a bounded weight.
        """
        results = self.route(prompt, top_k=len(DOMAIN_PROFILES))

        if not results:
            return Domain.GENERAL, 0.0

        domain, confidence = results[0]
        if confidence < confidence_threshold:
            return Domain.GENERAL, confidence

        return domain, confidence

    def explain(self, prompt: str) -> str:
        """Get human-readable routing explanation"""
        results = self.route(prompt, top_k=5)

        lines = ["Routing Analysis:"]
        for domain, score in results:
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            lines.append(f"  {domain.value:12} [{bar}] {score:.1%}")

        best, conf = results[0]
        lines.append(f"\n  → Selected: {best.value.upper()} ({conf:.1%})")

        return "\n".join(lines)


# Global router instance (lazy-loaded)
_router: Optional[SemanticRouter] = None


def get_router() -> SemanticRouter:
    """Get or create global router instance"""
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router


def auto_route(prompt: str) -> Tuple[Domain, float]:
    """Convenience function for routing"""
    return get_router().get_best_domain(prompt)
