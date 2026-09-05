"""
Semantic Router - State-of-the-Art Prompt-Based Routing

Uses embedding similarity + keyword boosting for intelligent
domain/pipeline selection.
"""

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
    ):
        self.model_name = model_name
        self.use_embeddings = use_embeddings
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

    def _semantic_score(self, text: str, domain: Domain) -> float:
        """Compute semantic similarity to domain centroid"""
        if self._model is None or domain not in self._domain_embeddings:
            return 0.0

        text_emb = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarity = torch.nn.functional.cosine_similarity(
            text_emb.unsqueeze(0),
            self._domain_embeddings[domain].unsqueeze(0),
        ).item()

        # Scale from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _keyword_score(self, text: str, profile: DomainProfile) -> float:
        """Compute keyword-based score"""
        text_lower = text.lower()

        positive = sum(1 for kw in profile.keywords if kw.lower() in text_lower)
        negative = sum(1 for kw in profile.negative_keywords if kw.lower() in text_lower)

        score = (positive * 0.1) - (negative * 0.15)
        return max(0, score)

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

        scores = {}
        for domain, profile in DOMAIN_PROFILES.items():
            semantic = self._semantic_score(prompt, domain)
            keyword = self._keyword_score(prompt, profile)

            # Weighted combination: 60% semantic, 40% keyword
            combined = (0.6 * semantic + 0.4 * keyword) * profile.weight
            scores[domain] = combined

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Normalize to probabilities
        total = sum(s for _, s in sorted_scores) + 1e-8
        normalized = [(d, s / total) for d, s in sorted_scores]

        return normalized[:top_k]

    def get_best_domain(
        self,
        prompt: str,
        confidence_threshold: float = 0.15,
    ) -> Tuple[Domain, float]:
        """
        Get single best domain for prompt.

        Uses keyword re-ranking as tiebreaker when confidence is low.
        Returns GENERAL only if confidence is very low.
        """
        results = self.route(prompt, top_k=5)

        if not results:
            return Domain.GENERAL, 0.0

        domain, confidence = results[0]

        # If top confidence is low, use keyword scoring as tiebreaker
        if confidence < 0.30:
            keyword_scores = {}
            for d, profile in DOMAIN_PROFILES.items():
                kw_score = self._keyword_score(prompt, profile)
                keyword_scores[d] = kw_score

            # Find domain with highest keyword score
            best_kw_domain = max(keyword_scores, key=keyword_scores.get)
            best_kw_score = keyword_scores[best_kw_domain]

            # If keyword scoring strongly favors a different domain, use it
            if best_kw_score > 0.1 and best_kw_domain != domain:
                # Check that it's also in the top-3 semantic results
                top3_domains = [d for d, _ in results[:3]]
                if best_kw_domain in top3_domains:
                    domain = best_kw_domain
                    # Recalculate confidence from the route results
                    for d, c in results:
                        if d == domain:
                            confidence = c
                            break

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
