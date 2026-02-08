"""
Semantic Router - State-of-the-Art Prompt-Based Routing

Uses embedding similarity + keyword boosting for intelligent
domain/pipeline selection.
"""

import torch
from typing import Dict, List, Tuple, Optional
from .domain_profiles import Domain, DomainProfile, DOMAIN_PROFILES


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
        # OPTIMIZATION: Cache query embeddings to avoid redundant encoding
        self._query_embedding_cache: Dict[str, torch.Tensor] = {}
        self._cache_max_size = 100
    
    def _lazy_init(self) -> None:
        """Lazy initialization of embedding model"""
        if self._initialized:
            return
        
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                import warnings
                import logging
                
                # Suppress harmless warnings about position_ids
                logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
                warnings.filterwarnings("ignore", message=".*position_ids.*")
                
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
    
    def _get_query_embedding(self, text: str) -> torch.Tensor:
        """Get or compute query embedding with caching"""
        # Check cache first
        if text in self._query_embedding_cache:
            return self._query_embedding_cache[text]
        
        # Compute embedding
        text_emb = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        # Cache it (with size limit)
        if len(self._query_embedding_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._query_embedding_cache))
            del self._query_embedding_cache[oldest_key]
        
        self._query_embedding_cache[text] = text_emb
        return text_emb
    
    def _semantic_score(self, text: str, domain: Domain) -> float:
        """Compute semantic similarity to domain centroid (with cached embedding)"""
        if self._model is None or domain not in self._domain_embeddings:
            return 0.0
        
        # OPTIMIZATION: Use cached query embedding
        text_emb = self._get_query_embedding(text)
        
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
        confidence_threshold: float = 0.20,
    ) -> Tuple[Domain, float]:
        """
        Get single best domain for prompt.
        
        Returns GENERAL if confidence is below threshold.
        """
        results = self.route(prompt, top_k=3)
        
        if not results:
            return Domain.GENERAL, 0.0
        
        domain, confidence = results[0]
        
        # Simply use the highest-scoring domain; no bias toward any specific domain
        # This ensures context-sensitive routing based purely on semantic similarity and keywords
        
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
