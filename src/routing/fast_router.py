"""
Fast Domain Router - Zero ML Dependencies

Ultra-fast keyword-based routing (~20μs per query, 50,000+ qps)
No torch, no transformers, no sentence-transformers.

Usage:
    from src.routing.fast_router import FastRouter
    router = FastRouter()
    domain, confidence = router.route("What is Bitcoin?")
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Domain(Enum):
    """Supported domains"""
    CODE = "code"
    MATH = "math"
    MEDICAL = "medical"
    FINANCE = "finance"
    REASONING = "reasoning"
    GENERAL = "general"


@dataclass
class FastRoutingResult:
    """Routing result"""
    domain: Domain
    confidence: float
    scores: Dict[str, float]
    method: str = "keyword_fast"


# Domain keywords (comprehensive)
DOMAIN_KEYWORDS = {
    Domain.MEDICAL: [
        "symptom", "symptoms", "disease", "treatment", "diagnosis", "patient",
        "medicine", "drug", "drugs", "doctor", "hospital", "clinical", "therapy",
        "prescription", "medical", "health", "condition", "illness", "cure",
        "surgery", "infection", "virus", "bacteria", "vaccine", "vaccines",
        "dosage", "side effect", "side effects", "blood", "heart", "lung",
        "liver", "kidney", "brain", "cancer", "tumor", "diabetes", "hypertension",
        "cholesterol", "antibiotic", "antibiotics", "painkiller", "painkillers",
        "anatomy", "physiology", "pathology", "pharmacology", "chronic", "acute",
        "cranial", "nerve", "artery", "vein", "seizure", "thyroid", "hemoglobin",
        "mutation", "protein", "enzyme", "deficiency", "syndrome", "prognosis",
        "presents with", "year-old", "history of", "drug of choice", "mechanism of action",
        # Dental/dentistry terms
        "dental", "dentist", "tooth", "teeth", "cavity", "incisor", "molar", "enamel",
        "gingival", "pulp", "caries", "preparation", "restoration", "crown", "filling",
    ],
    Domain.FINANCE: [
        "bitcoin", "btc", "crypto", "cryptocurrency", "blockchain", "ethereum",
        "eth", "trading", "investment", "stock", "stocks", "market", "price",
        "currency", "defi", "nft", "token", "tokens", "wallet", "exchange",
        "mining", "altcoin", "portfolio", "profit", "dividend", "interest",
        "loan", "mortgage", "bank", "financial", "hedge", "futures", "options",
        "bonds", "equity", "asset", "assets", "solana", "cardano", "dogecoin",
        "tron", "trx", "binance", "coinbase", "staking", "yield", "liquidity",
        "market cap", "bull", "bear", "hodl", "fomo",
    ],
    Domain.CODE: [
        "python", "javascript", "java", "function", "class", "code", "programming",
        "algorithm", "debug", "api", "database", "sql", "html", "css", "react",
        "node", "git", "docker", "aws", "deploy", "server", "frontend", "backend",
        "variable", "loop", "array", "object", "implement", "compile", "runtime",
        "exception", "library", "framework", "typescript", "rust", "golang",
        "def ", "return ", "import ", "const ", "let ", "var ", "async", "await",
        "try:", "except:", "catch", "throw", "error", "bug", "fix",
    ],
    Domain.MATH: [
        "calculate", "equation", "integral", "derivative", "matrix", "vector",
        "algebra", "geometry", "calculus", "probability", "statistics", "theorem",
        "proof", "formula", "solve", "graph", "logarithm", "exponential",
        "trigonometry", "polynomial", "factorial", "eigenvalue", "eigenvector",
        "limit", "sum", "product", "series", "permutation", "combination",
        "sin", "cos", "tan", "sqrt", "log", "ln", "x^", "x²", "x³",
    ],
    Domain.REASONING: [
        "analyze", "compare", "evaluate", "argue", "logic", "reasoning",
        "hypothesis", "conclusion", "evidence", "inference", "deduce",
        "critical thinking", "problem solving", "decision", "strategy",
        "pros and cons", "advantages", "disadvantages", "tradeoff",
    ],
}


class FastRouter:
    """
    Ultra-fast keyword router - NO ML dependencies.
    
    Speed: ~20μs per query (50,000+ queries/sec)
    """
    
    def __init__(self):
        self._patterns: Dict[Domain, re.Pattern] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for speed"""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            # Escape special chars and join with |
            escaped = [re.escape(kw) for kw in keywords]
            pattern = r'\b(' + '|'.join(escaped) + r')'
            self._patterns[domain] = re.compile(pattern, re.IGNORECASE)
    
    def route(self, query: str) -> Tuple[Domain, float]:
        """
        Route query to best domain.
        
        Returns:
            (domain, confidence) tuple
        """
        result = self.route_detailed(query)
        return result.domain, result.confidence
    
    def route_detailed(self, query: str) -> FastRoutingResult:
        """
        Route with detailed scores.
        
        Returns:
            FastRoutingResult with all domain scores
        """
        scores = {}
        
        for domain, pattern in self._patterns.items():
            matches = pattern.findall(query)
            scores[domain] = len(matches)
        
        # Find best
        total = sum(scores.values())
        if total == 0:
            return FastRoutingResult(
                domain=Domain.GENERAL,
                confidence=0.5,
                scores={d.value: 0.0 for d in Domain},
            )
        
        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain] / total
        
        return FastRoutingResult(
            domain=best_domain,
            confidence=confidence,
            scores={d.value: s / max(total, 1) for d, s in scores.items()},
        )
    
    def explain(self, query: str) -> str:
        """Get routing explanation"""
        result = self.route_detailed(query)
        
        lines = [
            "=" * 50,
            "FAST ROUTER ANALYSIS",
            "=" * 50,
            f"Query: {query[:60]}...",
            f"Result: {result.domain.value.upper()} ({result.confidence:.0%})",
            "",
            "Scores:",
        ]
        
        for domain, score in sorted(result.scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            lines.append(f"  {domain:12} {bar} {score:.0%}")
        
        return "\n".join(lines)


# Convenience function
def fast_route(query: str) -> Tuple[str, float]:
    """Quick routing function"""
    router = FastRouter()
    domain, conf = router.route(query)
    return domain.value, conf


if __name__ == "__main__":
    import time
    
    router = FastRouter()
    
    tests = [
        ("What are the symptoms of diabetes?", "medical"),
        ("Bitcoin price prediction", "finance"),
        ("Write a Python function", "code"),
        ("Solve x^2 + 2x = 0", "math"),
        ("What is the meaning of life?", "general"),
    ]
    
    print("Fast Router Test:")
    print("=" * 50)
    for query, expected in tests:
        domain, conf = router.route(query)
        status = "✓" if domain.value == expected else "✗"
        print(f"{status} {query[:35]:40} -> {domain.value:10} ({conf:.0%})")
    
    # Speed test
    print("\nSpeed Benchmark:")
    iterations = 5000
    start = time.perf_counter()
    for _ in range(iterations):
        for q, _ in tests:
            router.route(q)
    elapsed = time.perf_counter() - start
    total = iterations * len(tests)
    
    print(f"Total: {total} queries in {elapsed:.3f}s")
    print(f"Speed: {(elapsed/total)*1_000_000:.1f}μs/query ({total/elapsed:,.0f} qps)")
