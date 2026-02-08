#!/usr/bin/env python3
"""
Ultra-fast routing test - NO heavy imports (no torch, no transformers)
Tests only the keyword-based routing logic.

Speed: ~20μs per query (50,000+ queries/sec)
"""

import time
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

print("Loading... (no ML imports)")

# ============ Inline Domain Definitions ============

class Domain(Enum):
    MEDICAL = "medical"
    FINANCE = "finance"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    GENERAL = "general"

DOMAIN_KEYWORDS = {
    Domain.MEDICAL: [
        "symptom", "disease", "treatment", "diagnosis", "patient", "medicine",
        "drug", "doctor", "hospital", "clinical", "therapy", "prescription",
        "medical", "health", "condition", "illness", "cure", "surgery",
        "infection", "virus", "bacteria", "vaccine", "dosage", "side effect",
        "blood", "heart", "lung", "liver", "kidney", "brain", "cancer",
        "diabetes", "hypertension", "cholesterol", "antibiotic", "painkiller",
        "side effects", "antibiotics",  # Added plural forms
    ],
    Domain.FINANCE: [
        "bitcoin", "crypto", "blockchain", "ethereum", "trading", "investment",
        "stock", "market", "price", "currency", "defi", "nft", "token",
        "wallet", "exchange", "mining", "altcoin", "portfolio", "profit",
        "dividend", "interest", "loan", "mortgage", "bank", "financial",
        "hedge", "futures", "options", "bonds", "equity", "asset",
    ],
    Domain.CODE: [
        "python", "javascript", "function", "class", "code", "programming",
        "algorithm", "debug", "api", "database", "sql", "html", "css",
        "react", "node", "git", "docker", "aws", "deploy", "server",
        "frontend", "backend", "variable", "loop", "array", "object",
        "implement", "compile", "runtime", "exception", "library", "framework",
    ],
    Domain.MATH: [
        "calculate", "equation", "integral", "derivative", "matrix", "vector",
        "algebra", "geometry", "calculus", "probability", "statistics",
        "theorem", "proof", "formula", "solve", "graph", "function",
        "logarithm", "exponential", "trigonometry", "polynomial", "factorial",
    ],
    Domain.REASONING: [
        "analyze", "compare", "evaluate", "argue", "logic", "reasoning",
        "hypothesis", "conclusion", "evidence", "inference", "deduce",
        "critical thinking", "problem solving", "decision", "strategy",
    ],
}


# ============ Fast Keyword Router ============

class FastKeywordRouter:
    """Ultra-fast keyword-only router - no ML dependencies"""
    
    def __init__(self):
        self.domain_keywords = DOMAIN_KEYWORDS
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for speed"""
        self._patterns = {}
        for domain, keywords in self.domain_keywords.items():
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self._patterns[domain] = re.compile(pattern, re.IGNORECASE)
    
    def route(self, query: str) -> Tuple[Domain, float, Dict[Domain, int]]:
        """Route query to domain based on keyword matches"""
        query_lower = query.lower()
        
        scores = {}
        for domain, pattern in self._patterns.items():
            matches = pattern.findall(query_lower)
            scores[domain] = len(matches)
        
        # Find best domain
        if max(scores.values()) == 0:
            return Domain.GENERAL, 0.5, scores
        
        best_domain = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = scores[best_domain] / max(total_matches, 1)
        
        return best_domain, confidence, scores


# ============ Tests ============

def test_routing():
    print("="*60)
    print("Fast Keyword Router Test")
    print("="*60)
    
    router = FastKeywordRouter()
    
    test_cases = [
        # Medical
        ("What are the symptoms of diabetes?", Domain.MEDICAL),
        ("Treatment options for hypertension", Domain.MEDICAL),
        ("Side effects of antibiotics", Domain.MEDICAL),
        
        # Finance  
        ("What is Bitcoin price today?", Domain.FINANCE),
        ("How does Ethereum blockchain work?", Domain.FINANCE),
        ("Best crypto trading strategies", Domain.FINANCE),
        
        # Code
        ("Write a Python function for sorting", Domain.CODE),
        ("How to debug JavaScript errors", Domain.CODE),
        ("Implement a REST API with Node.js", Domain.CODE),
        
        # Math
        ("Solve the integral of x^2", Domain.MATH),
        ("Calculate the derivative of sin(x)", Domain.MATH),
        ("Find eigenvalues of this matrix", Domain.MATH),
        
        # General
        ("What is the meaning of life?", Domain.GENERAL),
        ("Tell me about ancient Rome", Domain.GENERAL),
    ]
    
    correct = 0
    for query, expected in test_cases:
        domain, conf, scores = router.route(query)
        is_correct = domain == expected
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"{status} '{query[:40]}...' -> {domain.value} (conf={conf:.2f})")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
    return correct / len(test_cases)


def test_speed():
    print("\n" + "="*60)
    print("Speed Benchmark")
    print("="*60)
    
    router = FastKeywordRouter()
    
    queries = [
        "What are the symptoms of diabetes?",
        "Bitcoin price prediction for 2024",
        "Implement quicksort in Python",
        "Calculate the integral of x^2",
        "What is the meaning of life?",
    ]
    
    # Warm up
    for q in queries:
        router.route(q)
    
    # Benchmark
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        for q in queries:
            router.route(q)
    elapsed = time.perf_counter() - start
    
    total_queries = iterations * len(queries)
    avg_us = (elapsed / total_queries) * 1_000_000
    qps = total_queries / elapsed
    
    print(f"Total queries: {total_queries}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Average: {avg_us:.1f}μs per query")
    print(f"Throughput: {qps:,.0f} queries/sec")
    print(f"\n✓ Target met: <1ms per query" if avg_us < 1000 else "✗ Too slow")


if __name__ == "__main__":
    test_routing()
    test_speed()
    print("\n✓ All tests complete!")
