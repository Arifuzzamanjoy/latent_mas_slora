#!/usr/bin/env python3
"""
Test script for Advanced Hybrid Router

Validates the SOTA routing implementation combining:
- CASTER-style dual-signal routing
- RouteLLM-inspired trainable MLP
- Task meta-feature extraction
- Confidence calibration

Usage:
    python test_advanced_routing.py          # Fast mode (keyword-only)
    python test_advanced_routing.py --full   # Full mode with embeddings
"""

import time
import sys
import os
import argparse

# Parse args early
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Use full embedding mode (slower)")
args = parser.parse_args()
FAST_MODE = not args.full

# Direct imports to avoid loading heavy dependencies
from src.routing.advanced_router import (
    AdvancedHybridRouter,
    TaskMetaFeatureExtractor,
    RoutingResult,
)
from src.routing.semantic_router import SemanticRouter


def test_meta_feature_extraction():
    """Test TaskMetaFeatureExtractor"""
    print("\n" + "="*60)
    print("Testing TaskMetaFeatureExtractor")
    print("="*60)
    
    extractor = TaskMetaFeatureExtractor()
    
    test_queries = [
        "What is diabetes?",
        "Explain the pathophysiology of congestive heart failure with detailed mechanisms of cardiac remodeling",
        "What is Bitcoin price today?",
        "Calculate 2 + 2",
        "def hello_world():",
        "Why is the sky blue? What causes this phenomenon?",
    ]
    
    for query in test_queries:
        features = extractor.extract(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"  Length: {features['length']:.2f}")
        print(f"  Word count: {features['word_count']:.2f}")
        print(f"  Question type: {[k for k, v in features.items() if k.startswith('qtype_') and v > 0]}")
        print(f"  Complexity: {features['estimated_difficulty']:.2f}")


def test_advanced_router_routing():
    """Test AdvancedHybridRouter routing decisions"""
    print("\n" + "="*60)
    print("Testing AdvancedHybridRouter")
    print("="*60)
    
    # Initialize router (skip embeddings for fast testing)
    router = AdvancedHybridRouter(skip_embeddings=FAST_MODE)
    print(f"\nRouter initialized with domains: {list(router.domain_profiles.keys())}")
    print(f"Mode: {'FAST (keyword-only)' if FAST_MODE else 'FULL (with embeddings)'}")
    
    test_cases = [
        # Medical queries
        ("What are the symptoms of diabetes mellitus?", "MEDICAL"),
        ("How does metformin work for blood glucose control?", "MEDICAL"),
        ("Explain the mechanism of action of ACE inhibitors", "MEDICAL"),
        
        # Finance queries
        ("What is the current Bitcoin price prediction?", "FINANCE"),
        ("How does blockchain consensus work in Ethereum?", "FINANCE"),
        ("Explain DeFi yield farming strategies", "FINANCE"),
        
        # Coding queries
        ("How do I implement a binary search tree in Python?", "CODING"),
        ("What is the difference between REST and GraphQL APIs?", "CODING"),
        ("Explain async/await in JavaScript", "CODING"),
        
        # Math queries
        ("Solve the integral of x^2 from 0 to 1", "MATH"),
        ("What is the eigenvalue of a 3x3 matrix?", "MATH"),
        ("Prove the Pythagorean theorem", "MATH"),
        
        # General queries
        ("What is the meaning of life?", "GENERAL"),
        ("Tell me about the history of Rome", "GENERAL"),
        ("How does photosynthesis work?", "GENERAL"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_domain in test_cases:
        result = router.route(query)
        is_correct = result.domain == expected_domain
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Query: {query[:50]}...")
        print(f"    Expected: {expected_domain}, Got: {result.domain}")
        print(f"    Confidence: {result.confidence:.3f}")
        print(f"    Method: {result.method}")
        if result.meta_features:
            print(f"    Complexity: {result.meta_features.get('estimated_difficulty', 0):.2f}")
    
    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("="*60)
    
    return accuracy


def test_routing_speed():
    """Benchmark routing speed"""
    print("\n" + "="*60)
    print("Testing Routing Speed")
    print("="*60)
    
    router = AdvancedHybridRouter(skip_embeddings=FAST_MODE)
    
    test_queries = [
        "What is the treatment for hypertension?",
        "How does Bitcoin mining work?",
        "Implement quicksort in Python",
        "Solve x^2 + 2x + 1 = 0",
        "Tell me about machine learning",
    ]
    
    # Warm-up
    for _ in range(3):
        for query in test_queries:
            router.route(query)
    
    # Benchmark
    num_iterations = 10
    total_time = 0
    
    for _ in range(num_iterations):
        for query in test_queries:
            start = time.perf_counter()
            router.route(query)
            elapsed = time.perf_counter() - start
            total_time += elapsed
    
    avg_time_ms = (total_time / (num_iterations * len(test_queries))) * 1000
    
    print(f"\nTotal iterations: {num_iterations * len(test_queries)}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average routing time: {avg_time_ms:.2f}ms per query")
    
    if avg_time_ms < 100:
        print("✓ Performance target met (<100ms)")
    else:
        print("✗ Performance target not met (>100ms)")
    
    return avg_time_ms


def test_explain_functionality():
    """Test routing explanation"""
    print("\n" + "="*60)
    print("Testing Routing Explanation")
    print("="*60)
    
    router = AdvancedHybridRouter(skip_embeddings=FAST_MODE)
    
    test_queries = [
        "What are the side effects of ibuprofen?",
        "Explain smart contracts in blockchain",
        "How to debug a Python memory leak?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        explanation = router.explain(query)
        print(explanation)


def test_comparison_with_semantic_router():
    """Compare Advanced vs Semantic Router"""
    print("\n" + "="*60)
    print("Comparing Advanced vs Semantic Router")
    print("="*60)
    
    advanced_router = AdvancedHybridRouter(skip_embeddings=FAST_MODE)
    semantic_router = SemanticRouter(use_embeddings=not FAST_MODE)  # Also use fast mode
    
    test_queries = [
        "What is the mechanism of insulin resistance?",
        "How does Ethereum 2.0 proof of stake work?",
        "Implement a hash table in Python",
        "Find the derivative of sin(x^2)",
        "What is the capital of France?",
    ]
    
    print("\nComparison results:")
    print("-" * 80)
    print(f"{'Query':<40} {'Advanced':<15} {'Semantic':<15} {'Match'}")
    print("-" * 80)
    
    for query in test_queries:
        advanced_result = advanced_router.route(query)
        semantic_result = semantic_router.route(query)
        
        match = "✓" if advanced_result.domain == semantic_result[0] else "✗"
        print(f"{query[:38]:<40} {advanced_result.domain:<15} {semantic_result[0]:<15} {match}")
    
    print("-" * 80)


def main():
    """Run all tests"""
    print("="*60)
    print("Advanced Hybrid Router Test Suite")
    print("="*60)
    print("\nThis tests the SOTA routing implementation combining:")
    print("- CASTER-style dual-signal routing")
    print("- RouteLLM-inspired trainable MLP")
    print("- Task meta-feature extraction")
    print("- Confidence calibration")
    
    try:
        test_meta_feature_extraction()
        accuracy = test_advanced_router_routing()
        avg_time = test_routing_speed()
        test_explain_functionality()
        test_comparison_with_semantic_router()
        
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Routing Accuracy: {accuracy:.1f}%")
        print(f"Average Routing Time: {avg_time:.2f}ms")
        print("All tests completed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
