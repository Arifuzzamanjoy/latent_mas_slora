#!/usr/bin/env python3
"""
Test Hybrid Domain + Role LoRA System
"""

from src import LatentMASSystem
from src.agents.configs import AgentConfig
from src.routing import Domain

def test_hybrid_system():
    """Test domain routing with role-based agents"""
    
    print("="*70)
    print("Testing Hybrid Domain + Role LoRA System")
    print("="*70)
    
    # Initialize system
    print("\n1. Initializing system...")
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype="bfloat16",
        latent_steps=10,
    )
    
    # Add role-based agents
    print("2. Adding role-based agents...")
    system.add_agent(AgentConfig.planner(max_tokens=100))
    system.add_agent(AgentConfig.critic(max_tokens=100))
    system.add_agent(AgentConfig.refiner(max_tokens=100))
    system.add_agent(AgentConfig.judger(max_tokens=500))
    
    # Enable domain routing
    print("3. Enabling domain routing...")
    system.enable_domain_routing(
        embedding_model="all-MiniLM-L6-v2",
        auto_load_adapters=False,
    )
    
    # Test queries
    test_cases = [
        {
            "query": "What is the treatment for hypertension?",
            "expected_domain": Domain.MEDICAL,
        },
        {
            "query": "Solve the equation x^2 + 5x + 6 = 0",
            "expected_domain": Domain.MATH,
        },
        {
            "query": "Write a Python function to reverse a string",
            "expected_domain": Domain.CODE,
        },
        {
            "query": "What is the capital of France?",
            "expected_domain": Domain.GENERAL,
        },
    ]
    
    print("\n" + "="*70)
    print("Testing Domain Detection")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected = test["expected_domain"]
        
        print(f"\n[Test {i}]")
        print(f"Query: {query}")
        
        # Get routing
        routes = system._semantic_router.route(query, top_k=1)
        if routes:
            domain, confidence = routes[0]
            print(f"✓ Detected: {domain.value} (confidence: {confidence:.2f})")
            print(f"  Expected: {expected.value}")
            
            if domain == expected or (expected == Domain.GENERAL and confidence < 0.3):
                print(f"  ✅ PASS")
            else:
                print(f"  ⚠️  MISMATCH")
    
    print("\n" + "="*70)
    print("Domain Routing Test Complete!")
    print("="*70)

if __name__ == "__main__":
    test_hybrid_system()
