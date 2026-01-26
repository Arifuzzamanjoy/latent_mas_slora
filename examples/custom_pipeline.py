#!/usr/bin/env python3
"""
Custom Pipeline Example

Demonstrates:
1. Creating custom agents with specific prompts
2. Building domain-specific pipelines
3. Loading multiple external LoRAs
"""

import sys
sys.path.insert(0, '/workspace/latent_mas_slora')

from src import LatentMASSystem, AgentConfig, AgentRole
from src.agents.configs import LoRASpec
from src.lora.adapter_manager import QWEN25_LORA_REGISTRY


def create_coding_pipeline():
    """Create a coding-focused pipeline"""
    print("\n" + "=" * 60)
    print("Coding Pipeline Example")
    print("=" * 60)
    
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="bfloat16",
        latent_steps=15,
    )
    
    # Code analyzer
    system.add_agent(AgentConfig(
        name="CodeAnalyzer",
        role=AgentRole.PLANNER,
        adapter_name="code_analyzer_lora",
        lora_spec=LoRASpec(rank=48, alpha=96),
        temperature=0.5,
        max_tokens=400,
        system_prompt=(
            "You are a Code Analyzer. Break down coding problems into: "
            "1) Input/output requirements, 2) Edge cases, 3) Algorithm approach. "
            "Identify the core algorithmic pattern needed."
        ),
    ))
    
    # Code generator
    system.add_agent(AgentConfig.coder())
    
    # Code reviewer
    system.add_agent(AgentConfig(
        name="CodeReviewer",
        role=AgentRole.CRITIC,
        adapter_name="code_reviewer_lora",
        lora_spec=LoRASpec(rank=48, alpha=96),
        temperature=0.3,
        max_tokens=350,
        system_prompt=(
            "You are a Code Reviewer. Check for: "
            "1) Correctness, 2) Edge case handling, 3) Time/space complexity, "
            "4) Code quality and best practices. Suggest improvements."
        ),
    ))
    
    # Final judger
    system.add_agent(AgentConfig.judger())
    
    # Example coding problem
    question = """Write a Python function to find the longest palindromic substring in a given string.

Example:
Input: "babad"
Output: "bab" (or "aba")

Input: "cbbd"
Output: "bb"

Provide the solution with explanation."""

    print(f"\nQuestion: {question[:150]}...")
    
    result = system.run(
        question=question,
        pipeline="hierarchical",
        max_new_tokens=500,
        temperature=0.4,
    )
    
    print("\n" + "-" * 40)
    print("FINAL OUTPUT:")
    print("-" * 40)
    print(result.final_answer)
    
    return result


def create_math_pipeline():
    """Create a math-focused pipeline"""
    print("\n" + "=" * 60)
    print("Math Pipeline Example")
    print("=" * 60)
    
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="bfloat16",
        latent_steps=15,
    )
    
    # Math analyzer
    system.add_agent(AgentConfig(
        name="MathAnalyzer",
        role=AgentRole.PLANNER,
        adapter_name="math_analyzer_lora",
        lora_spec=LoRASpec(rank=48, alpha=96),
        temperature=0.4,
        max_tokens=400,
        system_prompt=(
            "You are a Math Problem Analyzer. For each problem: "
            "1) Identify what is given and what to find, "
            "2) List relevant formulas/theorems, "
            "3) Plan the solution approach step by step."
        ),
    ))
    
    # Math solver
    system.add_agent(AgentConfig.math())
    
    # Math verifier
    system.add_agent(AgentConfig(
        name="MathVerifier",
        role=AgentRole.CRITIC,
        adapter_name="math_verifier_lora",
        lora_spec=LoRASpec(rank=32, alpha=64),
        temperature=0.2,
        max_tokens=300,
        system_prompt=(
            "You are a Math Verifier. Check: "
            "1) Are all calculations correct? "
            "2) Is the logic sound? "
            "3) Are there alternative approaches? "
            "Verify the final answer."
        ),
    ))
    
    # Final judger
    system.add_agent(AgentConfig.judger())
    
    # Example math problem
    question = """Find all real values of x that satisfy:
    
    log₂(x² - 4x + 3) = 1
    
    Show your work step by step."""

    print(f"\nQuestion: {question}")
    
    result = system.run(
        question=question,
        pipeline="hierarchical",
        max_new_tokens=500,
        temperature=0.3,
    )
    
    print("\n" + "-" * 40)
    print("FINAL OUTPUT:")
    print("-" * 40)
    print(result.final_answer)
    
    return result


def show_available_loras():
    """Show available LoRAs in registry"""
    print("\n" + "=" * 60)
    print("Available External LoRAs")
    print("=" * 60)
    
    print("\nQwen2.5 LoRA Registry:")
    for name, info in QWEN25_LORA_REGISTRY.items():
        print(f"\n  {name}:")
        print(f"    HF Path: {info.hf_path}")
        print(f"    Domain: {info.domain}")
        print(f"    Base: {info.base_model}")
        print(f"    Description: {info.description}")


def main():
    show_available_loras()
    
    print("\n" + "=" * 60)
    print("Running Examples")
    print("=" * 60)
    
    # Run coding example
    coding_result = create_coding_pipeline()
    
    print(f"\n[Coding Stats]")
    print(f"  Tokens: {coding_result.total_tokens}")
    print(f"  Latency: {coding_result.total_latency_ms}ms")
    
    # Run math example
    math_result = create_math_pipeline()
    
    print(f"\n[Math Stats]")
    print(f"  Tokens: {math_result.total_tokens}")
    print(f"  Latency: {math_result.total_latency_ms}ms")


if __name__ == "__main__":
    main()
