#!/usr/bin/env python3
"""
Quick comparison of LatentMAS modes with ONE prompt.
Shows clear latency and token comparison.
"""

import sys
import time
sys.path.insert(0, '/workspace/latent_mas_slora')

from src.system import LatentMASSystem
from src.agents.configs import AgentConfig

# Single test prompt (medium difficulty)
TEST_PROMPT = "A farmer has chickens and rabbits. He counts 35 heads and 94 legs total. How many of each animal does he have?"

def run_comparison():
    print("=" * 70)
    print("⚡ LatentMAS MODE COMPARISON - Single Prompt Test")
    print("=" * 70)
    print(f"\n📝 Prompt: {TEST_PROMPT}\n")
    print("Loading model...")
    
    # Create ONE system and reuse it
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="bfloat16",
    )
    
    # Initialize with a dummy agent first
    system.add_agent(AgentConfig.planner(max_tokens=10))
    
    results = {}
    
    # ================================================================
    # MODE 1: TRUE LATENT (fastest)
    # ================================================================
    print("\n" + "-" * 70)
    print("⚡ MODE 1: TRUE_LATENT (4 agents, only final outputs text)")
    print("-" * 70)
    
    # Clear and add agents for true_latent
    system._pool._agents.clear()
    system._agent_count = 0
    system.add_agent(AgentConfig.planner(max_tokens=10))
    system.add_agent(AgentConfig.critic(max_tokens=10))
    system.add_agent(AgentConfig.refiner(max_tokens=10))
    system.add_agent(AgentConfig.judger(max_tokens=800))
    
    start = time.time()
    result1 = system.run(question=TEST_PROMPT, pipeline="true_latent")
    time1 = time.time() - start
    
    results["true_latent"] = {
        "time": time1,
        "tokens": result1.total_tokens,
        "answer": result1.final_answer[:300] if result1.final_answer else ""
    }
    print(f"✓ Time: {time1:.1f}s | Tokens: {result1.total_tokens}")
    
    # ================================================================
    # MODE 2: FAST (3 agents)
    # ================================================================
    print("\n" + "-" * 70)
    print("🚀 MODE 2: FAST (3 agents, text at each step)")
    print("-" * 70)
    
    # Clear and add agents for fast
    system._pool._agents.clear()
    system._agent_count = 0
    system.add_agent(AgentConfig.planner(max_tokens=150))
    system.add_agent(AgentConfig.refiner(max_tokens=400))
    system.add_agent(AgentConfig.judger(max_tokens=300))
    
    start = time.time()
    result2 = system.run(question=TEST_PROMPT, pipeline="hierarchical")
    time2 = time.time() - start
    
    results["fast"] = {
        "time": time2,
        "tokens": result2.total_tokens,
        "answer": result2.final_answer[:300] if result2.final_answer else ""
    }
    print(f"✓ Time: {time2:.1f}s | Tokens: {result2.total_tokens}")
    
    # ================================================================
    # MODE 3: NORMAL (4 agents, full text)
    # ================================================================
    print("\n" + "-" * 70)
    print("🐢 MODE 3: NORMAL (4 agents, full text generation)")
    print("-" * 70)
    
    # Clear and add agents for normal
    system._pool._agents.clear()
    system._agent_count = 0
    system.add_agent(AgentConfig.planner(max_tokens=300))
    system.add_agent(AgentConfig.critic(max_tokens=250))
    system.add_agent(AgentConfig.refiner(max_tokens=400))
    system.add_agent(AgentConfig.judger(max_tokens=400))
    
    start = time.time()
    result3 = system.run(question=TEST_PROMPT, pipeline="hierarchical")
    time3 = time.time() - start
    
    results["normal"] = {
        "time": time3,
        "tokens": result3.total_tokens,
        "answer": result3.final_answer[:300] if result3.final_answer else ""
    }
    print(f"✓ Time: {time3:.1f}s | Tokens: {result3.total_tokens}")
    
    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print("\n")
    print("=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    # Header
    print(f"\n{'Mode':<20} {'Agents':>6} {'Time':>10} {'Tokens':>10} {'Speedup':>12} {'Token Δ':>12}")
    print("-" * 75)
    
    base_time = results["normal"]["time"]
    base_tokens = results["normal"]["tokens"]
    
    agents_count = {"true_latent": 4, "fast": 3, "normal": 4}
    
    for mode in ["true_latent", "fast", "normal"]:
        r = results[mode]
        speedup = base_time / r["time"] if r["time"] > 0 else 0
        token_diff = ((base_tokens - r["tokens"]) / base_tokens * 100) if base_tokens > 0 else 0
        
        emoji = {"true_latent": "⚡", "fast": "🚀", "normal": "🐢"}[mode]
        speedup_str = f"{speedup:.2f}x" if mode != "normal" else "baseline"
        token_str = f"-{token_diff:.0f}%" if token_diff > 0 else "baseline"
        
        print(f"{emoji} {mode:<17} {agents_count[mode]:>6} {r['time']:>8.1f}s {r['tokens']:>10} {speedup_str:>12} {token_str:>12}")
    
    # Visual bar chart
    print("\n" + "=" * 70)
    print("📈 TIME COMPARISON (lower is better)")
    print("=" * 70)
    
    max_time = max(r["time"] for r in results.values())
    for mode in ["true_latent", "fast", "normal"]:
        r = results[mode]
        bar_len = int((r["time"] / max_time) * 40) if max_time > 0 else 0
        bar = "█" * bar_len
        emoji = {"true_latent": "⚡", "fast": "🚀", "normal": "🐢"}[mode]
        print(f"{emoji} {mode:<12} |{bar:<40}| {r['time']:.1f}s")
    
    print("\n" + "=" * 70)
    print("📉 TOKEN COMPARISON (lower is better)")
    print("=" * 70)
    
    max_tok = max(r["tokens"] for r in results.values())
    for mode in ["true_latent", "fast", "normal"]:
        r = results[mode]
        bar_len = int((r["tokens"] / max_tok) * 40) if max_tok > 0 else 0
        bar = "█" * bar_len
        emoji = {"true_latent": "⚡", "fast": "🚀", "normal": "🐢"}[mode]
        print(f"{emoji} {mode:<12} |{bar:<40}| {r['tokens']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 SUMMARY")
    print("=" * 70)
    speedup_tl = base_time / results["true_latent"]["time"]
    token_save = (1 - results["true_latent"]["tokens"] / base_tokens) * 100
    print(f"\n⚡ TRUE_LATENT vs NORMAL:")
    print(f"   → {speedup_tl:.1f}x FASTER")
    print(f"   → {token_save:.0f}% FEWER tokens")
    print(f"\n✅ Benchmark complete!")
    
    return results


if __name__ == "__main__":
    run_comparison()
