#!/usr/bin/env python3
"""
Benchmark comparison of LatentMAS modes:
  - true_latent: 4 agents, ONLY final outputs text (~20-30s) ⚡
  - fast:        3 agents, text at each step (~40-50s)
  - normal:      4 agents, full text generation (~90-120s)
"""

import sys
import time
import json
from datetime import datetime
sys.path.insert(0, '/workspace/latent_mas_slora')

from src.system import LatentMASSystem
from src.agents.configs import AgentConfig

# ============================================================================
# MEDIUM DIFFICULTY BENCHMARK QUESTIONS (3 domains)
# ============================================================================

BENCHMARK_QUESTIONS = [
    {
        "id": 1,
        "domain": "MATH",
        "difficulty": "medium",
        "question": "Find all values of x that satisfy: 2x³ - 5x² - 3x = 0",
        "expected_hint": "x = 0, x = 3, x = -0.5"
    },
    {
        "id": 2,
        "domain": "CODE",
        "difficulty": "medium",
        "question": "Write a Python function to find the longest palindromic substring in a given string. Include time complexity analysis.",
        "expected_hint": "O(n²) or O(n) with Manacher's"
    },
    {
        "id": 3,
        "domain": "REASONING",
        "difficulty": "medium",
        "question": "A farmer has chickens and rabbits. He counts 35 heads and 94 legs total. How many chickens and how many rabbits does he have? Show your reasoning step by step.",
        "expected_hint": "23 chickens, 12 rabbits"
    },
]


def create_system_for_mode(mode: str) -> LatentMASSystem:
    """Create and configure a system for a specific mode."""
    
    latent_steps = {
        "true_latent": 10,
        "fast": 5,
        "normal": 15,
    }.get(mode, 10)
    
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="bfloat16",
        latent_steps=latent_steps,
    )
    
    # Add agents based on mode
    if mode == "true_latent":
        # 4 agents, only final generates text
        system.add_agent(AgentConfig.planner(max_tokens=10))
        system.add_agent(AgentConfig.critic(max_tokens=10))
        system.add_agent(AgentConfig.refiner(max_tokens=10))
        system.add_agent(AgentConfig.judger(max_tokens=800))
        
    elif mode == "fast":
        # 3 agents, moderate text
        system.add_agent(AgentConfig.planner(max_tokens=150))
        system.add_agent(AgentConfig.refiner(max_tokens=400))
        system.add_agent(AgentConfig.judger(max_tokens=300))
        
    else:  # normal
        # 4 agents, full text
        system.add_agent(AgentConfig.planner(max_tokens=300))
        system.add_agent(AgentConfig.critic(max_tokens=250))
        system.add_agent(AgentConfig.refiner(max_tokens=400))
        system.add_agent(AgentConfig.judger(max_tokens=400))
    
    return system


def run_benchmark():
    """Run all questions through all modes and compare."""
    
    print("=" * 80)
    print("🔬 LatentMAS MODE COMPARISON BENCHMARK")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")
    print("Modes: true_latent (⚡), fast (🚀), normal (🐢)")
    print("=" * 80)
    
    # Results storage
    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "questions": len(BENCHMARK_QUESTIONS)
        },
        "questions": []
    }
    
    # Aggregate stats
    mode_totals = {
        "true_latent": {"time": 0, "tokens": 0, "count": 0},
        "fast": {"time": 0, "tokens": 0, "count": 0},
        "normal": {"time": 0, "tokens": 0, "count": 0}
    }
    
    # Create systems for each mode (reuse model weights)
    print("\n⏳ Loading systems for each mode...")
    load_start = time.time()
    
    systems = {}
    for mode in ["true_latent", "fast", "normal"]:
        print(f"   Creating {mode} system...")
        systems[mode] = create_system_for_mode(mode)
    
    load_time = time.time() - load_start
    print(f"✅ All systems ready in {load_time:.1f}s\n")
    
    # Run each question through all modes
    for q_idx, question in enumerate(BENCHMARK_QUESTIONS):
        print("\n" + "=" * 80)
        print(f"📝 Question {q_idx + 1}/{len(BENCHMARK_QUESTIONS)}: [{question['domain']}]")
        print("=" * 80)
        print(f"Q: {question['question'][:100]}...")
        print(f"Expected: {question['expected_hint']}")
        print("-" * 80)
        
        q_result = {
            "id": question["id"],
            "domain": question["domain"],
            "question": question["question"],
            "expected": question["expected_hint"],
            "modes": {}
        }
        
        # ================================================================
        # MODE 1: TRUE LATENT (fastest)
        # ================================================================
        print(f"\n⚡ [1/3] TRUE_LATENT mode (4 agents, only final outputs text)...")
        try:
            start = time.time()
            result_tl = systems["true_latent"].run(
                question=question["question"],
                pipeline="true_latent",  # Uses run_true_latent internally
            )
            elapsed_tl = time.time() - start
            
            answer_tl = (result_tl.final_answer or "")[:500]
            tokens_tl = result_tl.total_tokens or 0
            
            print(f"   ✓ Time: {elapsed_tl:.1f}s | Tokens: {tokens_tl}")
            print(f"   Answer preview: {answer_tl[:150]}...")
            
            q_result["modes"]["true_latent"] = {
                "time_s": round(elapsed_tl, 2),
                "tokens": tokens_tl,
                "answer": answer_tl
            }
            mode_totals["true_latent"]["time"] += elapsed_tl
            mode_totals["true_latent"]["tokens"] += tokens_tl
            mode_totals["true_latent"]["count"] += 1
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            q_result["modes"]["true_latent"] = {"error": str(e)}
        
        # ================================================================
        # MODE 2: FAST (3 agents)
        # ================================================================
        print(f"\n🚀 [2/3] FAST mode (3 agents, text at each step)...")
        try:
            start = time.time()
            result_fast = systems["fast"].run(
                question=question["question"],
                pipeline="hierarchical",
            )
            elapsed_fast = time.time() - start
            
            answer_fast = (result_fast.final_answer or "")[:500]
            tokens_fast = result_fast.total_tokens or 0
            
            print(f"   ✓ Time: {elapsed_fast:.1f}s | Tokens: {tokens_fast}")
            print(f"   Answer preview: {answer_fast[:150]}...")
            
            q_result["modes"]["fast"] = {
                "time_s": round(elapsed_fast, 2),
                "tokens": tokens_fast,
                "answer": answer_fast
            }
            mode_totals["fast"]["time"] += elapsed_fast
            mode_totals["fast"]["tokens"] += tokens_fast
            mode_totals["fast"]["count"] += 1
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            q_result["modes"]["fast"] = {"error": str(e)}
        
        # ================================================================
        # MODE 3: NORMAL (full generation)
        # ================================================================
        print(f"\n🐢 [3/3] NORMAL mode (4 agents, full text generation)...")
        try:
            start = time.time()
            result_norm = systems["normal"].run(
                question=question["question"],
                pipeline="hierarchical",
            )
            elapsed_norm = time.time() - start
            
            answer_norm = (result_norm.final_answer or "")[:500]
            tokens_norm = result_norm.total_tokens or 0
            
            print(f"   ✓ Time: {elapsed_norm:.1f}s | Tokens: {tokens_norm}")
            print(f"   Answer preview: {answer_norm[:150]}...")
            
            q_result["modes"]["normal"] = {
                "time_s": round(elapsed_norm, 2),
                "tokens": tokens_norm,
                "answer": answer_norm
            }
            mode_totals["normal"]["time"] += elapsed_norm
            mode_totals["normal"]["tokens"] += tokens_norm
            mode_totals["normal"]["count"] += 1
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            q_result["modes"]["normal"] = {"error": str(e)}
        
        # Question summary
        print(f"\n   📊 Question {q_idx + 1} Summary:")
        if "time_s" in q_result["modes"].get("true_latent", {}):
            tl_t = q_result["modes"]["true_latent"]["time_s"]
            norm_t = q_result["modes"].get("normal", {}).get("time_s", 0)
            
            if norm_t > 0 and tl_t > 0:
                speedup = norm_t / tl_t
                print(f"   Speedup: true_latent is {speedup:.1f}x faster than normal")
        
        results["questions"].append(q_result)
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n")
    print("=" * 80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    n_questions = len(BENCHMARK_QUESTIONS)
    
    print(f"\n{'Mode':<15} {'Total Time':<15} {'Avg Time':<12} {'Total Tokens':<15} {'Avg Tokens':<12}")
    print("-" * 70)
    
    for mode_name in ["true_latent", "fast", "normal"]:
        stats = mode_totals[mode_name]
        count = stats["count"] or 1
        total_t = stats["time"]
        avg_t = total_t / count
        total_tok = stats["tokens"]
        avg_tok = total_tok / count
        
        emoji = {"true_latent": "⚡", "fast": "🚀", "normal": "🐢"}[mode_name]
        print(f"{emoji} {mode_name:<12} {total_t:>10.1f}s    {avg_t:>8.1f}s    {total_tok:>10}      {avg_tok:>8.0f}")
    
    # Speedup calculations
    print("\n" + "-" * 70)
    if mode_totals["normal"]["time"] > 0 and mode_totals["true_latent"]["time"] > 0:
        speedup_tl = mode_totals["normal"]["time"] / mode_totals["true_latent"]["time"]
        speedup_fast = mode_totals["normal"]["time"] / mode_totals["fast"]["time"] if mode_totals["fast"]["time"] > 0 else 0
        
        token_reduction_tl = (1 - mode_totals["true_latent"]["tokens"] / mode_totals["normal"]["tokens"]) * 100 if mode_totals["normal"]["tokens"] > 0 else 0
        token_reduction_fast = (1 - mode_totals["fast"]["tokens"] / mode_totals["normal"]["tokens"]) * 100 if mode_totals["normal"]["tokens"] > 0 else 0
        
        print(f"\n🏆 SPEEDUP vs NORMAL:")
        print(f"   ⚡ true_latent: {speedup_tl:.2f}x faster")
        print(f"   🚀 fast:        {speedup_fast:.2f}x faster")
        
        print(f"\n📉 TOKEN REDUCTION vs NORMAL:")
        print(f"   ⚡ true_latent: {token_reduction_tl:.1f}% fewer tokens")
        print(f"   🚀 fast:        {token_reduction_fast:.1f}% fewer tokens")
    
    # Save results
    results["summary"] = {
        "mode_totals": {
            k: {"time_s": round(v["time"], 2), "tokens": v["tokens"], "count": v["count"]} 
            for k, v in mode_totals.items()
        }
    }
    
    if mode_totals["normal"]["time"] > 0 and mode_totals["true_latent"]["time"] > 0:
        results["summary"]["speedup"] = {
            "true_latent_vs_normal": round(mode_totals["normal"]["time"] / mode_totals["true_latent"]["time"], 2),
            "fast_vs_normal": round(mode_totals["normal"]["time"] / mode_totals["fast"]["time"], 2) if mode_totals["fast"]["time"] > 0 else 0,
        }
    
    output_file = "/workspace/latent_mas_slora/benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Visual comparison
    print("\n" + "=" * 80)
    print("📈 VISUAL COMPARISON (avg time in seconds)")
    print("=" * 80)
    
    max_time = max(mode_totals[m]["time"] / (mode_totals[m]["count"] or 1) for m in mode_totals)
    
    for mode_name in ["true_latent", "fast", "normal"]:
        count = mode_totals[mode_name]["count"] or 1
        avg_t = mode_totals[mode_name]["time"] / count
        bar_len = int((avg_t / max_time) * 40) if max_time > 0 else 0
        bar = "█" * bar_len
        emoji = {"true_latent": "⚡", "fast": "🚀", "normal": "🐢"}[mode_name]
        print(f"{emoji} {mode_name:<12} |{bar:<40}| {avg_t:.1f}s")
    
    print("\n✅ Benchmark complete!")
    return results


if __name__ == "__main__":
    run_benchmark()
