#!/usr/bin/env python3
"""
Quick Start Demo for LatentMAS + S-LoRA System

Shows basic usage with Qwen2.5-3B and multiple agents.
Optimized for 24-48GB VRAM.
"""

import sys
sys.path.insert(0, '/workspace/latent_mas_slora')

from src import LatentMASSystem, AgentConfig


def main():
    print("=" * 60)
    print("LatentMAS + S-LoRA Quick Start Demo")
    print("=" * 60)
    
    # Initialize system with Qwen2.5-3B
    print("\n[1] Initializing system...")
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype="bfloat16",  # Full precision for 48GB
        latent_steps=15,
        cache_dir="/home/caches",
    )
    
    # Add specialized agents
    print("\n[2] Adding agents...")
    system.add_agent(AgentConfig.planner())
    system.add_agent(AgentConfig.critic())
    system.add_agent(AgentConfig.refiner())
    system.add_agent(AgentConfig.judger())
    
    print(f"    Registered agents: {system._pool.list_agents()}")
    
    # Example question
    question = """A 34-year-old man comes to the physician because of a 3-week history of colicky abdominal pain and diarrhea. He has bowel movements 10–12 times daily; the stool contains blood and mucus. Colonoscopy shows a bleeding, ulcerated rectal mucosa with several pseudopolyps. Which of the following is this patient at greatest risk of developing?
A. Hemolytic uremic syndrome
B. Oral ulcers
C. Colorectal cancer
D. Pancreatic cancer"""

    print("\n[3] Running hierarchical pipeline...")
    print(f"    Question: {question[:100]}...")
    
    result = system.run(
        question=question,
        pipeline="hierarchical",
        max_new_tokens=400,
        temperature=0.5,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for agent_output in result.agent_outputs:
        print(f"\n[{agent_output['agent']}] ({agent_output['latency_ms']}ms)")
        print(f"  Output: {agent_output['output'][:200]}...")
    
    print(f"\n{'=' * 60}")
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.final_answer)
    
    print(f"\n{'=' * 60}")
    print("STATISTICS")
    print("=" * 60)
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Total latency: {result.total_latency_ms}ms")
    print(f"  Latent steps: {result.latent_steps_total}")
    print(f"  Num agents: {len(result.agent_outputs)}")
    
    # System stats
    stats = system.get_stats()
    print(f"\n[System Stats]")
    print(f"  Model: {stats['model_name']}")
    print(f"  Device: {stats['device']}")
    print(f"  Dtype: {stats['dtype']}")


if __name__ == "__main__":
    main()
