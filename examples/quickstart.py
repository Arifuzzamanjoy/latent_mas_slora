#!/usr/bin/env python3
"""
Quick Start – LatentMAS clean API.

Demonstrates the refactored orchestrator: load a config, register
agents, run a question, read the result.
"""

import sys

sys.path.insert(0, "/workspace/latent_mas_slora")

from src import LatentMASSystem, AgentConfig


def main():
    print("=" * 60)
    print("LatentMAS Quick Start")
    print("=" * 60)

    # 1. Initialise from YAML (falls back to defaults if file missing)
    system = LatentMASSystem(
        "configs/base.yaml",
        cache_dir="/home/caches",
    )

    # 2. Register the standard agent hierarchy
    system.add_agent(AgentConfig.planner())
    system.add_agent(AgentConfig.critic())
    system.add_agent(AgentConfig.refiner())
    system.add_agent(AgentConfig.judger())

    print(f"\nRegistered agents: {system.pool.list_agents()}")

    # 3. Run a question
    question = (
        "A 34-year-old man presents with 3 weeks of colicky abdominal pain "
        "and bloody diarrhea (10-12 bowel movements/day). Colonoscopy shows "
        "bleeding, ulcerated rectal mucosa with pseudopolyps. Which of the "
        "following is this patient at greatest risk of developing?\n"
        "A. Hemolytic uremic syndrome\n"
        "B. Oral ulcers\n"
        "C. Colorectal cancer\n"
        "D. Pancreatic cancer"
    )

    print(f"\nRunning pipeline on: {question[:80]}…")
    result = system.run(question)

    # 4. Print results
    print(f"\n{'=' * 60}")
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.final_answer)

    print(f"\n{'=' * 60}")
    print("STATS")
    print("=" * 60)
    print(f"  Total tokens : {result.total_tokens}")
    print(f"  Latency      : {result.total_latency_ms} ms")
    print(f"  Latent steps : {result.latent_steps_total}")

    # 5. With observability
    print(f"\n{'=' * 60}")
    print("WITH OBSERVABILITY")
    print("=" * 60)
    result_obs = system.run("What is the capital of France?", observe=True)
    obs = result_obs.metadata.get("observability", {})
    for agent in obs.get("agents", []):
        print(
            f"  {agent['name']:12s}  "
            f"steps={agent['latent_steps_taken']}  "
            f"time={agent['wall_time_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
