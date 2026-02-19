#!/usr/bin/env python3
"""
Interactive pipeline debugger for LatentMAS.

Run a question through the hierarchical pipeline with full observability:
  • Per-agent latent-step counts, wall-time, convergence curves
  • Optional checkpoint decoding (see what agents "were thinking")
  • VRAM tracking

Usage
-----
    python scripts/debug_pipeline.py "What is the mechanism of action of aspirin?"

    # With checkpoint decoding (slower but more informative)
    python scripts/debug_pipeline.py --decode "Explain gradient descent."

    # Choose pipeline mode
    python scripts/debug_pipeline.py --mode run_true_latent "Some question"

    # Custom latent steps
    python scripts/debug_pipeline.py --latent-steps 20 "Some question"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
import time

import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.system import LatentMASSystem
from src.observability.metrics import MetricsCollector
from src.observability.checkpoint_decode import CheckpointDecoder

logger = logging.getLogger("debug_pipeline")


# ======================================================================
# Formatting helpers
# ======================================================================

def _bar(value: float, max_val: float = 1.0, width: int = 30) -> str:
    """Render a simple ASCII progress bar."""
    filled = int(width * min(value / max_val, 1.0))
    return "█" * filled + "░" * (width - filled)


def print_agent_metrics(agent) -> None:
    """Pretty-print one AgentMetrics object."""
    conv = "yes" if agent.converged() else "no"
    print(f"\n  ── {agent.name} {'─' * (50 - len(agent.name))}")
    print(f"  │  Mode              : {agent.mode}")
    print(f"  │  Latent steps      : {agent.latent_steps_taken} / {agent.latent_steps_max}")
    print(f"  │  Converged         : {conv}" +
          (f" (step {agent.converged_at_step})" if agent.converged() else ""))
    print(f"  │  Wall time         : {agent.wall_time_ms:.1f} ms")
    print(f"  │  KV cache tokens   : {agent.kv_cache_size_tokens}")
    print(f"  │  Hidden-state norm : {agent.hidden_state_norm:.4f}")

    if agent.convergence_curve:
        print(f"  │  Convergence curve :")
        for step_i, cos in enumerate(agent.convergence_curve):
            bar = _bar(cos, max_val=1.0, width=25)
            print(f"  │    step {step_i + 1:2d}  {bar}  {cos:.6f}")


def print_pipeline_summary(pm) -> None:
    """Pretty-print a PipelineMetrics object."""
    print("\n" + "=" * 60)
    print("  PIPELINE OBSERVABILITY REPORT")
    print("=" * 60)
    print(f"  Question      : {pm.question[:80]}{'…' if len(pm.question) > 80 else ''}")
    print(f"  Pipeline mode : {pm.pipeline_mode}")
    print(f"  Total time    : {pm.total_wall_time_ms:.1f} ms")
    print(f"  Total latent  : {pm.total_latent_steps} steps")
    print(f"  Peak VRAM     : {pm.peak_vram_mb:.1f} MB")

    for agent in pm.agents:
        print_agent_metrics(agent)

    print("\n" + "=" * 60)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a question through LatentMAS with full observability.",
    )
    parser.add_argument("question", type=str, help="Question to process.")
    parser.add_argument(
        "--mode", choices=["run", "run_true_latent"], default="run_true_latent",
        help="Pipeline method to invoke (default: run_true_latent).",
    )
    parser.add_argument(
        "--decode", action="store_true",
        help="Enable checkpoint decoding (force-decode hidden states to text).",
    )
    parser.add_argument(
        "--decode-tokens", type=int, default=40,
        help="Max tokens per checkpoint decode (default: 40).",
    )
    parser.add_argument(
        "--latent-steps", type=int, default=15,
        help="Latent reasoning steps per agent (default: 15).",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/home/caches",
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print raw JSON metrics instead of pretty output.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Load model via LatentMASSystem (reuses existing infra)
    # ------------------------------------------------------------------
    print(f"\n⏳  Loading model: {args.model}")
    t0 = time.time()
    system = LatentMASSystem(
        model_name=args.model,
        cache_dir=args.cache_dir,
        latent_steps=args.latent_steps,
    )
    system.add_default_agents()
    print(f"✓  Model loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Run with MetricsCollector
    # ------------------------------------------------------------------
    collector = MetricsCollector(
        question=args.question,
        pipeline_mode=args.mode,
    )

    pipeline = system.pipeline
    if args.mode == "run":
        result = pipeline.run(
            question=args.question,
            metrics_collector=collector,
        )
    else:
        result = pipeline.run_true_latent(
            question=args.question,
            metrics_collector=collector,
        )

    pm = collector.finalize()  # safe to call again - idempotent

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    if args.json:
        print(json.dumps(pm.to_dict(), indent=2))
    else:
        print_pipeline_summary(pm)

        print(f"\n  FINAL ANSWER:\n  {'─' * 56}")
        for line in result.final_answer.split("\n"):
            print(f"  {line}")

    # ------------------------------------------------------------------
    # Optional: checkpoint decoding
    # ------------------------------------------------------------------
    if args.decode:
        print(f"\n  CHECKPOINT DECODING (max {args.decode_tokens} tokens each)")
        print(f"  {'─' * 56}")

        decoder = CheckpointDecoder(
            model=system.model,
            tokenizer=system.tokenizer,
            device=str(system.device),
        )

        for agent_out in result.agent_outputs:
            agent_name = agent_out["agent"]
            hidden = system.pipeline.memory.get_hidden_state(agent_name)
            if hidden is not None:
                text = decoder.decode_hidden(hidden, max_new_tokens=args.decode_tokens)
                print(f"\n  [{agent_name}] would say:")
                print(f"    {text[:200]}")

                probe = decoder.probe_confidence(hidden, top_k=5)
                top_str = ", ".join(
                    f"{tok} ({p:.3f})" for tok, p in probe["top_tokens"]
                )
                print(f"    Confidence: max_prob={probe['max_prob']:.3f}  "
                      f"entropy={probe['entropy']:.3f}")
                print(f"    Top tokens: {top_str}")

    print()


if __name__ == "__main__":
    main()
