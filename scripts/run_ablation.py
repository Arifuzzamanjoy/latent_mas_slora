#!/usr/bin/env python3
"""
CLI – run the LatentMAS ablation study.

Examples
--------
# Quick smoke-test (5 samples, 2 configs, 1 benchmark)
python scripts/run_ablation.py --benchmarks gsm8k --max-samples 5 \
    --configs A_single_model D_latentmas_speclora

# Full ablation on two benchmarks
python scripts/run_ablation.py --benchmarks gsm8k medqa --max-samples 200

# All four configs on all benchmarks (the real experiment)
python scripts/run_ablation.py --benchmarks gsm8k medqa arc_challenge \
    --max-samples -1 --output-dir results/ablation_full/

# With external LoRA adapters for Config D
python scripts/run_ablation.py --benchmarks medqa --max-samples 100 \
    --lora-planner org/planner-lora --lora-critic org/critic-lora \
    --lora-refiner org/refiner-lora --lora-judger org/judger-lora

# With a shared adapter for Config C
python scripts/run_ablation.py --benchmarks gsm8k --max-samples 50 \
    --shared-lora org/general-lora
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on the import path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the LatentMAS 4-config ablation study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- what to run ---
    p.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k"],
        help="Benchmark names (default: gsm8k). "
             "Choices: gsm8k, medqa, arc_challenge, humaneval_plus",
    )
    p.add_argument(
        "--configs", nargs="+", default=None,
        help="Subset of configs to run. Default = all four. "
             "Choices: A_single_model, B_latentmas_nolora, "
             "C_latentmas_samelora, D_latentmas_speclora",
    )
    p.add_argument("--max-samples", type=int, default=-1,
                   help="Max examples per benchmark (-1 = full set)")
    p.add_argument("--seed", type=int, default=42)

    # --- model ---
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model id")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "4bit"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default="/home/caches")

    # --- LoRA paths ---
    p.add_argument("--lora-planner", default=None, help="HF path for planner adapter (Config D)")
    p.add_argument("--lora-critic", default=None, help="HF path for critic adapter (Config D)")
    p.add_argument("--lora-refiner", default=None, help="HF path for refiner adapter (Config D)")
    p.add_argument("--lora-judger", default=None, help="HF path for judger adapter (Config D)")
    p.add_argument("--shared-lora", default=None,
                   help="Single HF adapter path used for all agents in Config C")

    # --- latent reasoning ---
    p.add_argument("--latent-steps", type=int, default=10,
                   help="Latent reasoning steps for multi-agent configs")

    # --- output ---
    p.add_argument("--output-dir", default="results/ablation/")
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build lora_configs dict for Config D
    lora_configs: dict[str, str] = {}
    if args.lora_planner:
        lora_configs["planner"] = args.lora_planner
    if args.lora_critic:
        lora_configs["critic"] = args.lora_critic
    if args.lora_refiner:
        lora_configs["refiner"] = args.lora_refiner
    if args.lora_judger:
        lora_configs["judger"] = args.lora_judger

    # Import after path setup so src/ is resolvable
    from src.evaluation.ablation import AblationStudy

    study = AblationStudy(
        model_name=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        lora_configs=lora_configs or None,
        shared_lora_path=args.shared_lora,
        latent_steps=args.latent_steps,
    )

    result = study.run_all(
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        configs=args.configs,
    )

    # Summary
    print("\n" + "=" * 64)
    print("ABLATION COMPLETE")
    print("=" * 64)
    print(result.comparison_table())
    print(f"\nWall time: {result.wall_time_s:.1f}s")
    if result.significance:
        print("\nStatistical significance (B vs D):")
        for bench, s in result.significance.items():
            print(f"  {bench}: p={s['p_value']:.4f}  {'*' if s['significant'] else 'ns'}  Δ={s['delta']:+.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
