#!/usr/bin/env python3
"""
LatentMAS evaluation CLI.

Every knob is an argument, every method is independently selectable, and the
dataset can be sliced to any fraction. See docs/EVAL.md for the full reference.

Examples
--------
  # what is available
  python run_eval.py --list-methods --list-datasets

  # exercise the whole pipeline with no GPU and no weights
  python run_eval.py --methods all --dataset sample --dry-run

  # one method, 10% of MedQA
  python run_eval.py --methods latent-mas --dataset medqa --fraction 0.1

  # the core comparison at 10% / 50% / 100%
  python run_eval.py --methods core --dataset medqa --fractions 0.1,0.5,1.0

  # ablate latent depth, everything else held fixed
  python run_eval.py --methods latent-mas-kv --dataset medqa --fraction 0.1 \
      --set latent-mas-kv.latent_steps=0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.config import EvalConfig, parse_fraction, parse_set_args
from eval.data import HF_DATASETS
from eval.methods import METHOD_GROUPS, list_methods


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Evaluation pipeline for LatentMAS + S-LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_argument_group("discovery")
    g.add_argument("--list-methods", action="store_true", help="print methods and exit")
    g.add_argument("--list-datasets", action="store_true", help="print datasets and exit")
    g.add_argument("--report-only", metavar="RUN_DIR",
                   help="rebuild report from an existing run directory and exit")
    g.add_argument("--print-config", action="store_true",
                   help="print the resolved config as JSON and exit")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    g.add_argument("--device", default="cuda")
    g.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "4bit"])
    g.add_argument("--cache-dir", default="/home/caches")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="sample",
                   help="sample | local:path.json | medqa | gsm8k | mmlu:anatomy | mix:a,b")
    g.add_argument("--split", default=None, help="override the dataset split")
    g.add_argument("--fraction", default="1.0",
                   help="fraction of the dataset (0.1 = 10%%, 0.001 = 0.1%%). "
                        "A percentage also works: 0.1%%. Any value > 0 is allowed; "
                        "a fraction below one item still evaluates one item")
    g.add_argument("--fractions", default=None,
                   help="comma list, runs one eval per fraction and draws the "
                        "scaling curve, e.g. 0.001,0.01,0.1,1.0")
    g.add_argument("--min-per-group", type=int, default=1,
                   help="items each stratum keeps at tiny fractions (0 = allow "
                        "a stratum to drop out entirely)")
    g.add_argument("--limit", type=int, default=None, help="hard cap on item count")
    g.add_argument("--offset", type=int, default=0, help="skip this many items first")
    g.add_argument("--data-seed", type=int, default=0, help="seed for the segment shuffle")
    g.add_argument("--no-shuffle", action="store_true",
                   help="keep dataset order (slices become the first N items)")
    g.add_argument("--stratify-by", default="domain",
                   help="field to stratify the segment on, or 'none'")

    g = p.add_argument_group("methods")
    g.add_argument("--methods", default="baseline-cot",
                   help="comma list of method names or groups: " + ", ".join(METHOD_GROUPS))
    g.add_argument("--set", dest="set_args", action="append", default=[], metavar="M.KEY=VAL",
                   help="per-method override, repeatable, e.g. --set latent-mas.latent_steps=25")

    g = p.add_argument_group("decoding")
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--seeds", default="0", help="comma list of run seeds, e.g. 0,1,2")
    g.add_argument("--self-consistency", type=int, default=1,
                   help="samples per item, majority-voted (needs --temperature > 0)")

    g = p.add_argument_group("multi-agent")
    g.add_argument("--latent-steps", type=int, default=10)
    g.add_argument("--agents", default=None,
                   help="comma list to force a fixed pipeline, e.g. Planner,Critic,Judger")
    g.add_argument("--no-router", action="store_true", help="disable semantic routing")
    g.add_argument("--no-adaptive-steps", action="store_true",
                   help="do not vary latent steps by routed domain")
    g.add_argument("--loras", default="", help="comma list of registry LoRAs to load")
    g.add_argument("--kv-handoff", dest="kv_handoff", action="store_true", default=None,
                   help="hand the latent KV cache to the final decoder (reference behaviour)")
    g.add_argument("--no-kv-handoff", dest="kv_handoff", action="store_false",
                   help="discard the cache before decoding (legacy behaviour)")
    g.add_argument("--prompt-style", default=None, choices=["reason_first", "answer_first"],
                   help="judger prompt order; reason_first matches the reference")
    g.add_argument("--adapter-policy", default=None,
                   choices=["logo", "merge", "same", "none"],
                   help="how multi-lora composes adapters (default logo: probe, top-k, merge)")
    g.add_argument("--top-k", type=int, default=None,
                   help="adapters merged per instance by multi-lora (default 3)")

    g = p.add_argument_group("protocol")
    g.add_argument("--scoring", default="generate", choices=["generate", "loglikelihood"])
    g.add_argument("--permute-options", default="none", choices=["none", "cyclic", "all"],
                   help="answer-position bias test")

    g = p.add_argument_group("output")
    g.add_argument("--out-dir", default="eval_runs")
    g.add_argument("--run-name", default=None)
    g.add_argument("--no-resume", action="store_true", help="ignore existing records")
    g.add_argument("--no-generations", action="store_true", help="do not store model text")
    g.add_argument("--no-markdown", action="store_true")
    g.add_argument("--no-plots", action="store_true", help="skip comparison figures")
    g.add_argument("--live-plot", action="store_true",
                   help="refresh live.png while the eval runs")
    g.add_argument("--live-every", type=int, default=5,
                   help="records between live.png refreshes (default 5)")
    g.add_argument("--scaling-from", default=None, metavar="DIR",
                   help="build scaling.png from every run under DIR, then exit")
    g.add_argument("--progress", type=int, default=1, metavar="N",
                   help="print a progress line every N items (default 1, 0 = silent)")
    g.add_argument("-v", "--verbose", action="store_true")

    g = p.add_argument_group("analysis")
    g.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples, 0 disables")
    g.add_argument("--ci", type=float, default=0.95)
    g.add_argument("--compare-to", default=None,
                   help="reference method for paired tests (default: first baseline present)")

    g = p.add_argument_group("execution")
    g.add_argument("--dry-run", action="store_true",
                   help="mock backend: no weights, no GPU, exercises the full pipeline")
    g.add_argument("--config", default=None, help="load an EvalConfig json and apply args on top")

    return p


def config_from_args(a: argparse.Namespace) -> EvalConfig:
    cfg = EvalConfig.from_file(a.config) if a.config else EvalConfig()
    cfg.model = a.model
    cfg.device = a.device
    cfg.dtype = a.dtype
    cfg.cache_dir = a.cache_dir

    cfg.dataset = a.dataset
    cfg.split = a.split
    cfg.fraction = parse_fraction(a.fraction)
    cfg.limit = a.limit
    cfg.offset = a.offset
    cfg.data_seed = a.data_seed
    cfg.shuffle = not a.no_shuffle
    cfg.stratify_by = None if a.stratify_by in ("none", "", None) else a.stratify_by
    cfg.min_per_group = a.min_per_group

    cfg.methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    cfg.method_args = parse_set_args(a.set_args)

    cfg.max_new_tokens = a.max_new_tokens
    cfg.temperature = a.temperature
    cfg.top_p = a.top_p
    cfg.seeds = [int(s) for s in str(a.seeds).split(",") if str(s).strip()]
    cfg.self_consistency = a.self_consistency

    cfg.latent_steps = a.latent_steps
    cfg.agents = [x.strip() for x in a.agents.split(",")] if a.agents else None
    cfg.use_router = not a.no_router
    cfg.adaptive_latent_steps = not a.no_adaptive_steps
    cfg.loras = [x.strip() for x in a.loras.split(",") if x.strip()]
    if a.kv_handoff is not None:
        cfg.kv_handoff = a.kv_handoff
    if a.prompt_style is not None:
        cfg.prompt_style = a.prompt_style
    if a.adapter_policy is not None:
        cfg.adapter_policy = a.adapter_policy
    if a.top_k is not None:
        cfg.top_k = a.top_k

    cfg.scoring = a.scoring
    cfg.permute_options = a.permute_options

    cfg.out_dir = a.out_dir
    cfg.run_name = a.run_name
    cfg.resume = not a.no_resume
    cfg.save_generations = not a.no_generations
    cfg.report_markdown = not a.no_markdown
    cfg.plots = not a.no_plots
    cfg.live_plot = a.live_plot
    cfg.live_every = a.live_every
    cfg.progress_every = a.progress

    cfg.bootstrap = a.bootstrap
    cfg.ci = a.ci
    cfg.compare_to = a.compare_to
    cfg.dry_run = a.dry_run
    return cfg


def _hf_login() -> None:
    """Authenticate for gated datasets/models if a token is in the environment.

    Set it outside the repo - `export HF_TOKEN=...` - never in a config file
    that gets committed.
    """
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[hf] authenticated from HF_TOKEN")
    except Exception as e:
        print(f"[hf] login failed ({e}); continuing anonymously")


def main() -> int:
    a = build_parser().parse_args()
    _hf_login()

    if a.list_methods:
        print(f"{'method':<18} {'backend':<8} description")
        print("-" * 100)
        for m in list_methods():
            print(f"{m['name']:<18} {m['backend']:<8} {m['description']}")
        print("\ngroups:")
        for g, ms in METHOD_GROUPS.items():
            print(f"  {g:<12} {', '.join(ms)}")

    if a.list_datasets:
        print(f"\n{'dataset':<12} description")
        print("-" * 100)
        print(f"{'sample':<12} data/sample_data.json (the repo's 5 questions)")
        print(f"{'local:PATH':<12} any local json/jsonl with question/choices/answer fields")
        for k, v in HF_DATASETS.items():
            print(f"{k:<12} {v}")
        print(f"{'mix:a,b':<12} concatenate several of the above")

    if a.list_methods or a.list_datasets:
        return 0

    if a.report_only:
        from eval.report import report_only
        report_only(a.report_only)
        return 0

    if a.scaling_from:
        from eval.plots import scaling_from_runs
        root = Path(a.scaling_from)
        runs = sorted(d for d in root.iterdir() if (d / "summary.json").exists())
        path = scaling_from_runs(runs, root)
        print(f"[plots] scaling curve from {len(runs)} runs -> {path}")
        return 0

    cfg = config_from_args(a)

    if a.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return 0

    if a.self_consistency > 1 and a.temperature == 0.0:
        print("[warn] --self-consistency > 1 with greedy decoding produces identical samples; "
              "set --temperature 0.7 for real vote diversity.")

    from eval.runner import Runner

    fractions = ([parse_fraction(x) for x in a.fractions.split(",")]
                 if a.fractions else [cfg.fraction])
    base_run_name = cfg.run_name
    run_dirs = []

    for frac in fractions:
        cfg.fraction = frac
        if len(fractions) > 1:
            cfg.run_name = f"{base_run_name or 'scan'}-frac{frac:g}"
            print(f"\n########## fraction = {frac:g} ##########")
        runner = Runner(cfg, dry_run=a.dry_run, verbose=a.verbose)
        runner.run()
        run_dirs.append(runner.run_dir)

    if len(run_dirs) > 1 and cfg.plots:
        from eval.plots import scaling_from_runs
        path = scaling_from_runs(run_dirs, Path(cfg.out_dir))
        if path:
            print(f"[plots] scaling curve across {len(run_dirs)} fractions -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
