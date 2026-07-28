#!/usr/bin/env python3
"""
RunPod full-stack validation driver (GPU required).

Exercises the REAL path end-to-end on a subset of the labelled query set:
    route (staged router)  ->  load/switch LoRA adapter  ->  generate

Proves the adapter HOT-SWAP mechanism and measures cost, not routing accuracy
(routing accuracy is covered offline by run_eval.py in CI).

Per query it records: routed domain, routing confidence, chosen adapter,
adapter-swap time, peak VRAM, generation latency, and whether text was produced.
Writes metrics.json + summary.md into the evidence directory given as argv[1].

Exit codes (fail loudly, never swallow):
    0 success
    3 CUDA out-of-memory
    4 adapter load/switch failure
    5 base model / environment failure
    6 no GPU available

IMPORTANT (honesty): the domain->adapter map below is a TEST FIXTURE that exists
only in this validation script to exercise all four registry adapters. It is NOT a
production routing policy and NOT a claim that each domain has a matching
specialist adapter. Three of the four registry adapters are general-purpose VISION
adapters; we run them on text prompts purely to prove load/swap/generate works and
to measure VRAM and latency. Output QUALITY is not asserted, only output presence.
"""
import json
import os
import sys
import time
import subprocess
import importlib.util
import pathlib
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
QUERIES = REPO / "eval" / "routing" / "queries.jsonl"
REGISTRY = REPO / "data" / "lora_registry.json"
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 32

# TEST FIXTURE ONLY -- see module docstring. Maps a routed domain to one of the
# four registry adapters so that all four get exercised across the subset.
DOMAIN_TO_ADAPTER = {
    "medical":   "medical_vl",       # the one genuine domain match
    "finance":   "reward_vl",
    "code":      "comics_vl",
    "math":      "point_detect_vl",
    "reasoning": "reward_vl",
    "general":   None,               # abstain -> run base model, no adapter
}


def load_staged_router():
    spec = importlib.util.spec_from_file_location(
        "staged_router_runpod", REPO / "src" / "routing" / "staged_router.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["staged_router_runpod"] = mod
    spec.loader.exec_module(mod)
    return mod.StagedRouter()


def select_subset(rows, n_single=20, n_dual=12, n_ood=8):
    """Deterministic stratified subset spanning all buckets and (for single) domains."""
    rows = sorted(rows, key=lambda r: r["id"])
    single = [r for r in rows if r["bucket"] == "single_signal"]
    dual = [r for r in rows if r["bucket"] == "dual_signal"]
    ood = [r for r in rows if r["bucket"] == "out_of_domain"]
    # even spread of single across the 5 specialist domains
    by_dom = {}
    for r in single:
        by_dom.setdefault(r["expected_domain"], []).append(r)
    per = max(1, n_single // max(1, len(by_dom)))
    picked_single = []
    for d in sorted(by_dom):
        picked_single += by_dom[d][:per]
    picked = picked_single[:n_single] + dual[:n_dual] + ood[:n_ood]
    return sorted(picked, key=lambda r: r["id"])


def nvidia_smi(path):
    try:
        with open(path, "w") as f:
            subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        with open(path, "w") as f:
            f.write(f"nvidia-smi failed: {e}\n")


def main():
    if len(sys.argv) < 2:
        print("usage: runpod_driver.py <evidence_dir>")
        sys.exit(2)
    evid = pathlib.Path(sys.argv[1])
    evid.mkdir(parents=True, exist_ok=True)

    # ---- environment / GPU ----
    try:
        import torch
    except Exception as e:
        print(f"[FATAL] torch import failed: {e}")
        sys.exit(5)
    if not torch.cuda.is_available():
        print("[FATAL] no CUDA GPU available -- this validation requires a real GPU.")
        sys.exit(6)

    import transformers
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "base_model": BASE_MODEL,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        env["peft"] = __import__("peft").__version__
    except Exception:
        env["peft"] = "unavailable"
    (evid / "environment.json").write_text(json.dumps(env, indent=2))
    print("[env]", json.dumps(env))

    registry = json.loads(REGISTRY.read_text())["adapters"]
    router = load_staged_router()
    rows = [json.loads(l) for l in open(QUERIES) if l.strip()]
    subset = select_subset(rows)
    print(f"[info] validating {len(subset)} queries across all buckets")

    # ---- load base model ----
    try:
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except Exception:
            from transformers import AutoModelForCausalLM as VLModel
        print("[load] base model ...")
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = VLModel.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        model.eval()
        env["base_load_seconds"] = round(time.time() - t0, 2)
        print(f"[load] base ready in {env['base_load_seconds']}s")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[FATAL][OOM] base model load OOM: {e}")
        sys.exit(3)
    except Exception as e:
        print(f"[FATAL] base model load failed: {e}\n{traceback.format_exc()}")
        sys.exit(5)

    # ---- load the four adapters (fail loudly) ----
    adapter_load = {}
    for name, info in registry.items():
        hf_path = info["hf_path"]
        try:
            print(f"[adapter] loading {name} <- {hf_path}")
            t0 = time.time()
            model.load_adapter(hf_path, adapter_name=name)
            adapter_load[name] = round(time.time() - t0, 2)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[FATAL][OOM] loading adapter {name}: {e}")
            sys.exit(3)
        except Exception as e:
            print(f"[FATAL] adapter load failed for {name} ({hf_path}): {e}\n{traceback.format_exc()}")
            sys.exit(4)
    env["adapter_load_seconds"] = adapter_load
    print(f"[adapter] all {len(adapter_load)} adapters loaded: {adapter_load}")

    def generate(query):
        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(gen, skip_special_tokens=True)[0]

    # ---- per-query loop ----
    results = []
    mid = len(subset) // 2
    for i, r in enumerate(subset):
        if i == mid:
            nvidia_smi(evid / "nvidia-smi.during.txt")
        q = r["query"]
        route = router.route(q)
        domain, conf = route.domain, round(float(route.confidence), 4)
        adapter = DOMAIN_TO_ADAPTER.get(domain)

        torch.cuda.reset_peak_memory_stats()
        rec = {"id": r["id"], "bucket": r["bucket"], "query": q,
               "routed_domain": domain, "routing_confidence": conf,
               "routing_method": route.method, "chosen_adapter": adapter or "(base,no-adapter)"}
        try:
            # adapter swap timing
            t0 = time.time()
            if adapter is None:
                ctx = model.disable_adapter()
                ctx.__enter__()
            else:
                model.set_adapter(adapter)
            rec["adapter_swap_seconds"] = round(time.time() - t0, 4)

            t0 = time.time()
            text = generate(q)
            rec["generation_seconds"] = round(time.time() - t0, 3)
            rec["output_produced"] = bool(text.strip())
            rec["output_snippet"] = text.strip()[:160]

            if adapter is None:
                ctx.__exit__(None, None, None)

            rec["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            rec["status"] = "ok"
            print(f"  {r['id']} {r['bucket']:14s} dom={domain:9s} conf={conf:.3f} "
                  f"adapter={rec['chosen_adapter']:18s} swap={rec['adapter_swap_seconds']:.3f}s "
                  f"gen={rec['generation_seconds']:.2f}s vram={rec['peak_vram_mb']:.0f}MB "
                  f"out={rec['output_produced']}")
        except torch.cuda.OutOfMemoryError as e:
            rec["status"] = "OOM"
            results.append(rec)
            _flush(evid, env, results)
            print(f"[FATAL][OOM] during query {r['id']}: {e}")
            sys.exit(3)
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)
            results.append(rec)
            _flush(evid, env, results)
            print(f"[FATAL] generation failed on {r['id']}: {e}\n{traceback.format_exc()}")
            sys.exit(4)
        results.append(rec)

    _flush(evid, env, results)
    ok = sum(1 for r in results if r.get("output_produced"))
    print(f"\n[done] {ok}/{len(results)} queries produced output; evidence in {evid}")
    sys.exit(0 if ok == len(results) else 1)


def _flush(evid, env, results):
    peak = max((r.get("peak_vram_mb", 0) for r in results), default=0)
    agg = {
        "n": len(results),
        "produced_output": sum(1 for r in results if r.get("output_produced")),
        "peak_vram_mb_overall": peak,
        "mean_generation_seconds": round(
            sum(r.get("generation_seconds", 0) for r in results) / max(1, len(results)), 3),
        "mean_adapter_swap_seconds": round(
            sum(r.get("adapter_swap_seconds", 0) for r in results) / max(1, len(results)), 4),
    }
    (evid / "metrics.json").write_text(json.dumps(
        {"environment": env, "aggregate": agg, "per_query": results}, indent=2))
    lines = [
        f"# RunPod full-stack validation — {env.get('timestamp_utc','')}",
        "",
        f"- GPU: **{env.get('gpu_name','?')}**  | driver/CUDA {env.get('cuda','?')} "
        f"| torch {env.get('torch','?')} | transformers {env.get('transformers','?')}",
        f"- Base model: `{env.get('base_model','?')}` "
        f"(load {env.get('base_load_seconds','?')}s)",
        f"- Adapters loaded: {list((env.get('adapter_load_seconds') or {}).keys())}",
        "",
        "## Aggregate",
        f"- queries: {agg['n']}",
        f"- produced output: {agg['produced_output']}/{agg['n']}",
        f"- peak VRAM: {agg['peak_vram_mb_overall']} MB",
        f"- mean generation: {agg['mean_generation_seconds']} s",
        f"- mean adapter swap: {agg['mean_adapter_swap_seconds']} s",
        "",
        "See `metrics.json` for per-query detail and `nvidia-smi.*.txt` for GPU state.",
    ]
    (evid / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
