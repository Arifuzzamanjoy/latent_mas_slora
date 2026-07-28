# No RunPod evidence exists yet

**Status: the GPU validation has NOT been run. This directory contains no evidence.**

This file exists so an empty directory with a `.gitkeep` is not mistaken for a validation
that happened and produced nothing. Nothing here has touched a GPU.

## What has actually been done

- `eval/routing/runpod_validate.sh` — written, and syntax-checked with `bash -n`.
- `eval/routing/runpod_driver.py` — written, and syntax-checked with `python -m py_compile`.

That is the full extent of it. **Syntax-checked is not tested.** Neither script has been
executed against a real GPU, a real base model, or a real adapter. Specifically unverified:

- that `Qwen/Qwen2.5-VL-7B-Instruct` loads under the pinned dependency versions,
- that `model.load_adapter(...)` succeeds for all four registry adapters,
- that `model.set_adapter(...)` hot-swap works as the driver assumes,
- that the processor/chat-template call produces usable generation input,
- that the VRAM and timing instrumentation reports sane values,
- that the OOM and adapter-failure paths exit with the codes the script expects.

Any of these could fail on first contact with real hardware. Assume a debugging pass is
needed rather than a clean first run.

## What a real run would produce here

`bash eval/routing/runpod_validate.sh` writes a timestamped pack at
`eval/routing/results/runpod/<UTC timestamp>/` containing:

| File | Contents |
|---|---|
| `raw.log` | full stdout/stderr of the run |
| `environment.txt` / `environment.json` | GPU model, driver, CUDA, torch/transformers/peft versions |
| `nvidia-smi.before.txt` | GPU state before loading anything |
| `nvidia-smi.during.txt` | GPU state at the midpoint of the query loop |
| `nvidia-smi.after.txt` | GPU state after the run (written even on failure) |
| `metrics.json` | per-query: routed domain, confidence, chosen adapter, adapter-swap time, peak VRAM, generation latency, output produced |
| `summary.md` | aggregate: queries run, outputs produced, peak VRAM, mean swap and generation time |

It runs 40 queries drawn deterministically from `queries.jsonl` across all three buckets.

## Approximate cost of a run

Roughly **5–12 minutes** on a cold pod (most of it downloading ~16 GB of base weights),
or 3–5 minutes with weights cached on a network volume. On a 24–48 GB card that puts a
single run **well under $0.25** at typical RunPod rates. Pricing changes; check the console
rather than trusting this line. See `../../RUNPOD.md` for the pod spec and exact commands.

## When it is run

This file gets **deleted and replaced** by the real evidence pack. If you are reading this,
that has not happened yet — treat the full-stack path as unproven.
