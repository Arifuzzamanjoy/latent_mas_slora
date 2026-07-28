# RunPod full-stack validation

The CI job (`.github/workflows/routing-eval.yml`) proves **routing** only, on CPU.
This document is the **GPU** proof: that the full path — route → load/swap LoRA
adapter → generate — actually runs on real hardware with the real
`Qwen/Qwen2.5-VL-7B-Instruct` base and the four registry adapters
(`medical_vl`, `reward_vl`, `comics_vl`, `point_detect_vl`).

It measures the **mechanism and its cost** (adapter hot-swap time, peak VRAM,
generation latency, output presence) — not routing accuracy, and not output
quality. See the honesty note at the bottom.

> Nothing here has been run yet. `results/runpod/` is intentionally empty (only a
> `.gitkeep`). Run it on a real pod; the script writes a timestamped evidence pack.

## Pod spec

| Item | Recommended |
|---|---|
| GPU | **≥ 24 GB VRAM**. A5000 (24 GB) is the floor; **A6000 (48 GB)** or A100 (40/80 GB) is comfortable. |
| Base image | RunPod PyTorch template (CUDA 12.x, torch ≥ 2.1) **or** the repo's Docker image. |
| Container disk | **≥ 40 GB** (base weights ≈ 16 GB bf16 + adapters + HF cache). |
| Network volume | Optional but recommended to cache the model between runs. |
| Env vars | `HF_TOKEN` if you hit gated/rate-limited downloads. |

VRAM budget: base bf16 ≈ 16 GB; each LoRA adapter ≈ tens of MB; activations for a
32-token generation are small. 24 GB works; 40 GB+ removes all doubt.

## Exact commands to paste

```bash
# 1. On the pod, from a shell in the repo root (clone first if needed):
cd /workspace/latent_mas_slora        # adjust to your clone path

# 2. Ensure deps are present (the repo's requirements, on GPU):
pip install -r requirements.txt

# 3. Run the validation. It writes eval/routing/results/runpod/<timestamp>/
bash eval/routing/runpod_validate.sh

# 4. Inspect the evidence pack:
ls -R eval/routing/results/runpod/
cat eval/routing/results/runpod/*/summary.md
```

That single `bash eval/routing/runpod_validate.sh` does everything: captures the
environment and `nvidia-smi` before/during/after, loads the base + four adapters,
routes and generates on a 40-query subset spanning all buckets, and writes
`raw.log`, `environment.txt/json`, `nvidia-smi.{before,during,after}.txt`,
`metrics.json`, and `summary.md`.

## Expected runtime

| Phase | First run (cold) | Warm (weights cached) |
|---|---|---|
| Base model download + load | 3–8 min (≈16 GB download) | 30–90 s |
| Load 4 adapters | 10–60 s | 10–30 s |
| 40 queries (route + swap + 32-token gen) | 1–3 min | 1–3 min |
| **Total** | **~5–12 min** | **~3–5 min** |

## Expected cost (approximate — verify current RunPod pricing)

RunPod on-demand rates change; check the console. As a rough guide, at the time of
writing community/secure-cloud rates are on the order of **$0.3–0.8/hr for a
24–48 GB card (A5000/A6000)** and **~$1.5–2.5/hr for an A100**. A single run is
5–12 minutes, so **a full validation typically costs well under $0.25**. Stop the
pod afterward so idle time does not accrue.

## What a HEALTHY run looks like

- Exit code **0**; script prints `[VALIDATION PASSED]`.
- `summary.md`: `produced output: 40/40`.
- All four adapters appear in `environment.json` → `adapter_load_seconds`.
- `metrics.json` per-query: every record `status: ok`, `output_produced: true`,
  a plausible `peak_vram_mb` (roughly base + small overhead, e.g. ~16–20 GB on a
  7B bf16 model), sub-second-to-few-second `generation_seconds`, and small
  `adapter_swap_seconds`.
- `nvidia-smi.during.txt` shows the process resident on the GPU with VRAM in use.

## What an UNHEALTHY run looks like

- **Exit 3 / `CUDA OUT OF MEMORY`** — GPU too small or fragmented. Use a larger
  card. The evidence pack still contains everything up to the failure.
- **Exit 4 / `ADAPTER LOAD/SWAP FAILURE`** — an adapter hf_path is unreachable,
  gated, or incompatible with the base. Check `raw.log` for the offending name.
- **Exit 5 / base model or environment failure** — bad deps or download failure.
- **Exit 6 / no GPU** — you launched on a CPU pod.
- `summary.md` showing `produced output: N<40`, or any `status: error` record.

The script fails **loudly** and preserves partial evidence in all these cases — it
never swallows an error or invents a number.

## Honesty note on the domain→adapter mapping

The driver contains a `DOMAIN_TO_ADAPTER` **test fixture** used only to exercise
all four adapters. It is **not** a production routing policy and does not imply a
specialist adapter per domain. Three of the four registry adapters are
general-purpose **vision** adapters; running them on text prompts here proves the
load/swap/generate mechanics and measures VRAM/latency — it does **not** assert
that the generated text is good. Routing *quality* is measured separately and
offline by `run_eval.py`. See the LIMITATIONS section of `README.md`.
