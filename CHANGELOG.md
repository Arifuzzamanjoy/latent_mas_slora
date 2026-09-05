# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-09-05

### Added
- Evaluation harness (`eval/`, `run_eval.py`): 11 independently selectable
  methods, 11 dataset loaders, nested fractional slices, self-consistency,
  option permutation, resume, paired statistics and eight plot types.
  See [docs/EVAL.md](docs/EVAL.md).
- `multi-lora`: LoGo-style ([2511.07129](https://arxiv.org/html/2511.07129v3))
  per-instance adapter composition — probe, top-k, weighted merge — as an
  alternative to switching adapters mid-chain.
- `tools/validate_chain.py`: verifies the chain switches adapters, that they
  affect the computation, and that the latent memory reaches the final agent.
- Packaging (`pyproject.toml`), pinned requirements, `Makefile`, GitHub Actions
  CI, and pre-commit hooks.

### Fixed
- **The latent KV cache never reached the decoder.** `run_true_latent` built the
  shared working memory and then called `generate()` without it, making the
  latent collaboration inert: the pipeline scored identically to a single prompt
  on 124 of 127 MedQA items. `kv_handoff=True` is now the default; the previous
  behaviour remains reachable for comparison.
- **The judger prompt asked for the answer before the reasoning.** Now
  reason-first by default, matching the reference implementation, and
  format-neutral so numeric datasets work. `prompt_style="answer_first"` keeps
  the old order.
- Item-id collisions in `mix:` datasets silently dropped 36% of a segment
  (150 items evaluated as 96), because ids are the dedup key.
- `multi-lora` scored adapters by absolute hidden-state norm, which ranked
  untrained adapters above a trained one; it now scores each adapter's deviation
  from the base model, so an identity adapter scores exactly 0.
- Weight normalization drove the lowest-ranked selected adapter to zero weight,
  turning any `top_k` into `top_k - 1`.
- `LoRAAdapterManager` called `torch.cuda.memory_allocated()` on a CPU device,
  making `--device cpu` unusable on a GPU host.
- Progress printed every 25 items, which at 10-30s per item was indistinguishable
  from a hung process. Now one line per item with running accuracy and an ETA.

### Changed
- README benchmark table replaced with measured results. The previous figures
  (78% vs 45% accuracy, 3-7x speedup) were not reproducible: the measured gap
  between the best pipeline and plain chain-of-thought is +3.9pp, p=0.49, at 1.9x
  the tokens and 3x the latency.
- LoRA registry: removed `coder_7b` (upstream 404) and `medical_instruct`
  (duplicate). Remaining entries carry a `verified` flag reflecting whether their
  `lora_B` tensors were checked to be non-zero.
- Resume refuses to continue a run whose configuration or segment changed, rather
  than mixing two conditions into one result file.

### Removed
- `evaluate_latent_collaboration.py` and `manual_eval.py` — superseded by
  `run_eval.py`, which covers both as method selections.
- `benchmark_results.json`, `results_*.json` — run artifacts; `eval_runs/` is the
  output directory and is git-ignored.
- Root `__init__.py` — mutated `sys.path` on import and forced torch into test
  collection. Use `from src import ...`, or install the package.
- `run.sh` — replaced by `make`.

## [0.1.0]

Initial implementation: latent reasoning core, agent pool with LoRA switching,
hierarchical and sequential pipelines, semantic router.
