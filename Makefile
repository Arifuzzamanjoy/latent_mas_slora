# LatentMAS + S-LoRA — common tasks.
# `make help` lists everything.

PY      ?= python
VENV    ?= venv
BIN     := $(VENV)/bin
MODEL   ?= Qwen/Qwen2.5-7B-Instruct
DATASET ?= medqa
FRACTION ?= 0.1

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev test test-fast lint fmt validate eval eval-dry clean distclean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: venv  ## Install runtime dependencies
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements.txt

install-dev: venv  ## Install runtime + dev dependencies
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements-dev.txt

test:  ## Run the test suite
	$(BIN)/python -m pytest tests/

test-fast:  ## Run only tests that need no model weights (what CI runs)
	$(BIN)/python -m pytest tests/ -m "not gpu"

lint:  ## Check formatting and lint rules
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

fmt:  ## Apply formatting and autofixable lint rules
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

validate:  ## Verify the agent chain switches, applies and transfers (small model, CPU)
	$(BIN)/python tools/validate_chain.py --model Qwen/Qwen2.5-0.5B-Instruct --device cpu

eval-dry:  ## Exercise the whole eval pipeline with no weights and no GPU
	$(BIN)/python run_eval.py --methods all --dataset sample --dry-run

eval:  ## Run the ablation ladder (MODEL/DATASET/FRACTION overridable)
	$(BIN)/python run_eval.py --methods ladder --model $(MODEL) \
	  --dataset $(DATASET) --fraction $(FRACTION) --max-new-tokens 2048 --latent-steps 50

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

distclean: clean  ## Also remove the virtualenv and every eval run
	rm -rf $(VENV) eval_runs
