"""
Shared test fixtures for LatentMAS test suite.

Loads Qwen/Qwen2.5-7B-Instruct once per session (session-scoped)
so all tests share the same model and tokenizer without reloading.

Requires: 48GB VRAM (NVIDIA A40 / A6000 / etc.)
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "/home/caches"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


@pytest.fixture(scope="session")
def model():
    """Load the full 7B model once for the entire test session."""
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    m.eval()
    return m


@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer once for the entire test session."""
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def hidden_dim(model):
    """Return the model's hidden dimension."""
    return model.config.hidden_size


@pytest.fixture(scope="session")
def device():
    """Return the compute device string."""
    return DEVICE


@pytest.fixture(scope="session")
def dtype():
    """Return the compute dtype."""
    return DTYPE
