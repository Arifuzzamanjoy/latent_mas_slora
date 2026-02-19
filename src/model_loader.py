"""
Model Loader – load model + tokenizer with proper dtype / device / quantisation.

Extracted from the original monolithic ``system.py`` so that any part of
the codebase can load a model without importing the full LatentMASSystem.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# Map string dtype names → torch dtypes
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ModelLoader:
    """Load a HuggingFace causal-LM and its tokenizer.

    Handles dtype selection, quantisation (4-bit / BNB), device placement,
    KV-cache enablement, and pad-token fixup.
    """

    # Substrings that signal a VLM checkpoint
    _VLM_INDICATORS = ("-VL-", "-VL/", "_VL_", "_VL/", "vision", "vlm")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        cache_dir: str = "/home/caches",
    ) -> Tuple:
        """Load model + tokenizer.

        Parameters
        ----------
        model_name : str
            HuggingFace Hub id or local path.
        dtype : str
            ``"bfloat16"`` | ``"float16"`` | ``"float32"`` | ``"4bit"``.
        device : str
            Target device (``"cuda"`` / ``"cpu"``).
        cache_dir : str
            Download cache directory.

        Returns
        -------
        tuple[PreTrainedModel, PreTrainedTokenizer]
        """
        os.makedirs(cache_dir, exist_ok=True)

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available – falling back to CPU")
            device = "cpu"

        is_vlm = cls._is_vlm(model_name)

        # ---- tokenizer / processor --------------------------------
        if is_vlm:
            tokenizer = cls._load_vlm_processor(model_name, cache_dir)
        else:
            tokenizer = cls._load_tokenizer(model_name, cache_dir)

        # ---- model -------------------------------------------------
        model = cls._load_model(
            model_name,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir,
            is_vlm=is_vlm,
        )

        logger.info(
            "Loaded %s  dtype=%s  device=%s  vlm=%s",
            model_name, dtype, device, is_vlm,
        )
        return model, tokenizer

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _is_vlm(cls, model_name: str) -> bool:
        lower = model_name.lower()
        return any(ind.lower() in lower for ind in cls._VLM_INDICATORS)

    @staticmethod
    def _load_tokenizer(model_name: str, cache_dir: str):
        tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    @staticmethod
    def _load_vlm_processor(model_name: str, cache_dir: str):
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        return proc.tokenizer  # return the tokenizer only

    @classmethod
    def _load_model(
        cls,
        model_name: str,
        *,
        dtype: str,
        device: str,
        cache_dir: str,
        is_vlm: bool,
    ):
        kwargs: dict = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }

        if dtype == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = _DTYPE_MAP.get(dtype, torch.float32)

        if is_vlm:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, **kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **kwargs,
            )

        # Move to device (skip for 4-bit – already placed by BNB)
        if dtype != "4bit":
            model = model.to(device)

        model.eval()

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        return model
