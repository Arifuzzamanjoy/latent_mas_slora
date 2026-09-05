"""
LoRA Adapter Manager - Scalable Multi-Adapter Serving

Implements S-LoRA patterns for efficient adapter management:
- Dynamic loading from HuggingFace Hub
- Adapter merging and combination
- Memory-efficient switching

Optimized for 24-48GB VRAM with 10-20+ concurrent adapters.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ExternalLoRAInfo:
    """Information about an external LoRA adapter"""

    name: str
    hf_path: str
    description: str = ""
    domain: str = "general"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    rank: Optional[int] = None
    verified: bool = False


# Registry of external LoRAs for Qwen2.5-7B.
#
# `verified` means the repository was downloaded and its lora_B tensors checked
# to be non-zero. This matters because a LoRA whose B matrices are all zero is an
# exact identity function: it loads, reports millions of parameters, switches
# without error, and changes nothing. Two entries here are in that state
# upstream, so they are marked verified=False and left in place as a warning
# rather than silently offered as working adapters.
#
# Last checked: 2026-09-05
QWEN25_LORA_REGISTRY = {
    # Reasoning - the only entry both trained and built for this base model
    "reasoning_lora": ExternalLoRAInfo(
        name="reasoning_lora",
        hf_path="PandurangMopgar/qwen-2.5-7b-reasoning-lora",
        description="General reasoning (7B). Trained: ||lora_B|| = 6.83.",
        domain="reasoning",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        verified=True,
    ),
    # Math - trained, but built for a different base model. Applying it to
    # Qwen2.5-7B-Instruct is off-base and may only add noise; measure before use.
    "math_instruct": ExternalLoRAInfo(
        name="math_instruct",
        hf_path="SKNahin/Qwen2.5-Math-7B-Instruct-bnb-4bit-lora",
        description="Mathematical reasoning. Trained (||lora_B|| = 7.20) but for "
        "Qwen2.5-Math-7B, not Qwen2.5-7B-Instruct.",
        domain="math",
        base_model="Qwen/Qwen2.5-Math-7B-Instruct",
        verified=False,
    ),
    # Medical - the upstream repository ships zero lora_B weights, so loading it
    # is a no-op. Kept so the name resolves to an explanation instead of silently
    # doing nothing.
    "medical_reasoner": ExternalLoRAInfo(
        name="medical_reasoner",
        hf_path="zjudai/flowertune-medical-lora-qwen2.5-7b-instruct",
        description="Medical (7B). UNTRAINED upstream: ||lora_B|| = 0.00, so this "
        "adapter is an identity function and changes nothing.",
        domain="medical",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        verified=False,
    ),
}

# Removed 2026-09-05: "coder_7b" -> Alexis-Az/Qwen-2.5-Coder-7B-Instruct-LoRA
#   returns HTTP 404, the repository no longer exists.
# Removed 2026-09-05: "medical_instruct" -> duplicate of medical_reasoner
#   (identical hf_path), so it offered two names for one untrained adapter.


class LoRAAdapterManager:
    """
    Manager for LoRA adapter lifecycle.

    Features:
    - Load adapters from HuggingFace Hub
    - Dynamic adapter switching (S-LoRA style)
    - Adapter merging for combined capabilities
    - Memory tracking and optimization

    Memory Budget (24-48GB VRAM):
    - Base model (Qwen2.5-7B BF16): ~14GB
    - Per adapter (rank 32): ~80MB
    - Can load 10-20+ adapters with room to spare
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        cache_dir: str = "/home/caches",
        max_loaded_adapters: int = 20,
    ):
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.max_loaded_adapters = max_loaded_adapters

        os.makedirs(cache_dir, exist_ok=True)

        self._loaded_adapters: Dict[str, Dict[str, Any]] = {}
        self._adapter_usage: Dict[str, int] = {}
        self._lock = threading.RLock()

        # Track memory
        self._initial_memory = self._get_gpu_memory()

    def _get_gpu_memory(self) -> int:
        """Get current GPU memory usage in bytes (0 when not on a CUDA device).

        cuda.is_available() alone is not enough: the host can have a GPU while
        this system is configured for CPU, and memory_allocated("cpu") raises.
        """
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            return torch.cuda.memory_allocated(self.device)
        return 0

    def load_external_lora(
        self,
        name: str,
        hf_path: str,
        adapter_name: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Load a LoRA adapter from HuggingFace Hub.

        Args:
            name: Local name for the adapter
            hf_path: HuggingFace Hub path (e.g., "user/model-lora")
            adapter_name: Name to use in PEFT (defaults to name)
            **kwargs: Additional arguments for PeftModel.from_pretrained

        Returns:
            True if loaded successfully
        """
        adapter_name = adapter_name or name

        with self._lock:
            if adapter_name in self._loaded_adapters:
                print(f"[INFO] Adapter '{adapter_name}' already loaded")
                return True

            # Check if we need to unload adapters
            if len(self._loaded_adapters) >= self.max_loaded_adapters:
                self._unload_least_used()

            try:
                print(f"[INFO] Loading LoRA from {hf_path}...")

                # Load adapter
                self.model.load_adapter(
                    hf_path,
                    adapter_name=adapter_name,
                    cache_dir=self.cache_dir,
                    **kwargs,
                )

                memory_used = self._get_gpu_memory() - self._initial_memory

                self._loaded_adapters[adapter_name] = {
                    "name": name,
                    "hf_path": hf_path,
                    "memory_mb": memory_used / 1024 / 1024,
                }
                self._adapter_usage[adapter_name] = 0

                print(f"[INFO] Loaded '{adapter_name}' ({memory_used / 1024 / 1024:.1f} MB)")
                return True

            except Exception as e:
                print(f"[ERROR] Failed to load adapter from {hf_path}: {e}")
                return False

    def load_from_registry(self, registry_name: str) -> bool:
        """Load adapter from the built-in registry"""
        if registry_name not in QWEN25_LORA_REGISTRY:
            print(
                f"[ERROR] '{registry_name}' not in registry. Available: {list(QWEN25_LORA_REGISTRY.keys())}"
            )
            return False

        info = QWEN25_LORA_REGISTRY[registry_name]
        return self.load_external_lora(info.name, info.hf_path)

    def _unload_least_used(self) -> None:
        """Unload the least recently used adapter"""
        if not self._adapter_usage:
            return

        # Find least used
        least_used = min(self._adapter_usage.items(), key=lambda x: x[1])
        adapter_name = least_used[0]

        self.unload_adapter(adapter_name)

    def unload_adapter(self, adapter_name: str) -> None:
        """Unload an adapter to free memory"""
        with self._lock:
            if adapter_name not in self._loaded_adapters:
                return

            if hasattr(self.model, "delete_adapter"):
                self.model.delete_adapter(adapter_name)

            del self._loaded_adapters[adapter_name]
            del self._adapter_usage[adapter_name]

            # Force garbage collection
            torch.cuda.empty_cache()

            print(f"[INFO] Unloaded adapter: {adapter_name}")

    def switch_adapter(self, adapter_name: str) -> None:
        """Switch to a specific adapter"""
        with self._lock:
            if adapter_name not in self._loaded_adapters:
                raise ValueError(f"Adapter '{adapter_name}' not loaded")

            self.model.set_adapter(adapter_name)
            self._adapter_usage[adapter_name] += 1

    def merge_adapters(
        self,
        adapter_names: List[str],
        weights: Optional[List[float]] = None,
        new_adapter_name: str = "merged",
    ) -> None:
        """
        Merge multiple adapters into one.

        Args:
            adapter_names: Names of adapters to merge
            weights: Weights for each adapter (default: equal)
            new_adapter_name: Name for the merged adapter
        """
        if weights is None:
            weights = [1.0 / len(adapter_names)] * len(adapter_names)

        if len(weights) != len(adapter_names):
            raise ValueError("Number of weights must match number of adapters")

        # Use PEFT's add_weighted_adapter
        self.model.add_weighted_adapter(
            adapters=adapter_names,
            weights=weights,
            adapter_name=new_adapter_name,
            combination_type="linear",
        )

        self._loaded_adapters[new_adapter_name] = {
            "name": new_adapter_name,
            "merged_from": adapter_names,
            "weights": weights,
        }
        self._adapter_usage[new_adapter_name] = 0

        print(f"[INFO] Created merged adapter '{new_adapter_name}' from {adapter_names}")

    def list_loaded(self) -> List[str]:
        """List all loaded adapter names"""
        return list(self._loaded_adapters.keys())

    def list_registry(self) -> Dict[str, ExternalLoRAInfo]:
        """List available adapters in registry"""
        return dict(QWEN25_LORA_REGISTRY)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        current_memory = self._get_gpu_memory()

        return {
            "initial_memory_mb": self._initial_memory / 1024 / 1024,
            "current_memory_mb": current_memory / 1024 / 1024,
            "adapter_memory_mb": (current_memory - self._initial_memory) / 1024 / 1024,
            "num_loaded_adapters": len(self._loaded_adapters),
            "adapters": {
                name: info.get("memory_mb", "unknown")
                for name, info in self._loaded_adapters.items()
            },
        }


class AdapterRouter:
    """
    Dynamic adapter routing based on input.

    Implements semantic routing to select the best adapter
    for a given query.
    """

    def __init__(
        self,
        adapter_manager: LoRAAdapterManager,
        routing_strategy: str = "keyword",
    ):
        self.manager = adapter_manager
        self.strategy = routing_strategy

        # Keyword-based routing rules
        self._keyword_rules = {
            "medical": ["patient", "diagnosis", "symptom", "treatment", "disease", "clinical"],
            "math": ["calculate", "equation", "solve", "number", "mathematical", "formula"],
            "code": ["function", "code", "program", "algorithm", "implement", "debug"],
            "reasoning": ["reason", "logic", "analyze", "deduce", "infer", "conclude"],
        }

    def route(self, query: str, available_adapters: List[str]) -> str:
        """
        Route query to appropriate adapter.

        Args:
            query: Input query
            available_adapters: List of available adapter names

        Returns:
            Selected adapter name
        """
        if self.strategy == "keyword":
            return self._keyword_route(query, available_adapters)
        else:
            # Default to first available
            return available_adapters[0] if available_adapters else None

    def _keyword_route(self, query: str, available_adapters: List[str]) -> str:
        """Route based on keyword matching"""
        query_lower = query.lower()

        scores = {}
        for domain, keywords in self._keyword_rules.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                # Find adapter matching this domain
                for adapter in available_adapters:
                    if domain in adapter.lower():
                        scores[adapter] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        # Default to first available
        return available_adapters[0] if available_adapters else None
