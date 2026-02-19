"""
LoRA Adapter Manager - Scalable Multi-Adapter Serving

Implements S-LoRA patterns for efficient adapter management:
- Dynamic loading from HuggingFace Hub
- Adapter merging and combination
- Memory-efficient switching

Optimized for 24-48GB VRAM with 10-20+ concurrent adapters.
"""

import logging
import os
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading

from src.core.embedding_guard import EmbeddingGuard

logger = logging.getLogger(__name__)


@dataclass
class ExternalLoRAInfo:
    """Information about an external LoRA adapter"""
    name: str
    hf_path: str
    description: str = ""
    domain: str = "general"
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    rank: Optional[int] = None
    verified: bool = False


# Registry of known open-source LoRAs for Qwen2.5-VL-7B
QWEN25_LORA_REGISTRY = {
    # Medical Domain (VL-7B)
    "medical_vl": ExternalLoRAInfo(
        name="medical_vl",
        hf_path="sarathi-balakrishnan/Qwen2.5-VL-7B-Medical-LoRA",
        description="Medical visual reasoning and diagnosis",
        domain="medical",
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    ),

    # Visual Understanding
    "reward_vl": ExternalLoRAInfo(
        name="reward_vl",
        hf_path="DJ-Kim/Qwen2.5_VL_7B_Reward_LoRA_VLFeedBack_144000step",
        description="Vision-language reward model for improved responses",
        domain="general",
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    ),

    # Document / OCR
    "comics_vl": ExternalLoRAInfo(
        name="comics_vl",
        hf_path="VLR-CVC/Qwen2.5-VL-7B-Instruct-lora-ComicsPAP",
        description="Comics panel analysis and understanding",
        domain="general",
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    ),

    # Point Detection & Counting
    "point_detect_vl": ExternalLoRAInfo(
        name="point_detect_vl",
        hf_path="SimulaMet/PointDetectCount-Qwen2.5-VL-7B-LoRA",
        description="Point detection and counting in images",
        domain="general",
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    ),
}


class LoRAAdapterManager:
    """
    Manager for LoRA adapter lifecycle.
    
    Features:
    - Load adapters from HuggingFace Hub
    - Dynamic adapter switching (S-LoRA style)
    - Adapter merging for combined capabilities
    - Memory tracking and optimization
    
    Memory Budget (48GB VRAM):
    - Base model (Qwen2.5-VL-7B BF16): ~16GB
    - Per adapter (rank 32): ~50MB
    - Can load 20+ adapters with room to spare
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
        
        # Embedding safety guard
        self._guard = EmbeddingGuard(model)
    
    def _get_gpu_memory(self) -> int:
        """Get current GPU memory usage in bytes"""
        if torch.cuda.is_available():
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
                logger.info("Adapter '%s' already loaded", adapter_name)
                return True
            
            # Check if we need to unload adapters
            if len(self._loaded_adapters) >= self.max_loaded_adapters:
                self._unload_least_used()
            
            try:
                from peft import PeftModel
                
                logger.info("Loading LoRA from %s...", hf_path)
                
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
                
                logger.info("Loaded '%s' (%.1f MB)", adapter_name, memory_used / 1024 / 1024)
                
                # Verify embeddings were not modified
                self._guard.assert_frozen()
                
                return True
                
            except Exception as e:
                logger.error("Failed to load adapter from %s: %s", hf_path, e)
                return False
    
    def load_from_registry(self, registry_name: str) -> bool:
        """Load adapter from the built-in registry"""
        if registry_name not in QWEN25_LORA_REGISTRY:
            logger.error("'%s' not in registry. Available: %s", registry_name, list(QWEN25_LORA_REGISTRY.keys()))
            return False
        
        info = QWEN25_LORA_REGISTRY[registry_name]
        return self.load_external_lora(info.name, info.hf_path)
    
    def is_loaded(self, adapter_name: str) -> bool:
        """Check if an adapter is currently loaded"""
        return adapter_name in self._loaded_adapters
    
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
            
            if hasattr(self.model, 'delete_adapter'):
                self.model.delete_adapter(adapter_name)
            
            del self._loaded_adapters[adapter_name]
            del self._adapter_usage[adapter_name]
            
            # Force garbage collection
            torch.cuda.empty_cache()
            
            logger.info("Unloaded adapter: %s", adapter_name)
    
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
        
        logger.info("Created merged adapter '%s' from %s", new_adapter_name, adapter_names)
    
    def list_loaded(self) -> List[str]:
        """List all loaded adapter names"""
        return list(self._loaded_adapters.keys())
    
    @property
    def loaded_adapters(self) -> Dict[str, Dict[str, Any]]:
        """Access loaded adapters (read-only view)"""
        return dict(self._loaded_adapters)
    
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

    # Forbidden modules that should never be LoRA targets
    FORBIDDEN_TARGET_MODULES = {"embed_tokens", "lm_head"}

    def get_adapter_target_modules(self, name: str) -> List[str]:
        """
        Check what target_modules an external adapter uses.

        Inspects the loaded PEFT config for the given adapter and
        logs a WARNING if it includes 'embed_tokens' or 'lm_head'.

        Args:
            name: Adapter name (must already be loaded).

        Returns:
            List of target module names, or empty list if unavailable.
        """
        try:
            if hasattr(self.model, "peft_config") and name in self.model.peft_config:
                cfg = self.model.peft_config[name]
                modules = list(getattr(cfg, "target_modules", []) or [])
            else:
                logger.warning("No PEFT config found for adapter '%s'", name)
                return []

            dangerous = self.FORBIDDEN_TARGET_MODULES.intersection(modules)
            if dangerous:
                logger.warning(
                    "Adapter '%s' targets FORBIDDEN modules %s — "
                    "this can corrupt shared embeddings!",
                    name,
                    sorted(dangerous),
                )
            return modules

        except Exception as e:
            logger.error("Failed to inspect target_modules for '%s': %s", name, e)
            return []


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
