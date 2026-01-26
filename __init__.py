"""
LatentMAS + S-LoRA Package

Import from here for the main API:
    from latent_mas_slora import LatentMASSystem, AgentConfig
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src import (
    LatentMASSystem,
    AgentConfig,
    AgentRole,
    LatentMemory,
    LoRAAdapterManager,
)

__version__ = "0.1.0"
__all__ = [
    "LatentMASSystem",
    "AgentConfig",
    "AgentRole", 
    "LatentMemory",
    "LoRAAdapterManager",
]
