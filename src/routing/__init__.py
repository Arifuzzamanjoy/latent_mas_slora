"""
Routing Module - Fast Domain/Pipeline Selection

Provides keyword-based routing for automatic LoRA and pipeline selection
based on prompt analysis.

Includes:
- Domain: Supported domain enum
- FastRouter: Ultra-fast keyword-based routing
"""

from .fast_router import Domain, FastRouter, FastRoutingResult

__all__ = [
    "Domain",
    "FastRouter",
    "FastRoutingResult",
]
