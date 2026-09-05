"""
Routing Module - Intelligent Domain/Pipeline Selection

Provides semantic routing for automatic LoRA and pipeline selection
based on prompt analysis.
"""

from .domain_profiles import DOMAIN_PROFILES, Domain, DomainProfile
from .semantic_router import SemanticRouter, auto_route, get_router

__all__ = [
    "Domain",
    "DomainProfile",
    "DOMAIN_PROFILES",
    "SemanticRouter",
    "get_router",
    "auto_route",
]
