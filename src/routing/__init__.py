"""
Routing Module - Intelligent Domain/Pipeline Selection

Provides semantic routing for automatic LoRA and pipeline selection
based on prompt analysis.
"""

from .domain_profiles import Domain, DomainProfile, DOMAIN_PROFILES
from .semantic_router import SemanticRouter, get_router, auto_route

__all__ = [
    "Domain",
    "DomainProfile", 
    "DOMAIN_PROFILES",
    "SemanticRouter",
    "get_router",
    "auto_route",
]
