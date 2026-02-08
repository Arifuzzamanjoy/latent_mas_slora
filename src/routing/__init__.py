"""
Routing Module - Intelligent Domain/Pipeline Selection

Provides semantic routing for automatic LoRA and pipeline selection
based on prompt analysis.

Includes:
- SemanticRouter: Basic embedding + keyword routing
- AdvancedHybridRouter: SOTA dual-signal routing with MLP support
"""

from .domain_profiles import Domain, DomainProfile, DOMAIN_PROFILES
from .semantic_router import SemanticRouter, get_router, auto_route
from .advanced_router import (
    AdvancedHybridRouter,
    RoutingResult,
    TaskMetaFeatureExtractor,
    MLPRouter,
    get_advanced_router,
    advanced_route,
)

__all__ = [
    # Domain profiles
    "Domain",
    "DomainProfile", 
    "DOMAIN_PROFILES",
    # Basic router
    "SemanticRouter",
    "get_router",
    "auto_route",
    # Advanced router (SOTA)
    "AdvancedHybridRouter",
    "RoutingResult",
    "TaskMetaFeatureExtractor",
    "MLPRouter",
    "get_advanced_router",
    "advanced_route",
]
