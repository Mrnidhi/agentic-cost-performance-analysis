"""
Models module for AI Agent Performance Intelligence System.

This module provides ML model services with dependency injection,
instrumentation, and production-grade error handling.

Course: DATA 230 (Data Visualization) at SJSU
"""

from src.models.ensemble import (
    PerformanceOptimizationEngine,
    FailurePredictionSystem,
    AgentRecommendationEngine,
)
from src.models.optimization import (
    CostPerformanceTradeoffAnalyzer,
    MultiObjectiveOptimizer,
    ParetoFrontierExtractor,
)
from src.models.service import (
    ModelService,
    InstrumentedModel,
    ModelRegistry,
)
from src.models.explainer import (
    SHAPExplainer,
    FeatureImportanceAnalyzer,
)

__all__ = [
    # Ensemble models
    "PerformanceOptimizationEngine",
    "FailurePredictionSystem",
    "AgentRecommendationEngine",
    # Optimization
    "CostPerformanceTradeoffAnalyzer",
    "MultiObjectiveOptimizer",
    "ParetoFrontierExtractor",
    # Service
    "ModelService",
    "InstrumentedModel",
    "ModelRegistry",
    # Explainability
    "SHAPExplainer",
    "FeatureImportanceAnalyzer",
]
