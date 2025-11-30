"""
API module for AI Agent Performance Intelligence System.

This module provides the FastAPI application with production-grade
endpoints, validation, and observability.

Course: DATA 230 (Data Visualization) at SJSU
"""

from src.api.main import app, create_app
from src.api.schemas import (
    OptimizationRequest,
    OptimizationResponse,
    AgentConfig,
    TradeoffOption,
    HealthResponse,
)
from src.api.dependencies import (
    get_model_service,
    get_feature_pipeline,
)

__all__ = [
    "app",
    "create_app",
    "OptimizationRequest",
    "OptimizationResponse",
    "AgentConfig",
    "TradeoffOption",
    "HealthResponse",
    "get_model_service",
    "get_feature_pipeline",
]
