"""
Dependency injection for FastAPI endpoints.

This module provides dependency injection functions for accessing
shared services and resources with proper lifecycle management.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Generator
from functools import lru_cache
from pathlib import Path
import logging

from src.core.config import get_settings, Settings
from src.models.service import ModelService, ModelRegistry


logger = logging.getLogger(__name__)


# Singleton instances
_model_registry: Optional[ModelRegistry] = None
_model_service: Optional[ModelService] = None


def get_settings_dep() -> Settings:
    """
    Dependency for accessing application settings.
    
    Returns:
        Settings singleton instance.
    """
    return get_settings()


def get_model_registry() -> ModelRegistry:
    """
    Get or create the model registry singleton.
    
    Returns:
        ModelRegistry instance.
    """
    global _model_registry
    
    if _model_registry is None:
        settings = get_settings()
        _model_registry = ModelRegistry(
            model_dir=settings.model_dir
        )
        logger.info("Created ModelRegistry singleton")
    
    return _model_registry


def get_model_service() -> ModelService:
    """
    Dependency for accessing model service.
    
    Returns:
        ModelService instance.
        
    Example:
        >>> @app.get("/predict")
        ... async def predict(
        ...     model_service: ModelService = Depends(get_model_service)
        ... ):
        ...     return model_service.predict(...)
    """
    global _model_service
    
    if _model_service is None:
        registry = get_model_registry()
        _model_service = ModelService(registry=registry)
        logger.info("Created ModelService singleton")
    
    return _model_service


def get_feature_pipeline():
    """
    Dependency for accessing feature engineering pipeline.
    
    Returns:
        FeatureEngineeringPipeline instance.
    """
    from src.features.pipeline import (
        FeatureEngineeringPipeline,
        CompositeFeatureStrategy,
    )
    from src.features.business_metrics import BusinessMetricsStrategy
    from src.features.temporal_features import TemporalFeaturesStrategy
    from src.features.strategic_groupings import StrategicGroupingsStrategy
    
    strategy = CompositeFeatureStrategy([
        BusinessMetricsStrategy(),
        TemporalFeaturesStrategy(),
        StrategicGroupingsStrategy(),
    ])
    
    return FeatureEngineeringPipeline(strategy=strategy)


async def startup_event() -> None:
    """
    Application startup event handler.
    
    Initializes services and loads models.
    """
    logger.info("Application starting up...")
    
    # Initialize model registry
    get_model_registry()
    
    # Initialize model service
    get_model_service()
    
    logger.info("Application startup complete")


async def shutdown_event() -> None:
    """
    Application shutdown event handler.
    
    Cleans up resources and saves state.
    """
    global _model_registry, _model_service
    
    logger.info("Application shutting down...")
    
    if _model_registry is not None:
        _model_registry.clear_cache()
    
    _model_registry = None
    _model_service = None
    
    logger.info("Application shutdown complete")


def reset_singletons() -> None:
    """
    Reset singleton instances.
    
    Useful for testing to ensure clean state.
    """
    global _model_registry, _model_service
    
    _model_registry = None
    _model_service = None
    
    logger.info("Singletons reset")

