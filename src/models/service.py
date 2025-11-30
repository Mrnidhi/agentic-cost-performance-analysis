"""
Model service layer with instrumentation and dependency injection.

This module provides production-grade model serving with:
- Prometheus metrics integration
- Structured logging
- Model versioning and registry
- Graceful error handling

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Any, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
import time
import logging
import pickle

import numpy as np

from src.core.config import ModelConfig, get_model_config
from src.core.exceptions import (
    ModelLoadingError,
    PredictionError,
)


logger = logging.getLogger(__name__)


# Type variable for generic model wrapper
ModelT = TypeVar("ModelT")


# Metrics (using simple counters for compatibility without prometheus)
class SimpleMetrics:
    """Simple metrics collector when prometheus is not available."""
    
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}
    
    def inc_counter(self, name: str, labels: dict[str, str]) -> None:
        key = f"{name}_{labels}"
        self._counters[key] = self._counters.get(key, 0) + 1
    
    def observe_histogram(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
    
    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value


# Global metrics instance
_metrics = SimpleMetrics()


try:
    from prometheus_client import Counter, Histogram, Gauge
    
    PREDICTION_COUNTER = Counter(
        "model_predictions_total",
        "Total number of predictions",
        ["model_name", "status"]
    )
    PREDICTION_DURATION = Histogram(
        "prediction_duration_seconds",
        "Prediction latency in seconds",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    ACTIVE_MODELS = Gauge(
        "active_models_count",
        "Number of currently loaded models"
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, using simple metrics")


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable metadata for a trained model."""
    
    name: str
    version: str
    created_at: datetime
    feature_names: tuple[str, ...]
    target_name: str
    model_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    
    @property
    def model_id(self) -> str:
        """Unique identifier for this model version."""
        return f"{self.name}_v{self.version}"


class InstrumentedModel(Generic[ModelT]):
    """
    Wrapper that adds instrumentation to any ML model.
    
    Provides:
    - Prediction timing metrics
    - Success/failure counters
    - Structured logging
    - Error handling with context
    
    Attributes:
        model: The underlying ML model.
        metadata: Model metadata.
        
    Example:
        >>> xgb_model = xgboost.XGBRegressor()
        >>> instrumented = InstrumentedModel(
        ...     model=xgb_model,
        ...     metadata=ModelMetadata(name="performance_predictor", ...)
        ... )
        >>> predictions = instrumented.predict(features)
    """
    
    def __init__(
        self,
        model: ModelT,
        metadata: ModelMetadata,
        timeout_seconds: float = 30.0
    ) -> None:
        """
        Initialize instrumented model.
        
        Args:
            model: The underlying ML model.
            metadata: Model metadata.
            timeout_seconds: Prediction timeout in seconds.
        """
        self._model = model
        self._metadata = metadata
        self._timeout = timeout_seconds
        self._prediction_count = 0
        self._error_count = 0
        
        logger.info(
            "Initialized InstrumentedModel",
            extra={
                "model_name": metadata.name,
                "model_version": metadata.version,
                "model_type": metadata.model_type
            }
        )
    
    @property
    def model(self) -> ModelT:
        """Access the underlying model."""
        return self._model
    
    @property
    def metadata(self) -> ModelMetadata:
        """Access model metadata."""
        return self._metadata
    
    @property
    def prediction_count(self) -> int:
        """Total number of predictions made."""
        return self._prediction_count
    
    @property
    def error_count(self) -> int:
        """Total number of prediction errors."""
        return self._error_count
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions with instrumentation.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predictions array.
            
        Raises:
            PredictionError: If prediction fails.
        """
        start_time = time.time()
        
        with self._prediction_context():
            try:
                # Validate input shape
                self._validate_input(features)
                
                # Make prediction
                predictions = self._model.predict(features)
                
                # Record success metrics
                duration = time.time() - start_time
                self._record_success(duration, features.shape)
                
                return predictions
                
            except PredictionError:
                raise
            except Exception as e:
                self._record_failure(str(e))
                raise PredictionError(
                    message=f"Prediction failed: {e}",
                    model_name=self._metadata.name,
                    input_shape=features.shape,
                    cause=e
                ) from e
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Make probability predictions with instrumentation.
        
        Args:
            features: Feature matrix.
            
        Returns:
            Probability predictions.
            
        Raises:
            PredictionError: If prediction fails or model doesn't support proba.
        """
        if not hasattr(self._model, "predict_proba"):
            raise PredictionError(
                message="Model does not support probability predictions",
                model_name=self._metadata.name
            )
        
        start_time = time.time()
        
        with self._prediction_context():
            try:
                self._validate_input(features)
                predictions = self._model.predict_proba(features)
                
                duration = time.time() - start_time
                self._record_success(duration, features.shape)
                
                return predictions
                
            except PredictionError:
                raise
            except Exception as e:
                self._record_failure(str(e))
                raise PredictionError(
                    message=f"Probability prediction failed: {e}",
                    model_name=self._metadata.name,
                    input_shape=features.shape,
                    cause=e
                ) from e
    
    @contextmanager
    def _prediction_context(self):
        """Context manager for prediction with logging."""
        logger.debug(
            "Starting prediction",
            extra={"model_name": self._metadata.name}
        )
        try:
            yield
        finally:
            logger.debug(
                "Prediction completed",
                extra={"model_name": self._metadata.name}
            )
    
    def _validate_input(self, features: np.ndarray) -> None:
        """Validate input features."""
        if features.ndim != 2:
            raise PredictionError(
                message=f"Expected 2D array, got {features.ndim}D",
                model_name=self._metadata.name,
                input_shape=features.shape
            )
        
        expected_features = len(self._metadata.feature_names)
        if features.shape[1] != expected_features:
            raise PredictionError(
                message=f"Expected {expected_features} features, got {features.shape[1]}",
                model_name=self._metadata.name,
                input_shape=features.shape
            )
    
    def _record_success(self, duration: float, input_shape: tuple) -> None:
        """Record successful prediction metrics."""
        self._prediction_count += 1
        
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNTER.labels(
                model_name=self._metadata.name,
                status="success"
            ).inc()
            PREDICTION_DURATION.observe(duration)
        else:
            _metrics.inc_counter(
                "predictions",
                {"model": self._metadata.name, "status": "success"}
            )
            _metrics.observe_histogram("prediction_duration", duration)
        
        logger.info(
            "Prediction successful",
            extra={
                "model_name": self._metadata.name,
                "duration_seconds": round(duration, 4),
                "input_shape": input_shape,
                "prediction_count": self._prediction_count
            }
        )
    
    def _record_failure(self, error: str) -> None:
        """Record failed prediction metrics."""
        self._error_count += 1
        
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNTER.labels(
                model_name=self._metadata.name,
                status="error"
            ).inc()
        else:
            _metrics.inc_counter(
                "predictions",
                {"model": self._metadata.name, "status": "error"}
            )
        
        logger.error(
            "Prediction failed",
            extra={
                "model_name": self._metadata.name,
                "error": error,
                "error_count": self._error_count
            }
        )


class ModelRegistry:
    """
    Registry for managing model versions and lifecycle.
    
    Provides:
    - Model registration and versioning
    - Model loading and caching
    - Model discovery and listing
    
    Example:
        >>> registry = ModelRegistry(model_dir=Path("models"))
        >>> registry.register(model, metadata)
        >>> loaded = registry.load("performance_predictor", version="latest")
    """
    
    def __init__(
        self,
        model_dir: Path,
        config: Optional[ModelConfig] = None
    ) -> None:
        """
        Initialize model registry.
        
        Args:
            model_dir: Directory for model storage.
            config: Optional model configuration.
        """
        self._model_dir = Path(model_dir)
        self._config = config or get_model_config()
        self._cache: dict[str, InstrumentedModel] = {}
        
        # Ensure model directory exists
        self._model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Initialized ModelRegistry",
            extra={"model_dir": str(self._model_dir)}
        )
    
    def register(
        self,
        model: Any,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> Path:
        """
        Register a model in the registry.
        
        Args:
            model: The trained model to register.
            metadata: Model metadata.
            overwrite: Whether to overwrite existing version.
            
        Returns:
            Path to saved model file.
            
        Raises:
            ValueError: If model version exists and overwrite is False.
        """
        model_path = self._get_model_path(metadata.name, metadata.version)
        
        if model_path.exists() and not overwrite:
            raise ValueError(
                f"Model {metadata.model_id} already exists. "
                "Set overwrite=True to replace."
            )
        
        # Create model directory
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata}, f)
        
        # Update latest symlink
        latest_path = self._get_model_path(metadata.name, "latest")
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(model_path.name)
        
        logger.info(
            "Model registered",
            extra={
                "model_name": metadata.name,
                "model_version": metadata.version,
                "path": str(model_path)
            }
        )
        
        return model_path
    
    def load(
        self,
        name: str,
        version: str = "latest"
    ) -> InstrumentedModel:
        """
        Load a model from the registry.
        
        Args:
            name: Model name.
            version: Model version or "latest".
            
        Returns:
            Instrumented model wrapper.
            
        Raises:
            ModelLoadingError: If model cannot be loaded.
        """
        cache_key = f"{name}_{version}"
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Returning cached model: {cache_key}")
            return self._cache[cache_key]
        
        model_path = self._get_model_path(name, version)
        
        if not model_path.exists():
            raise ModelLoadingError(
                message=f"Model not found: {name} v{version}",
                model_path=str(model_path)
            )
        
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            
            instrumented = InstrumentedModel(
                model=data["model"],
                metadata=data["metadata"],
                timeout_seconds=self._config.prediction_timeout_seconds
            )
            
            # Cache the model
            if len(self._cache) < self._config.model_cache_size:
                self._cache[cache_key] = instrumented
            
            if PROMETHEUS_AVAILABLE:
                ACTIVE_MODELS.set(len(self._cache))
            
            logger.info(
                "Model loaded",
                extra={
                    "model_name": name,
                    "model_version": version,
                    "path": str(model_path)
                }
            )
            
            return instrumented
            
        except Exception as e:
            raise ModelLoadingError(
                message=f"Failed to load model: {e}",
                model_path=str(model_path)
            ) from e
    
    def list_models(self) -> list[str]:
        """List all registered model names."""
        models = []
        for path in self._model_dir.iterdir():
            if path.is_dir():
                models.append(path.name)
        return sorted(models)
    
    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        model_dir = self._model_dir / name
        if not model_dir.exists():
            return []
        
        versions = []
        for path in model_dir.iterdir():
            if path.suffix == ".pkl" and not path.is_symlink():
                versions.append(path.stem)
        return sorted(versions)
    
    def _get_model_path(self, name: str, version: str) -> Path:
        """Get path for a model version."""
        return self._model_dir / name / f"{version}.pkl"
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()
        if PROMETHEUS_AVAILABLE:
            ACTIVE_MODELS.set(0)
        logger.info("Model cache cleared")


class ModelService:
    """
    High-level service for model operations with dependency injection.
    
    Coordinates model loading, prediction, and lifecycle management
    with proper error handling and observability.
    
    Example:
        >>> service = ModelService(registry=ModelRegistry(...))
        >>> result = service.predict(
        ...     model_name="performance_predictor",
        ...     features=feature_array
        ... )
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        config: Optional[ModelConfig] = None
    ) -> None:
        """
        Initialize model service.
        
        Args:
            registry: Model registry for loading models.
            config: Optional model configuration.
        """
        self._registry = registry
        self._config = config or get_model_config()
        
        logger.info("Initialized ModelService")
    
    def predict(
        self,
        model_name: str,
        features: np.ndarray,
        version: str = "latest"
    ) -> np.ndarray:
        """
        Make predictions using a registered model.
        
        Args:
            model_name: Name of the model to use.
            features: Feature matrix.
            version: Model version to use.
            
        Returns:
            Predictions array.
            
        Raises:
            ModelLoadingError: If model cannot be loaded.
            PredictionError: If prediction fails.
        """
        model = self._registry.load(model_name, version)
        return model.predict(features)
    
    def predict_with_metadata(
        self,
        model_name: str,
        features: np.ndarray,
        version: str = "latest"
    ) -> dict[str, Any]:
        """
        Make predictions and return with metadata.
        
        Args:
            model_name: Name of the model to use.
            features: Feature matrix.
            version: Model version to use.
            
        Returns:
            Dictionary with predictions and metadata.
        """
        model = self._registry.load(model_name, version)
        predictions = model.predict(features)
        
        return {
            "predictions": predictions.tolist(),
            "model_name": model.metadata.name,
            "model_version": model.metadata.version,
            "model_type": model.metadata.model_type,
            "feature_names": list(model.metadata.feature_names),
            "prediction_count": model.prediction_count,
        }
    
    def get_model_info(self, model_name: str, version: str = "latest") -> dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model.
            version: Model version.
            
        Returns:
            Dictionary with model information.
        """
        model = self._registry.load(model_name, version)
        metadata = model.metadata
        
        return {
            "name": metadata.name,
            "version": metadata.version,
            "model_id": metadata.model_id,
            "model_type": metadata.model_type,
            "target_name": metadata.target_name,
            "feature_names": list(metadata.feature_names),
            "metrics": metadata.metrics,
            "created_at": metadata.created_at.isoformat(),
        }
    
    def health_check(self) -> dict[str, Any]:
        """
        Check health of model service.
        
        Returns:
            Health status dictionary.
        """
        models = self._registry.list_models()
        
        return {
            "status": "healthy",
            "registered_models": len(models),
            "model_names": models,
        }

