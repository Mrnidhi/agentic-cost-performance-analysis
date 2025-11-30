"""
Feature engineering pipeline with composite strategy support.

This module provides a pipeline for orchestrating multiple feature
engineering strategies with caching, validation, and error handling.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Sequence
import logging
import time
from functools import lru_cache
from contextlib import contextmanager

import pandas as pd

from src.core.protocols import FeatureEngineeringStrategy
from src.core.config import FeatureConfig, get_feature_config
from src.core.exceptions import FeatureEngineeringError


logger = logging.getLogger(__name__)


class CompositeFeatureStrategy:
    """
    Composite strategy that combines multiple feature engineering strategies.
    
    Implements the Composite pattern to apply multiple strategies in sequence,
    with validation and error handling at each step.
    
    Attributes:
        strategies: Sequence of strategies to apply.
        
    Example:
        >>> composite = CompositeFeatureStrategy([
        ...     BusinessMetricsStrategy(),
        ...     TemporalFeaturesStrategy(),
        ...     StrategicGroupingsStrategy()
        ... ])
        >>> result = composite.engineer_features(raw_df)
    """
    
    def __init__(
        self,
        strategies: Sequence[FeatureEngineeringStrategy]
    ) -> None:
        """
        Initialize composite strategy.
        
        Args:
            strategies: Sequence of strategies to apply in order.
            
        Raises:
            ValueError: If no strategies provided.
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        
        self._strategies = list(strategies)
        self._feature_names: list[str] = []
        
        for strategy in self._strategies:
            self._feature_names.extend(strategy.feature_names)
        
        logger.info(
            "Initialized CompositeFeatureStrategy",
            extra={
                "num_strategies": len(self._strategies),
                "total_features": len(self._feature_names)
            }
        )
    
    @property
    def feature_names(self) -> list[str]:
        """Return list of all feature names produced by all strategies."""
        return self._feature_names.copy()
    
    @property
    def strategies(self) -> list[FeatureEngineeringStrategy]:
        """Return list of strategies."""
        return self._strategies.copy()
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input against all strategies.
        
        Args:
            data: Input DataFrame to validate.
            
        Returns:
            True if valid for all strategies.
            
        Raises:
            FeatureEngineeringError: If validation fails for any strategy.
        """
        for strategy in self._strategies:
            strategy.validate_input(data)
        return True
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all strategies in sequence.
        
        Args:
            data: Raw input data.
            
        Returns:
            DataFrame with all engineered features.
            
        Raises:
            FeatureEngineeringError: If any strategy fails.
        """
        result = data.copy()
        
        for i, strategy in enumerate(self._strategies):
            strategy_name = strategy.__class__.__name__
            
            with _strategy_context(strategy_name):
                logger.debug(f"Applying strategy {i+1}/{len(self._strategies)}: {strategy_name}")
                result = strategy.engineer_features(result)
        
        logger.info(
            "Composite feature engineering completed",
            extra={"total_features_added": len(self._feature_names)}
        )
        
        return result


class FeatureEngineeringPipeline:
    """
    Production-grade feature engineering pipeline.
    
    Provides orchestration of feature engineering with:
    - Caching for repeated computations
    - Chunked processing for large datasets
    - Comprehensive validation and error handling
    - Performance monitoring
    
    Attributes:
        strategy: Feature engineering strategy to use.
        config: Pipeline configuration.
        
    Example:
        >>> pipeline = FeatureEngineeringPipeline(
        ...     strategy=CompositeFeatureStrategy([...]),
        ...     config=FeatureConfig(chunk_size=10000)
        ... )
        >>> result = pipeline.run(large_df)
    """
    
    def __init__(
        self,
        strategy: FeatureEngineeringStrategy,
        config: Optional[FeatureConfig] = None
    ) -> None:
        """
        Initialize feature engineering pipeline.
        
        Args:
            strategy: Feature engineering strategy to use.
            config: Optional configuration override.
        """
        self._strategy = strategy
        self._config = config or get_feature_config()
        self._cache: dict[int, pd.DataFrame] = {}
        
        logger.info(
            "Initialized FeatureEngineeringPipeline",
            extra={
                "strategy": strategy.__class__.__name__,
                "chunk_size": self._config.chunk_size,
                "caching_enabled": self._config.enable_caching
            }
        )
    
    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names from strategy."""
        return self._strategy.feature_names
    
    def run(
        self,
        data: pd.DataFrame,
        use_chunking: bool = True
    ) -> pd.DataFrame:
        """
        Run the feature engineering pipeline.
        
        Args:
            data: Input DataFrame to process.
            use_chunking: Whether to use chunked processing for large data.
            
        Returns:
            DataFrame with engineered features.
            
        Raises:
            FeatureEngineeringError: If pipeline fails.
        """
        start_time = time.time()
        
        logger.info(
            "Starting feature engineering pipeline",
            extra={"input_shape": data.shape}
        )
        
        # Check cache
        if self._config.enable_caching:
            cache_key = self._compute_cache_key(data)
            if cache_key in self._cache:
                logger.info("Returning cached result")
                return self._cache[cache_key].copy()
        
        # Validate input
        self._strategy.validate_input(data)
        
        # Process data
        if use_chunking and len(data) > self._config.chunk_size:
            result = self._process_chunked(data)
        else:
            result = self._strategy.engineer_features(data)
        
        # Cache result
        if self._config.enable_caching:
            self._cache[cache_key] = result.copy()
        
        duration = time.time() - start_time
        logger.info(
            "Feature engineering pipeline completed",
            extra={
                "output_shape": result.shape,
                "duration_seconds": round(duration, 2),
                "features_added": len(self.feature_names)
            }
        )
        
        return result
    
    def _process_chunked(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data in chunks for memory efficiency.
        
        Args:
            data: Large input DataFrame.
            
        Returns:
            Concatenated results from all chunks.
        """
        chunk_size = self._config.chunk_size
        n_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        logger.info(
            "Processing in chunks",
            extra={"n_chunks": n_chunks, "chunk_size": chunk_size}
        )
        
        results = []
        
        for i, start in enumerate(range(0, len(data), chunk_size)):
            end = min(start + chunk_size, len(data))
            chunk = data.iloc[start:end]
            
            logger.debug(f"Processing chunk {i+1}/{n_chunks}")
            
            chunk_result = self._strategy.engineer_features(chunk)
            results.append(chunk_result)
        
        return pd.concat(results, ignore_index=True)
    
    def _compute_cache_key(self, data: pd.DataFrame) -> int:
        """
        Compute cache key for data.
        
        Uses hash of shape and sample values for efficiency.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Integer cache key.
        """
        # Use shape and hash of first/last rows for quick comparison
        key_parts = [
            data.shape,
            tuple(data.columns),
            tuple(data.iloc[0].values) if len(data) > 0 else (),
            tuple(data.iloc[-1].values) if len(data) > 0 else (),
        ]
        return hash(tuple(str(p) for p in key_parts))
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")


@contextmanager
def _strategy_context(strategy_name: str):
    """
    Context manager for strategy execution with timing and error handling.
    
    Args:
        strategy_name: Name of the strategy being executed.
        
    Yields:
        None
        
    Raises:
        FeatureEngineeringError: If strategy execution fails.
    """
    start_time = time.time()
    
    try:
        yield
    except FeatureEngineeringError:
        raise
    except Exception as e:
        logger.error(
            f"Strategy {strategy_name} failed",
            exc_info=True,
            extra={"strategy": strategy_name, "error": str(e)}
        )
        raise FeatureEngineeringError(
            message=f"Strategy execution failed: {e}",
            strategy_name=strategy_name
        ) from e
    finally:
        duration = time.time() - start_time
        logger.debug(
            f"Strategy {strategy_name} completed",
            extra={"strategy": strategy_name, "duration_seconds": round(duration, 3)}
        )

