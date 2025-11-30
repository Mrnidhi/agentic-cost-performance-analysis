"""
Temporal feature engineering strategy.

This module implements time-based feature calculations for trend analysis,
stability metrics, and seasonality detection.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import FeatureConfig, get_feature_config
from src.core.exceptions import FeatureEngineeringError


logger = logging.getLogger(__name__)


# Required columns for temporal feature calculation
REQUIRED_COLUMNS = frozenset({
    "timestamp",
    "success_rate",
    "agent_id",
})


class TemporalFeaturesStrategy:
    """
    Strategy for calculating temporal/time-based features.
    
    Implements the FeatureEngineeringStrategy protocol to provide
    trend analysis, stability metrics, and seasonality features.
    
    Attributes:
        config: Feature configuration with temporal settings.
        
    Example:
        >>> strategy = TemporalFeaturesStrategy()
        >>> engineered_df = strategy.engineer_features(raw_df)
        >>> print(strategy.feature_names)
        ['performance_trend_7d', 'stability_index', ...]
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """
        Initialize temporal features strategy.
        
        Args:
            config: Optional configuration override. Uses default if not provided.
        """
        self._config = config or get_feature_config()
        self._feature_names = [
            "performance_trend_7d",
            "stability_index",
            "degradation_risk_score",
            "seasonality_impact",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
        ]
        logger.info(
            "Initialized TemporalFeaturesStrategy",
            extra={"window_days": self._config.temporal_window_days}
        )
    
    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names produced by this strategy."""
        return self._feature_names.copy()
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.
        
        Args:
            data: Input DataFrame to validate.
            
        Returns:
            True if valid.
            
        Raises:
            FeatureEngineeringError: If required columns are missing.
        """
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise FeatureEngineeringError(
                message=f"Missing required columns for temporal features",
                missing_columns=list(missing),
                strategy_name="TemporalFeaturesStrategy"
            )
        return True
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into temporal features.
        
        Args:
            data: Raw agent performance data with timestamp column.
            
        Returns:
            DataFrame with temporal features added.
            
        Raises:
            FeatureEngineeringError: If transformation fails.
        """
        self.validate_input(data)
        
        result = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result["timestamp"]):
            result["timestamp"] = pd.to_datetime(result["timestamp"])
        
        # Sort by timestamp for rolling calculations
        result = result.sort_values("timestamp")
        
        logger.debug("Extracting basic temporal features")
        result["hour_of_day"] = result["timestamp"].dt.hour
        result["day_of_week"] = result["timestamp"].dt.dayofweek
        result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
        
        logger.debug("Calculating performance_trend_7d")
        result["performance_trend_7d"] = calculate_performance_trend(
            data=result,
            metric_col="success_rate",
            group_col="agent_id",
            window_days=self._config.temporal_window_days
        )
        
        logger.debug("Calculating stability_index")
        result["stability_index"] = calculate_stability_index(
            data=result,
            metric_col="success_rate",
            group_col="agent_id"
        )
        
        logger.debug("Calculating degradation_risk_score")
        result["degradation_risk_score"] = calculate_degradation_risk(
            trend=result["performance_trend_7d"],
            stability=result["stability_index"]
        )
        
        logger.debug("Calculating seasonality_impact")
        result["seasonality_impact"] = calculate_seasonality_impact(
            data=result,
            metric_col="success_rate"
        )
        
        logger.info(
            "Temporal feature engineering completed",
            extra={"features_added": len(self._feature_names)}
        )
        
        return result


def calculate_performance_trend(
    data: pd.DataFrame,
    metric_col: str,
    group_col: str,
    window_days: int = 7
) -> pd.Series:
    """
    Calculate performance trend over rolling window.
    
    Uses linear regression slope to measure trend direction and magnitude.
    Positive values indicate improving performance.
    
    Args:
        data: DataFrame with timestamp and metric columns.
        metric_col: Name of the metric column to analyze.
        group_col: Name of the grouping column (e.g., agent_id).
        window_days: Number of days for rolling window.
        
    Returns:
        Series of trend slopes (positive = improving).
        
    Example:
        >>> trend = calculate_performance_trend(
        ...     data=df,
        ...     metric_col="success_rate",
        ...     group_col="agent_id",
        ...     window_days=7
        ... )
    """
    def _calculate_slope(group: pd.DataFrame) -> pd.Series:
        """Calculate rolling slope for a group."""
        if len(group) < 3:
            return pd.Series(0.0, index=group.index)
        
        # Convert timestamps to numeric (days since start)
        time_numeric = (
            (group["timestamp"] - group["timestamp"].min())
            .dt.total_seconds() / 86400
        )
        
        # Rolling slope calculation
        window = min(len(group), window_days * 24)  # Approximate hourly data
        slopes = []
        
        for i in range(len(group)):
            start_idx = max(0, i - window + 1)
            window_time = time_numeric.iloc[start_idx:i+1].values
            window_metric = group[metric_col].iloc[start_idx:i+1].values
            
            if len(window_time) >= 3 and np.std(window_time) > 0:
                slope, _, _, _, _ = stats.linregress(window_time, window_metric)
                slopes.append(slope)
            else:
                slopes.append(0.0)
        
        return pd.Series(slopes, index=group.index)
    
    # Apply slope calculation per group
    result = data.groupby(group_col, group_keys=False).apply(_calculate_slope)
    
    # Handle any NaN values
    return result.fillna(0.0)


def calculate_stability_index(
    data: pd.DataFrame,
    metric_col: str,
    group_col: str
) -> pd.Series:
    """
    Calculate stability index based on coefficient of variation.
    
    Formula:
        stability_index = 1 - CV, where CV = std / mean
    
    Higher values indicate more stable performance.
    
    Args:
        data: DataFrame with metric column.
        metric_col: Name of the metric column to analyze.
        group_col: Name of the grouping column.
        
    Returns:
        Series of stability indices (0-1, higher = more stable).
    """
    def _calculate_cv_stability(group: pd.DataFrame) -> pd.Series:
        """Calculate stability for a group."""
        mean_val = group[metric_col].mean()
        std_val = group[metric_col].std()
        
        if mean_val == 0 or pd.isna(mean_val):
            cv = 0
        else:
            cv = std_val / mean_val
        
        # Stability is inverse of CV, clamped to [0, 1]
        stability = max(0.0, min(1.0, 1 - cv))
        
        return pd.Series(stability, index=group.index)
    
    result = data.groupby(group_col, group_keys=False).apply(_calculate_cv_stability)
    
    return result.fillna(0.5)  # Default to medium stability


def calculate_degradation_risk(
    trend: pd.Series,
    stability: pd.Series,
    trend_weight: float = 0.6,
    stability_weight: float = 0.4
) -> pd.Series:
    """
    Calculate risk of performance degradation.
    
    Combines negative trend and low stability into a risk score.
    
    Args:
        trend: Performance trend values (negative = degrading).
        stability: Stability index values.
        trend_weight: Weight for trend component.
        stability_weight: Weight for stability component.
        
    Returns:
        Series of degradation risk scores (0-1, higher = more risk).
    """
    # Negative trend increases risk
    trend_risk = (-trend).clip(lower=0)
    
    # Normalize trend risk to 0-1 range using sigmoid
    trend_risk_normalized = 2 / (1 + np.exp(-trend_risk * 10)) - 1
    
    # Low stability increases risk
    stability_risk = 1 - stability
    
    risk_score = (
        trend_weight * trend_risk_normalized +
        stability_weight * stability_risk
    )
    
    return risk_score.clip(lower=0.0, upper=1.0)


def calculate_seasonality_impact(
    data: pd.DataFrame,
    metric_col: str
) -> pd.Series:
    """
    Calculate seasonality impact on performance.
    
    Measures deviation from overall mean based on hour of day patterns.
    
    Args:
        data: DataFrame with timestamp and metric columns.
        metric_col: Name of the metric column to analyze.
        
    Returns:
        Series of seasonality impact values (-1 to 1).
    """
    # Ensure we have hour_of_day
    if "hour_of_day" not in data.columns:
        hour_of_day = pd.to_datetime(data["timestamp"]).dt.hour
    else:
        hour_of_day = data["hour_of_day"]
    
    overall_mean = data[metric_col].mean()
    
    if overall_mean == 0 or pd.isna(overall_mean):
        return pd.Series(0.0, index=data.index)
    
    # Calculate hourly averages
    hourly_avg = data.groupby(hour_of_day)[metric_col].transform("mean")
    
    # Seasonality impact is deviation from overall mean
    impact = (hourly_avg - overall_mean) / overall_mean
    
    return impact.clip(lower=-1.0, upper=1.0)
