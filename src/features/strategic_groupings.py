"""
Strategic groupings feature engineering strategy.

This module implements strategic classification features for agent
categorization, performance tiers, and business importance rankings.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.core.config import FeatureConfig, get_feature_config
from src.core.types import StrategicImportance, CostEfficiencyTier
from src.core.exceptions import FeatureEngineeringError


logger = logging.getLogger(__name__)


# Required columns for strategic groupings
REQUIRED_COLUMNS = frozenset({
    "success_rate",
    "cost_efficiency_ratio",
    "agent_type",
    "task_complexity",
})


class StrategicGroupingsStrategy:
    """
    Strategy for calculating strategic classification features.
    
    Implements the FeatureEngineeringStrategy protocol to provide
    performance quartiles, efficiency tiers, and importance rankings.
    
    Attributes:
        config: Feature configuration with grouping settings.
        
    Example:
        >>> strategy = StrategicGroupingsStrategy()
        >>> engineered_df = strategy.engineer_features(raw_df)
        >>> print(strategy.feature_names)
        ['performance_quartile', 'cost_efficiency_tier', ...]
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """
        Initialize strategic groupings strategy.
        
        Args:
            config: Optional configuration override. Uses default if not provided.
        """
        self._config = config or get_feature_config()
        self._feature_names = [
            "performance_quartile",
            "cost_efficiency_tier",
            "strategic_importance",
            "performance_percentile",
            "cost_efficiency_percentile",
        ]
        logger.info(
            "Initialized StrategicGroupingsStrategy",
            extra={
                "quartiles": self._config.performance_quartiles,
                "tiers": self._config.cost_efficiency_tiers
            }
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
                message=f"Missing required columns for strategic groupings",
                missing_columns=list(missing),
                strategy_name="StrategicGroupingsStrategy"
            )
        return True
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into strategic grouping features.
        
        Args:
            data: Raw agent performance data.
            
        Returns:
            DataFrame with strategic grouping features added.
            
        Raises:
            FeatureEngineeringError: If transformation fails.
        """
        self.validate_input(data)
        
        result = data.copy()
        
        logger.debug("Calculating performance_quartile")
        result["performance_quartile"], result["performance_percentile"] = (
            calculate_performance_quartile(
                data=result,
                metric_col="success_rate",
                group_col="agent_type",
                n_quartiles=self._config.performance_quartiles
            )
        )
        
        logger.debug("Calculating cost_efficiency_tier")
        result["cost_efficiency_tier"], result["cost_efficiency_percentile"] = (
            calculate_cost_efficiency_tier(
                data=result,
                metric_col="cost_efficiency_ratio",
                n_tiers=self._config.cost_efficiency_tiers
            )
        )
        
        logger.debug("Calculating strategic_importance")
        # Use business_value_score if available, otherwise calculate proxy
        if "business_value_score" in result.columns:
            business_value = result["business_value_score"]
        else:
            business_value = (
                result["success_rate"] * 0.5 +
                result["cost_efficiency_ratio"] * 0.5
            )
        
        result["strategic_importance"] = calculate_strategic_importance(
            business_value=business_value,
            complexity=result["task_complexity"]
        )
        
        logger.info(
            "Strategic groupings feature engineering completed",
            extra={"features_added": len(self._feature_names)}
        )
        
        return result


def calculate_performance_quartile(
    data: pd.DataFrame,
    metric_col: str,
    group_col: str,
    n_quartiles: int = 4
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate performance quartile within each group.
    
    Args:
        data: DataFrame with metric and group columns.
        metric_col: Name of the performance metric column.
        group_col: Name of the grouping column (e.g., agent_type).
        n_quartiles: Number of quartiles (default 4).
        
    Returns:
        Tuple of (quartile labels, percentile ranks).
        
    Example:
        >>> quartile, percentile = calculate_performance_quartile(
        ...     data=df,
        ...     metric_col="success_rate",
        ...     group_col="agent_type"
        ... )
    """
    def _assign_quartile(group: pd.DataFrame) -> pd.DataFrame:
        """Assign quartile within a group."""
        percentile = group[metric_col].rank(pct=True)
        
        # Create quartile labels (1 = bottom, n = top)
        quartile = pd.cut(
            percentile,
            bins=n_quartiles,
            labels=[f"Q{i+1}" for i in range(n_quartiles)],
            include_lowest=True
        )
        
        return pd.DataFrame({
            "quartile": quartile,
            "percentile": percentile * 100
        }, index=group.index)
    
    result = data.groupby(group_col, group_keys=False).apply(_assign_quartile)
    
    return result["quartile"], result["percentile"]


def calculate_cost_efficiency_tier(
    data: pd.DataFrame,
    metric_col: str,
    n_tiers: int = 5
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate cost efficiency tier.
    
    Args:
        data: DataFrame with cost efficiency metric.
        metric_col: Name of the cost efficiency column.
        n_tiers: Number of tiers (default 5).
        
    Returns:
        Tuple of (tier labels, percentile ranks).
    """
    percentile = data[metric_col].rank(pct=True) * 100
    
    # Map percentile to tier using CostEfficiencyTier enum
    tier_labels = percentile.apply(CostEfficiencyTier.from_percentile)
    tier_labels = tier_labels.apply(lambda x: x.value)
    
    return tier_labels, percentile


def calculate_strategic_importance(
    business_value: pd.Series,
    complexity: pd.Series
) -> pd.Series:
    """
    Calculate strategic importance classification.
    
    Uses business value and task complexity to determine strategic
    importance level for resource allocation decisions.
    
    Args:
        business_value: Business value scores (0-1).
        complexity: Task complexity values (1-10).
        
    Returns:
        Series of strategic importance labels.
        
    Example:
        >>> importance = calculate_strategic_importance(
        ...     business_value=pd.Series([0.8, 0.5, 0.3]),
        ...     complexity=pd.Series([8, 5, 3])
        ... )
    """
    def _classify(row: pd.Series) -> str:
        """Classify single row."""
        return StrategicImportance.classify(
            business_value=row["bv"],
            complexity=int(row["cx"])
        ).value
    
    df = pd.DataFrame({"bv": business_value, "cx": complexity})
    
    return df.apply(_classify, axis=1)


def classify_agent_portfolio(
    data: pd.DataFrame,
    performance_col: str = "success_rate",
    cost_col: str = "cost_efficiency_ratio",
    volume_col: Optional[str] = None
) -> pd.Series:
    """
    Classify agents into portfolio categories.
    
    Categories:
        - Workhorses: High volume, good efficiency
        - Specialists: High performance, lower volume
        - Underperformers: Below average on key metrics
        - Rising Stars: Improving trend
    
    Args:
        data: DataFrame with agent metrics.
        performance_col: Name of performance metric column.
        cost_col: Name of cost efficiency column.
        volume_col: Optional name of volume/usage column.
        
    Returns:
        Series of portfolio category labels.
    """
    perf_median = data[performance_col].median()
    cost_median = data[cost_col].median()
    
    def _classify(row: pd.Series) -> str:
        """Classify single agent."""
        high_perf = row[performance_col] >= perf_median
        high_cost_eff = row[cost_col] >= cost_median
        
        if high_perf and high_cost_eff:
            return "Workhorse"
        elif high_perf and not high_cost_eff:
            return "Specialist"
        elif not high_perf and high_cost_eff:
            return "Rising Star"
        else:
            return "Underperformer"
    
    return data.apply(_classify, axis=1)

