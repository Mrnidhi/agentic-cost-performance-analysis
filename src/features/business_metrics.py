"""
Business metrics feature engineering strategy.

This module implements business-centric feature calculations following
the Strategy pattern for clean separation and testability.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.core.config import BusinessMetricsConfig, get_business_metrics_config
from src.core.exceptions import FeatureEngineeringError


logger = logging.getLogger(__name__)


# Required columns for business metrics calculation
REQUIRED_COLUMNS = frozenset({
    "success_rate",
    "cost_efficiency_ratio",
    "efficiency_score",
    "cpu_usage_percent",
    "memory_usage_mb",
    "human_intervention_required",
    "error_recovery_rate",
    "privacy_compliance_score",
    "cost_per_task_cents",
    "task_complexity",
})


class BusinessMetricsStrategy:
    """
    Strategy for calculating business-centric features.
    
    Implements the FeatureEngineeringStrategy protocol to provide
    business value, operational risk, and cost metrics.
    
    Attributes:
        config: Business metrics configuration with weights.
        
    Example:
        >>> strategy = BusinessMetricsStrategy()
        >>> engineered_df = strategy.engineer_features(raw_df)
        >>> print(strategy.feature_names)
        ['business_value_score', 'operational_risk_index', ...]
    """
    
    def __init__(self, config: Optional[BusinessMetricsConfig] = None) -> None:
        """
        Initialize business metrics strategy.
        
        Args:
            config: Optional configuration override. Uses default if not provided.
        """
        self._config = config or get_business_metrics_config()
        self._feature_names = [
            "business_value_score",
            "operational_risk_index",
            "scalability_potential",
            "total_cost_of_ownership",
        ]
        logger.info(
            "Initialized BusinessMetricsStrategy",
            extra={"config": self._config.model_dump()}
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
                message=f"Missing required columns for business metrics",
                missing_columns=list(missing),
                strategy_name="BusinessMetricsStrategy"
            )
        return True
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into business-centric features.
        
        Args:
            data: Raw agent performance data.
            
        Returns:
            DataFrame with business metric features added.
            
        Raises:
            FeatureEngineeringError: If transformation fails.
        """
        self.validate_input(data)
        
        result = data.copy()
        
        logger.debug("Calculating business_value_score")
        result["business_value_score"] = calculate_business_value_score(
            success_rate=data["success_rate"],
            cost_efficiency_ratio=data["cost_efficiency_ratio"],
            resource_efficiency=_calculate_resource_efficiency(
                data["efficiency_score"],
                data["cpu_usage_percent"],
                data["memory_usage_mb"]
            ),
            human_intervention_freq=data["human_intervention_required"].astype(float),
            error_recovery_rate=data["error_recovery_rate"],
            privacy_compliance=data["privacy_compliance_score"],
            config=self._config
        )
        
        logger.debug("Calculating operational_risk_index")
        result["operational_risk_index"] = calculate_operational_risk_index(
            failure_rate=1 - data["success_rate"],
            complexity=data["task_complexity"],
            cpu_usage=data["cpu_usage_percent"],
            memory_usage=data["memory_usage_mb"]
        )
        
        logger.debug("Calculating scalability_potential")
        result["scalability_potential"] = calculate_scalability_potential(
            efficiency_score=data["efficiency_score"],
            resource_efficiency=_calculate_resource_efficiency(
                data["efficiency_score"],
                data["cpu_usage_percent"],
                data["memory_usage_mb"]
            ),
            cpu_usage=data["cpu_usage_percent"]
        )
        
        logger.debug("Calculating total_cost_of_ownership")
        result["total_cost_of_ownership"] = calculate_total_cost_of_ownership(
            cost_per_task=data["cost_per_task_cents"],
            complexity=data["task_complexity"],
            maintenance_factor=_estimate_maintenance_factor(
                data["cpu_usage_percent"],
                data["memory_usage_mb"],
                data["error_recovery_rate"]
            )
        )
        
        logger.info(
            "Business metrics feature engineering completed",
            extra={"features_added": len(self._feature_names)}
        )
        
        return result


def calculate_business_value_score(
    success_rate: pd.Series,
    cost_efficiency_ratio: pd.Series,
    resource_efficiency: pd.Series,
    human_intervention_freq: pd.Series,
    error_recovery_rate: pd.Series,
    privacy_compliance: pd.Series,
    config: Optional[BusinessMetricsConfig] = None
) -> pd.Series:
    """
    Calculate weighted business value score.
    
    Formula:
        BVS = w1*success_rate + w2*cost_efficiency + w3*resource_efficiency
              + w4*(1 - human_intervention) + w5*error_recovery + w6*privacy
    
    Args:
        success_rate: Task success rate (0-1).
        cost_efficiency_ratio: Cost efficiency metric (0-1).
        resource_efficiency: Resource utilization efficiency (0-1).
        human_intervention_freq: Frequency of human intervention (0-1).
        error_recovery_rate: Error recovery success rate (0-1).
        privacy_compliance: Privacy compliance score (0-1).
        config: Configuration with weights. Uses default if not provided.
        
    Returns:
        Series of business value scores (0-1).
        
    Example:
        >>> bvs = calculate_business_value_score(
        ...     success_rate=pd.Series([0.95, 0.80]),
        ...     cost_efficiency_ratio=pd.Series([0.85, 0.70]),
        ...     resource_efficiency=pd.Series([0.90, 0.75]),
        ...     human_intervention_freq=pd.Series([0.1, 0.3]),
        ...     error_recovery_rate=pd.Series([0.88, 0.72]),
        ...     privacy_compliance=pd.Series([0.95, 0.90])
        ... )
    """
    cfg = config or get_business_metrics_config()
    
    # Vectorized calculation for performance
    score = (
        cfg.success_rate_weight * success_rate +
        cfg.cost_efficiency_weight * cost_efficiency_ratio +
        cfg.resource_efficiency_weight * resource_efficiency +
        cfg.human_intervention_weight * (1 - human_intervention_freq) +
        cfg.error_recovery_weight * error_recovery_rate +
        cfg.privacy_compliance_weight * privacy_compliance
    )
    
    # Clamp to valid range
    return score.clip(lower=0.0, upper=1.0)


def calculate_operational_risk_index(
    failure_rate: pd.Series,
    complexity: pd.Series,
    cpu_usage: pd.Series,
    memory_usage: pd.Series,
    failure_weight: float = 0.4,
    complexity_weight: float = 0.3,
    resource_weight: float = 0.3
) -> pd.Series:
    """
    Calculate operational risk index.
    
    Combines failure probability, task complexity, and resource pressure
    into a single risk metric.
    
    Formula:
        ORI = w1*failure_rate + w2*(complexity/10) + w3*resource_pressure
    
    Args:
        failure_rate: Task failure rate (0-1).
        complexity: Task complexity (1-10).
        cpu_usage: CPU usage percentage (0-100).
        memory_usage: Memory usage in MB.
        failure_weight: Weight for failure rate component.
        complexity_weight: Weight for complexity component.
        resource_weight: Weight for resource pressure component.
        
    Returns:
        Series of operational risk indices (0-1).
    """
    # Normalize complexity to 0-1 range
    normalized_complexity = complexity / 10.0
    
    # Calculate resource pressure (normalized)
    resource_pressure = (
        (cpu_usage / 100.0) * 0.6 +
        (memory_usage / 500.0).clip(upper=1.0) * 0.4
    )
    
    risk_index = (
        failure_weight * failure_rate +
        complexity_weight * normalized_complexity +
        resource_weight * resource_pressure
    )
    
    return risk_index.clip(lower=0.0, upper=1.0)


def calculate_scalability_potential(
    efficiency_score: pd.Series,
    resource_efficiency: pd.Series,
    cpu_usage: pd.Series,
    efficiency_weight: float = 0.4,
    resource_weight: float = 0.3,
    headroom_weight: float = 0.3
) -> pd.Series:
    """
    Calculate scalability potential score.
    
    Measures ability to handle increased load based on current
    efficiency and resource headroom.
    
    Args:
        efficiency_score: Current efficiency score (0-1).
        resource_efficiency: Resource utilization efficiency (0-1).
        cpu_usage: Current CPU usage percentage.
        efficiency_weight: Weight for efficiency component.
        resource_weight: Weight for resource efficiency component.
        headroom_weight: Weight for resource headroom component.
        
    Returns:
        Series of scalability potential scores (0-1).
    """
    # Calculate resource headroom (inverse of usage)
    resource_headroom = 1 - (cpu_usage / 100.0)
    
    scalability = (
        efficiency_weight * efficiency_score +
        resource_weight * resource_efficiency +
        headroom_weight * resource_headroom
    )
    
    return scalability.clip(lower=0.0, upper=1.0)


def calculate_total_cost_of_ownership(
    cost_per_task: pd.Series,
    complexity: pd.Series,
    maintenance_factor: pd.Series,
    base_execution_frequency: float = 100.0
) -> pd.Series:
    """
    Calculate total cost of ownership.
    
    Formula:
        TCO = (cost_per_task * execution_frequency) + (maintenance_factor * complexity)
    
    Args:
        cost_per_task: Cost per task in cents.
        complexity: Task complexity (1-10).
        maintenance_factor: Estimated maintenance cost factor.
        base_execution_frequency: Base execution frequency for estimation.
        
    Returns:
        Series of TCO values in cents.
    """
    execution_cost = cost_per_task * base_execution_frequency
    maintenance_cost = maintenance_factor * complexity * 10  # Scale factor
    
    return execution_cost + maintenance_cost


def _calculate_resource_efficiency(
    efficiency_score: pd.Series,
    cpu_usage: pd.Series,
    memory_usage: pd.Series
) -> pd.Series:
    """
    Calculate resource efficiency metric.
    
    Internal helper function for computing resource utilization efficiency.
    
    Args:
        efficiency_score: Base efficiency score.
        cpu_usage: CPU usage percentage.
        memory_usage: Memory usage in MB.
        
    Returns:
        Series of resource efficiency scores (0-1).
    """
    # Avoid division by zero
    resource_denominator = (cpu_usage * memory_usage).replace(0, 1)
    
    raw_efficiency = (efficiency_score / resource_denominator) * 1000
    
    # Normalize using sigmoid-like transformation
    normalized = 2 / (1 + np.exp(-raw_efficiency / 100)) - 1
    
    return normalized.clip(lower=0.0, upper=1.0)


def _estimate_maintenance_factor(
    cpu_usage: pd.Series,
    memory_usage: pd.Series,
    error_recovery_rate: pd.Series
) -> pd.Series:
    """
    Estimate maintenance cost factor.
    
    Higher resource usage and lower error recovery indicate higher
    maintenance requirements.
    
    Args:
        cpu_usage: CPU usage percentage.
        memory_usage: Memory usage in MB.
        error_recovery_rate: Error recovery success rate.
        
    Returns:
        Series of maintenance factors.
    """
    resource_stress = (cpu_usage / 100.0) * 0.5 + (memory_usage / 500.0).clip(upper=1.0) * 0.5
    recovery_penalty = 1 - error_recovery_rate
    
    return (resource_stress * 0.6 + recovery_penalty * 0.4) * 100
