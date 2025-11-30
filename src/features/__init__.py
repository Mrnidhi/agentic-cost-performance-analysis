"""
Feature engineering module for AI Agent Performance Intelligence System.

This module provides strategic feature engineering capabilities using
the Strategy pattern for flexible, testable feature creation.

Course: DATA 230 (Data Visualization) at SJSU
"""

from src.features.business_metrics import (
    BusinessMetricsStrategy,
    calculate_business_value_score,
    calculate_operational_risk_index,
    calculate_scalability_potential,
    calculate_total_cost_of_ownership,
)
from src.features.temporal_features import (
    TemporalFeaturesStrategy,
    calculate_performance_trend,
    calculate_stability_index,
    calculate_degradation_risk,
    calculate_seasonality_impact,
)
from src.features.strategic_groupings import (
    StrategicGroupingsStrategy,
    calculate_performance_quartile,
    calculate_cost_efficiency_tier,
    calculate_strategic_importance,
)
from src.features.pipeline import (
    FeatureEngineeringPipeline,
    CompositeFeatureStrategy,
)

__all__ = [
    # Strategies
    "BusinessMetricsStrategy",
    "TemporalFeaturesStrategy",
    "StrategicGroupingsStrategy",
    "CompositeFeatureStrategy",
    "FeatureEngineeringPipeline",
    # Business metrics functions
    "calculate_business_value_score",
    "calculate_operational_risk_index",
    "calculate_scalability_potential",
    "calculate_total_cost_of_ownership",
    # Temporal features functions
    "calculate_performance_trend",
    "calculate_stability_index",
    "calculate_degradation_risk",
    "calculate_seasonality_impact",
    # Strategic groupings functions
    "calculate_performance_quartile",
    "calculate_cost_efficiency_tier",
    "calculate_strategic_importance",
]
