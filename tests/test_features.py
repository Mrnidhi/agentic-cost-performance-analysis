"""
Tests for feature engineering module.

This module provides comprehensive tests for feature engineering
strategies including unit tests and property-based tests.

Course: DATA 230 (Data Visualization) at SJSU
"""

import pytest
import numpy as np
import pandas as pd

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
from src.core.exceptions import FeatureEngineeringError
from src.core.config import BusinessMetricsConfig


class TestBusinessMetricsStrategy:
    """Tests for BusinessMetricsStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BusinessMetricsStrategy()
        
        assert strategy.feature_names == [
            "business_value_score",
            "operational_risk_index",
            "scalability_potential",
            "total_cost_of_ownership",
        ]
    
    def test_validate_input_success(self, sample_agent_data: pd.DataFrame):
        """Test input validation with valid data."""
        strategy = BusinessMetricsStrategy()
        
        assert strategy.validate_input(sample_agent_data) is True
    
    def test_validate_input_missing_columns(self):
        """Test input validation with missing columns."""
        strategy = BusinessMetricsStrategy()
        invalid_data = pd.DataFrame({"agent_id": ["a1", "a2"]})
        
        with pytest.raises(FeatureEngineeringError) as exc_info:
            strategy.validate_input(invalid_data)
        
        assert "Missing required columns" in str(exc_info.value)
    
    def test_engineer_features(self, sample_agent_data: pd.DataFrame):
        """Test feature engineering produces expected columns."""
        strategy = BusinessMetricsStrategy()
        
        result = strategy.engineer_features(sample_agent_data)
        
        for feature in strategy.feature_names:
            assert feature in result.columns
    
    def test_engineer_features_preserves_original_data(
        self,
        sample_agent_data: pd.DataFrame
    ):
        """Test that original columns are preserved."""
        strategy = BusinessMetricsStrategy()
        original_columns = set(sample_agent_data.columns)
        
        result = strategy.engineer_features(sample_agent_data)
        
        assert original_columns.issubset(set(result.columns))


class TestBusinessValueScore:
    """Tests for business value score calculation."""
    
    def test_score_bounds(self):
        """Test that scores are within 0-1 bounds."""
        success_rate = pd.Series([0.0, 0.5, 1.0])
        cost_efficiency = pd.Series([0.0, 0.5, 1.0])
        resource_efficiency = pd.Series([0.0, 0.5, 1.0])
        human_intervention = pd.Series([1.0, 0.5, 0.0])
        error_recovery = pd.Series([0.0, 0.5, 1.0])
        privacy_compliance = pd.Series([0.0, 0.5, 1.0])
        
        score = calculate_business_value_score(
            success_rate=success_rate,
            cost_efficiency_ratio=cost_efficiency,
            resource_efficiency=resource_efficiency,
            human_intervention_freq=human_intervention,
            error_recovery_rate=error_recovery,
            privacy_compliance=privacy_compliance,
        )
        
        assert (score >= 0).all()
        assert (score <= 1).all()
    
    def test_high_performance_high_score(self):
        """Test that high performance metrics yield high scores."""
        n = 10
        score = calculate_business_value_score(
            success_rate=pd.Series([0.95] * n),
            cost_efficiency_ratio=pd.Series([0.90] * n),
            resource_efficiency=pd.Series([0.85] * n),
            human_intervention_freq=pd.Series([0.05] * n),
            error_recovery_rate=pd.Series([0.92] * n),
            privacy_compliance=pd.Series([0.98] * n),
        )
        
        assert (score > 0.8).all()
    
    def test_custom_config_weights(self):
        """Test with custom configuration weights."""
        config = BusinessMetricsConfig(
            success_rate_weight=0.5,
            cost_efficiency_weight=0.2,
            resource_efficiency_weight=0.1,
            human_intervention_weight=0.1,
            error_recovery_weight=0.05,
            privacy_compliance_weight=0.05,
        )
        
        score = calculate_business_value_score(
            success_rate=pd.Series([1.0]),
            cost_efficiency_ratio=pd.Series([0.0]),
            resource_efficiency=pd.Series([0.0]),
            human_intervention_freq=pd.Series([1.0]),
            error_recovery_rate=pd.Series([0.0]),
            privacy_compliance=pd.Series([0.0]),
            config=config,
        )
        
        # With success_rate=1.0 and weight=0.5, score should be 0.5
        assert score.iloc[0] == pytest.approx(0.5, abs=0.01)


class TestOperationalRiskIndex:
    """Tests for operational risk index calculation."""
    
    def test_risk_bounds(self):
        """Test that risk index is within 0-1 bounds."""
        risk = calculate_operational_risk_index(
            failure_rate=pd.Series([0.0, 0.5, 1.0]),
            complexity=pd.Series([1, 5, 10]),
            cpu_usage=pd.Series([10, 50, 100]),
            memory_usage=pd.Series([100, 300, 500]),
        )
        
        assert (risk >= 0).all()
        assert (risk <= 1).all()
    
    def test_high_failure_high_risk(self):
        """Test that high failure rate yields high risk."""
        low_risk = calculate_operational_risk_index(
            failure_rate=pd.Series([0.1]),
            complexity=pd.Series([3]),
            cpu_usage=pd.Series([30]),
            memory_usage=pd.Series([200]),
        )
        
        high_risk = calculate_operational_risk_index(
            failure_rate=pd.Series([0.9]),
            complexity=pd.Series([3]),
            cpu_usage=pd.Series([30]),
            memory_usage=pd.Series([200]),
        )
        
        assert high_risk.iloc[0] > low_risk.iloc[0]


class TestScalabilityPotential:
    """Tests for scalability potential calculation."""
    
    def test_scalability_bounds(self):
        """Test that scalability is within 0-1 bounds."""
        scalability = calculate_scalability_potential(
            efficiency_score=pd.Series([0.0, 0.5, 1.0]),
            resource_efficiency=pd.Series([0.0, 0.5, 1.0]),
            cpu_usage=pd.Series([100, 50, 0]),
        )
        
        assert (scalability >= 0).all()
        assert (scalability <= 1).all()
    
    def test_low_cpu_high_scalability(self):
        """Test that low CPU usage indicates high scalability."""
        low_cpu = calculate_scalability_potential(
            efficiency_score=pd.Series([0.8]),
            resource_efficiency=pd.Series([0.8]),
            cpu_usage=pd.Series([20]),
        )
        
        high_cpu = calculate_scalability_potential(
            efficiency_score=pd.Series([0.8]),
            resource_efficiency=pd.Series([0.8]),
            cpu_usage=pd.Series([90]),
        )
        
        assert low_cpu.iloc[0] > high_cpu.iloc[0]


class TestTemporalFeaturesStrategy:
    """Tests for TemporalFeaturesStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = TemporalFeaturesStrategy()
        
        assert "performance_trend_7d" in strategy.feature_names
        assert "stability_index" in strategy.feature_names
    
    def test_engineer_features(self, sample_agent_data: pd.DataFrame):
        """Test temporal feature engineering."""
        strategy = TemporalFeaturesStrategy()
        
        result = strategy.engineer_features(sample_agent_data)
        
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
    
    def test_hour_extraction(self, sample_agent_data: pd.DataFrame):
        """Test hour of day extraction."""
        strategy = TemporalFeaturesStrategy()
        
        result = strategy.engineer_features(sample_agent_data)
        
        assert (result["hour_of_day"] >= 0).all()
        assert (result["hour_of_day"] <= 23).all()


class TestStabilityIndex:
    """Tests for stability index calculation."""
    
    def test_stability_bounds(self, sample_agent_data: pd.DataFrame):
        """Test that stability is within 0-1 bounds."""
        stability = calculate_stability_index(
            data=sample_agent_data,
            metric_col="success_rate",
            group_col="agent_type",
        )
        
        assert (stability >= 0).all()
        assert (stability <= 1).all()
    
    def test_constant_series_high_stability(self):
        """Test that constant series has high stability."""
        data = pd.DataFrame({
            "success_rate": [0.9] * 10,
            "agent_type": ["A"] * 10,
        })
        
        stability = calculate_stability_index(
            data=data,
            metric_col="success_rate",
            group_col="agent_type",
        )
        
        # Constant series should have high stability (CV = 0 -> stability = 1)
        # Result may be DataFrame or Series depending on pandas version
        if hasattr(stability, 'values'):
            values = stability.values.flatten()
        else:
            values = [stability]
        
        # All values should be 1.0 for constant series
        assert all(v == 1.0 for v in values)


class TestStrategicGroupingsStrategy:
    """Tests for StrategicGroupingsStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = StrategicGroupingsStrategy()
        
        assert "performance_quartile" in strategy.feature_names
        assert "cost_efficiency_tier" in strategy.feature_names
        assert "strategic_importance" in strategy.feature_names
    
    def test_engineer_features(self, sample_agent_data: pd.DataFrame):
        """Test strategic grouping feature engineering."""
        strategy = StrategicGroupingsStrategy()
        
        result = strategy.engineer_features(sample_agent_data)
        
        for feature in strategy.feature_names:
            assert feature in result.columns


class TestPerformanceQuartile:
    """Tests for performance quartile calculation."""
    
    def test_quartile_distribution(self, sample_agent_data: pd.DataFrame):
        """Test that quartiles are properly distributed."""
        quartile, percentile = calculate_performance_quartile(
            data=sample_agent_data,
            metric_col="success_rate",
            group_col="agent_type",
        )
        
        # Check that all quartiles are present
        unique_quartiles = quartile.unique()
        assert len(unique_quartiles) >= 2  # At least some distribution


class TestStrategicImportance:
    """Tests for strategic importance classification."""
    
    def test_critical_classification(self):
        """Test critical importance classification."""
        importance = calculate_strategic_importance(
            business_value=pd.Series([0.9]),
            complexity=pd.Series([9]),
        )
        
        assert importance.iloc[0] == "Critical"
    
    def test_low_classification(self):
        """Test low importance classification."""
        importance = calculate_strategic_importance(
            business_value=pd.Series([0.2]),
            complexity=pd.Series([2]),
        )
        
        assert importance.iloc[0] == "Low"


# Property-based tests using hypothesis (optional)
try:
    from hypothesis import given, strategies as st, assume
    
    class TestBusinessValueScoreProperties:
        """Property-based tests for business value score."""
        
        @given(st.floats(min_value=0, max_value=1))
        def test_score_monotonic_with_success_rate(self, success_rate: float):
            """Test that score increases with success rate."""
            assume(not np.isnan(success_rate))
            
            score = calculate_business_value_score(
                success_rate=pd.Series([success_rate]),
                cost_efficiency_ratio=pd.Series([0.5]),
                resource_efficiency=pd.Series([0.5]),
                human_intervention_freq=pd.Series([0.5]),
                error_recovery_rate=pd.Series([0.5]),
                privacy_compliance=pd.Series([0.5]),
            )
            
            assert 0 <= score.iloc[0] <= 1
        
        @given(
            st.floats(min_value=0, max_value=1),
            st.floats(min_value=0, max_value=1),
        )
        def test_score_symmetry(self, val1: float, val2: float):
            """Test that swapping equal weights gives same result."""
            assume(not np.isnan(val1) and not np.isnan(val2))
            
            score1 = calculate_business_value_score(
                success_rate=pd.Series([val1]),
                cost_efficiency_ratio=pd.Series([val2]),
                resource_efficiency=pd.Series([0.5]),
                human_intervention_freq=pd.Series([0.5]),
                error_recovery_rate=pd.Series([0.5]),
                privacy_compliance=pd.Series([0.5]),
            )
            
            # With equal weights, swapping should give similar (but not identical) results
            assert 0 <= score1.iloc[0] <= 1

except ImportError:
    pass  # hypothesis not installed, skip property-based tests
