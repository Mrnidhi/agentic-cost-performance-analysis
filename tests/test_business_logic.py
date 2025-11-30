"""
Tests for business logic and rules.

This module provides tests for business rule validation,
domain logic, and calculation correctness.

Course: DATA 230 (Data Visualization) at SJSU
"""

import pytest
import numpy as np
import pandas as pd
from typing import Any

from src.core.types import (
    RiskLevel,
    StrategicImportance,
    CostEfficiencyTier,
    CompetitivePosition,
    DeploymentEnvironment,
    PerformanceMetrics,
    ResourceUsage,
)
from src.core.config import BusinessMetricsConfig


class TestRiskLevelClassification:
    """Tests for risk level classification logic."""
    
    @pytest.mark.parametrize("score,expected_level", [
        (0, RiskLevel.LOW),
        (10, RiskLevel.LOW),
        (19, RiskLevel.LOW),
        (20, RiskLevel.MEDIUM),
        (35, RiskLevel.MEDIUM),
        (49, RiskLevel.MEDIUM),
        (50, RiskLevel.HIGH),
        (60, RiskLevel.HIGH),
        (74, RiskLevel.HIGH),
        (75, RiskLevel.CRITICAL),
        (90, RiskLevel.CRITICAL),
        (100, RiskLevel.CRITICAL),
    ])
    def test_risk_level_from_score(self, score: float, expected_level: RiskLevel):
        """Test risk level classification from scores."""
        result = RiskLevel.from_score(score)
        assert result == expected_level
    
    def test_requires_immediate_action(self):
        """Test immediate action requirement logic."""
        assert RiskLevel.LOW.requires_immediate_action is False
        assert RiskLevel.MEDIUM.requires_immediate_action is False
        assert RiskLevel.HIGH.requires_immediate_action is True
        assert RiskLevel.CRITICAL.requires_immediate_action is True
    
    def test_color_codes(self):
        """Test risk level color codes."""
        assert RiskLevel.LOW.color_code == "#4caf50"  # Green
        assert RiskLevel.CRITICAL.color_code == "#9c27b0"  # Purple


class TestStrategicImportanceClassification:
    """Tests for strategic importance classification."""
    
    @pytest.mark.parametrize("business_value,complexity,expected", [
        (0.9, 9, StrategicImportance.CRITICAL),
        (0.7, 7, StrategicImportance.CRITICAL),
        (0.6, 6, StrategicImportance.HIGH),
        (0.5, 5, StrategicImportance.HIGH),
        (0.4, 4, StrategicImportance.MEDIUM),
        (0.3, 3, StrategicImportance.MEDIUM),
        (0.2, 2, StrategicImportance.LOW),
        (0.1, 1, StrategicImportance.LOW),
    ])
    def test_strategic_importance_classification(
        self,
        business_value: float,
        complexity: int,
        expected: StrategicImportance
    ):
        """Test strategic importance classification."""
        result = StrategicImportance.classify(business_value, complexity)
        assert result == expected
    
    def test_edge_cases(self):
        """Test edge cases for classification."""
        # Exactly at threshold
        assert StrategicImportance.classify(0.7, 7) == StrategicImportance.CRITICAL
        assert StrategicImportance.classify(0.5, 5) == StrategicImportance.HIGH
        assert StrategicImportance.classify(0.3, 3) == StrategicImportance.MEDIUM


class TestCostEfficiencyTier:
    """Tests for cost efficiency tier classification."""
    
    @pytest.mark.parametrize("percentile,expected_tier", [
        (5, CostEfficiencyTier.VERY_LOW),
        (15, CostEfficiencyTier.VERY_LOW),
        (25, CostEfficiencyTier.LOW),
        (35, CostEfficiencyTier.LOW),
        (45, CostEfficiencyTier.MEDIUM),
        (55, CostEfficiencyTier.MEDIUM),
        (65, CostEfficiencyTier.HIGH),
        (75, CostEfficiencyTier.HIGH),
        (85, CostEfficiencyTier.VERY_HIGH),
        (95, CostEfficiencyTier.VERY_HIGH),
    ])
    def test_tier_from_percentile(
        self,
        percentile: float,
        expected_tier: CostEfficiencyTier
    ):
        """Test tier classification from percentile."""
        result = CostEfficiencyTier.from_percentile(percentile)
        assert result == expected_tier


class TestCompetitivePosition:
    """Tests for competitive position classification."""
    
    @pytest.mark.parametrize("percentile,expected_position", [
        (90, CompetitivePosition.LEADER),
        (80, CompetitivePosition.LEADER),
        (70, CompetitivePosition.CHALLENGER),
        (60, CompetitivePosition.CHALLENGER),
        (50, CompetitivePosition.FOLLOWER),
        (40, CompetitivePosition.FOLLOWER),
        (30, CompetitivePosition.LAGGARD),
        (10, CompetitivePosition.LAGGARD),
    ])
    def test_position_from_percentile(
        self,
        percentile: float,
        expected_position: CompetitivePosition
    ):
        """Test competitive position from percentile."""
        result = CompetitivePosition.from_percentile(percentile)
        assert result == expected_position


class TestDeploymentEnvironment:
    """Tests for deployment environment properties."""
    
    def test_distributed_environments(self):
        """Test distributed environment identification."""
        assert DeploymentEnvironment.CLOUD.is_distributed is True
        assert DeploymentEnvironment.HYBRID.is_distributed is True
        assert DeploymentEnvironment.EDGE.is_distributed is False
        assert DeploymentEnvironment.SERVER.is_distributed is False
    
    def test_edge_optimization_support(self):
        """Test edge optimization support."""
        assert DeploymentEnvironment.EDGE.supports_edge_optimization is True
        assert DeploymentEnvironment.HYBRID.supports_edge_optimization is True
        assert DeploymentEnvironment.MOBILE.supports_edge_optimization is True
        assert DeploymentEnvironment.CLOUD.supports_edge_optimization is False


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_valid_metrics(self):
        """Test creation with valid metrics."""
        metrics = PerformanceMetrics(
            success_rate=0.95,
            accuracy_score=0.90,
            efficiency_score=0.85,
            response_latency_ms=150.0,
            cost_efficiency_ratio=0.80,
        )
        
        assert metrics.success_rate == 0.95
        assert metrics.overall_performance > 0
    
    def test_invalid_success_rate(self):
        """Test validation for invalid success rate."""
        with pytest.raises(ValueError):
            PerformanceMetrics(
                success_rate=1.5,  # Invalid: > 1
                accuracy_score=0.90,
                efficiency_score=0.85,
                response_latency_ms=150.0,
                cost_efficiency_ratio=0.80,
            )
    
    def test_overall_performance_calculation(self):
        """Test overall performance calculation."""
        metrics = PerformanceMetrics(
            success_rate=1.0,
            accuracy_score=1.0,
            efficiency_score=1.0,
            response_latency_ms=100.0,
            cost_efficiency_ratio=1.0,
        )
        
        # With all 1.0 values and weights summing to 1.0
        assert metrics.overall_performance == 1.0


class TestResourceUsage:
    """Tests for ResourceUsage dataclass."""
    
    def test_valid_usage(self):
        """Test creation with valid usage."""
        usage = ResourceUsage(
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            execution_time_seconds=5.0,
        )
        
        assert usage.memory_usage_mb == 256.0
        assert usage.is_resource_constrained is False
    
    def test_resource_constrained(self):
        """Test resource constrained detection."""
        high_cpu = ResourceUsage(
            memory_usage_mb=200.0,
            cpu_usage_percent=85.0,  # > 80%
            execution_time_seconds=5.0,
        )
        assert high_cpu.is_resource_constrained is True
        
        high_memory = ResourceUsage(
            memory_usage_mb=450.0,  # > 400 MB
            cpu_usage_percent=50.0,
            execution_time_seconds=5.0,
        )
        assert high_memory.is_resource_constrained is True
    
    def test_invalid_cpu_usage(self):
        """Test validation for invalid CPU usage."""
        with pytest.raises(ValueError):
            ResourceUsage(
                memory_usage_mb=256.0,
                cpu_usage_percent=150.0,  # Invalid: > 100
                execution_time_seconds=5.0,
            )


class TestBusinessMetricsConfig:
    """Tests for BusinessMetricsConfig validation."""
    
    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = BusinessMetricsConfig()
        
        total = (
            config.success_rate_weight +
            config.cost_efficiency_weight +
            config.resource_efficiency_weight +
            config.human_intervention_weight +
            config.error_recovery_weight +
            config.privacy_compliance_weight
        )
        
        assert abs(total - 1.0) < 0.001
    
    def test_invalid_weights_sum(self):
        """Test that invalid weight sum raises error."""
        with pytest.raises(ValueError):
            BusinessMetricsConfig(
                success_rate_weight=0.5,
                cost_efficiency_weight=0.5,
                resource_efficiency_weight=0.5,  # Sum > 1.0
                human_intervention_weight=0.0,
                error_recovery_weight=0.0,
                privacy_compliance_weight=0.0,
            )


class TestROICalculation:
    """Tests for ROI calculation logic."""
    
    def calculate_roi(
        self,
        business_value: float,
        cost: float,
        efficiency: float
    ) -> float:
        """Calculate ROI from business metrics."""
        if cost <= 0:
            return 0
        return (business_value * efficiency) / cost
    
    def test_positive_roi(self):
        """Test positive ROI calculation."""
        roi = self.calculate_roi(
            business_value=0.9,
            cost=2.0,
            efficiency=0.85
        )
        
        assert roi > 0
    
    def test_zero_cost_handling(self):
        """Test ROI with zero cost."""
        roi = self.calculate_roi(
            business_value=0.9,
            cost=0,
            efficiency=0.85
        )
        
        assert roi == 0
    
    def test_roi_comparison(self):
        """Test ROI comparison between configurations."""
        high_value_roi = self.calculate_roi(
            business_value=0.95,
            cost=3.0,
            efficiency=0.90
        )
        
        low_value_roi = self.calculate_roi(
            business_value=0.70,
            cost=3.0,
            efficiency=0.70
        )
        
        assert high_value_roi > low_value_roi


class TestOptimizationPotential:
    """Tests for optimization potential calculation."""
    
    def calculate_optimization_potential(
        self,
        current_performance: float,
        benchmark_performance: float,
        cost_efficiency: float
    ) -> float:
        """Calculate optimization potential."""
        performance_gap = benchmark_performance - current_performance
        return max(0, performance_gap * cost_efficiency)
    
    def test_positive_potential(self):
        """Test positive optimization potential."""
        potential = self.calculate_optimization_potential(
            current_performance=0.75,
            benchmark_performance=0.90,
            cost_efficiency=0.80
        )
        
        assert potential > 0
    
    def test_no_potential_when_at_benchmark(self):
        """Test no potential when at benchmark."""
        potential = self.calculate_optimization_potential(
            current_performance=0.90,
            benchmark_performance=0.90,
            cost_efficiency=0.80
        )
        
        assert potential == 0
    
    def test_no_potential_when_above_benchmark(self):
        """Test no potential when above benchmark."""
        potential = self.calculate_optimization_potential(
            current_performance=0.95,
            benchmark_performance=0.90,
            cost_efficiency=0.80
        )
        
        assert potential == 0


class TestParetoOptimality:
    """Tests for Pareto optimality logic."""
    
    def is_pareto_dominated(
        self,
        solution_a: dict[str, float],
        solution_b: dict[str, float],
        maximize: list[str],
        minimize: list[str]
    ) -> bool:
        """Check if solution_a is dominated by solution_b."""
        at_least_as_good = True
        strictly_better = False
        
        for key in maximize:
            if solution_b[key] < solution_a[key]:
                at_least_as_good = False
            if solution_b[key] > solution_a[key]:
                strictly_better = True
        
        for key in minimize:
            if solution_b[key] > solution_a[key]:
                at_least_as_good = False
            if solution_b[key] < solution_a[key]:
                strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def test_dominated_solution(self):
        """Test identification of dominated solution."""
        solution_a = {"performance": 0.7, "cost": 5.0}
        solution_b = {"performance": 0.9, "cost": 3.0}  # Better in both
        
        assert self.is_pareto_dominated(
            solution_a, solution_b,
            maximize=["performance"],
            minimize=["cost"]
        ) is True
    
    def test_non_dominated_solution(self):
        """Test non-dominated solution."""
        solution_a = {"performance": 0.9, "cost": 5.0}  # High perf, high cost
        solution_b = {"performance": 0.7, "cost": 3.0}  # Low perf, low cost
        
        # Neither dominates the other
        assert self.is_pareto_dominated(
            solution_a, solution_b,
            maximize=["performance"],
            minimize=["cost"]
        ) is False
        
        assert self.is_pareto_dominated(
            solution_b, solution_a,
            maximize=["performance"],
            minimize=["cost"]
        ) is False


class TestBusinessValueCalculation:
    """Tests for business value calculation."""
    
    def calculate_business_value(
        self,
        success_rate: float,
        accuracy: float,
        efficiency: float,
        cost_efficiency: float,
        weights: dict[str, float] = None
    ) -> float:
        """Calculate business value score."""
        if weights is None:
            weights = {
                "success_rate": 0.3,
                "accuracy": 0.3,
                "efficiency": 0.2,
                "cost_efficiency": 0.2,
            }
        
        return (
            weights["success_rate"] * success_rate +
            weights["accuracy"] * accuracy +
            weights["efficiency"] * efficiency +
            weights["cost_efficiency"] * cost_efficiency
        )
    
    def test_perfect_scores(self):
        """Test business value with perfect scores."""
        value = self.calculate_business_value(
            success_rate=1.0,
            accuracy=1.0,
            efficiency=1.0,
            cost_efficiency=1.0
        )
        
        assert value == 1.0
    
    def test_zero_scores(self):
        """Test business value with zero scores."""
        value = self.calculate_business_value(
            success_rate=0.0,
            accuracy=0.0,
            efficiency=0.0,
            cost_efficiency=0.0
        )
        
        assert value == 0.0
    
    def test_custom_weights(self):
        """Test business value with custom weights."""
        value = self.calculate_business_value(
            success_rate=1.0,
            accuracy=0.0,
            efficiency=0.0,
            cost_efficiency=0.0,
            weights={
                "success_rate": 1.0,
                "accuracy": 0.0,
                "efficiency": 0.0,
                "cost_efficiency": 0.0,
            }
        )
        
        assert value == 1.0
