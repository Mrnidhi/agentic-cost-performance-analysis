"""
Tests for FastAPI endpoints.

This module provides integration tests for all API endpoints
with comprehensive validation and error handling tests.

Course: DATA 230 (Data Visualization) at SJSU
"""

import pytest
from fastapi.testclient import TestClient
from typing import Any


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, api_client: TestClient):
        """Test successful health check."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_check_components(self, api_client: TestClient):
        """Test health check includes components."""
        response = api_client.get("/health")
        
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], dict)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_info(self, api_client: TestClient):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestOptimizationEndpoint:
    """Tests for agent configuration optimization endpoint."""
    
    def test_optimize_success(
        self,
        api_client: TestClient,
        optimization_request_data: dict[str, Any]
    ):
        """Test successful optimization request."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            json=optimization_request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "recommended_config" in data
        assert "expected_performance" in data
        assert "optimization_insights" in data
    
    def test_optimize_invalid_accuracy(self, api_client: TestClient):
        """Test optimization with invalid accuracy value."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            json={
                "task_category": "Data Analysis",
                "required_accuracy": 1.5,  # Invalid: > 1.0
                "budget_constraint": 5.0,
                "latency_requirement": 500,
                "privacy_requirements": "medium",
                "business_criticality": "high",
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_optimize_invalid_budget(self, api_client: TestClient):
        """Test optimization with invalid budget."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            json={
                "task_category": "Data Analysis",
                "required_accuracy": 0.85,
                "budget_constraint": 0.001,  # Too low
                "latency_requirement": 500,
                "privacy_requirements": "medium",
                "business_criticality": "high",
            }
        )
        
        assert response.status_code == 422
    
    def test_optimize_invalid_privacy(self, api_client: TestClient):
        """Test optimization with invalid privacy level."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            json={
                "task_category": "Data Analysis",
                "required_accuracy": 0.85,
                "budget_constraint": 5.0,
                "latency_requirement": 500,
                "privacy_requirements": "invalid",  # Invalid value
                "business_criticality": "high",
            }
        )
        
        assert response.status_code == 422
    
    def test_optimize_response_structure(
        self,
        api_client: TestClient,
        optimization_request_data: dict[str, Any]
    ):
        """Test optimization response has correct structure."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            json=optimization_request_data
        )
        
        data = response.json()
        
        # Check recommended_config structure
        config = data["recommended_config"]
        assert "agent_type" in config
        assert "model_architecture" in config
        assert "deployment_environment" in config
        assert "optimal_autonomy_level" in config
        assert "expected_cost_per_task" in config
        
        # Check expected_performance structure
        perf = data["expected_performance"]
        assert "expected_success_rate" in perf
        assert "expected_accuracy" in perf


class TestBenchmarkEndpoint:
    """Tests for performance benchmarking endpoint."""
    
    def test_benchmark_success(
        self,
        api_client: TestClient,
        benchmark_request_data: dict[str, Any]
    ):
        """Test successful benchmark request."""
        response = api_client.post(
            "/v1/performance-benchmarking",
            json=benchmark_request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "competitive_position" in data
        assert "percentile_rankings" in data
        assert "recommendations" in data
    
    def test_benchmark_invalid_success_rate(self, api_client: TestClient):
        """Test benchmark with invalid success rate."""
        response = api_client.post(
            "/v1/performance-benchmarking",
            json={
                "agent_type": "Data Analyst",
                "success_rate": 1.5,  # Invalid: > 1.0
                "accuracy_score": 0.82,
                "efficiency_score": 0.79,
                "cost_per_task_cents": 3.2,
                "response_latency_ms": 180.0,
            }
        )
        
        assert response.status_code == 422
    
    def test_benchmark_competitive_positions(
        self,
        api_client: TestClient
    ):
        """Test different competitive positions."""
        # High performer
        response = api_client.post(
            "/v1/performance-benchmarking",
            json={
                "agent_type": "Data Analyst",
                "success_rate": 0.98,
                "accuracy_score": 0.95,
                "efficiency_score": 0.92,
                "cost_per_task_cents": 1.5,
                "response_latency_ms": 100.0,
            }
        )
        
        data = response.json()
        assert data["competitive_position"] in ["Leader", "Challenger"]


class TestTradeoffEndpoint:
    """Tests for cost-performance tradeoff endpoint."""
    
    def test_tradeoff_success(self, api_client: TestClient):
        """Test successful tradeoff analysis."""
        response = api_client.post(
            "/v1/cost-performance-tradeoffs",
            json={
                "min_performance": 0.75,
                "max_cost": 10.0,
                "max_risk": 0.3,
                "optimization_priority": "balanced",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "pareto_optimal_options" in data
        assert "tradeoff_summary" in data
    
    def test_tradeoff_invalid_priority(self, api_client: TestClient):
        """Test tradeoff with invalid priority."""
        response = api_client.post(
            "/v1/cost-performance-tradeoffs",
            json={
                "min_performance": 0.75,
                "max_cost": 10.0,
                "optimization_priority": "invalid",
            }
        )
        
        assert response.status_code == 422
    
    def test_tradeoff_sweet_spot(self, api_client: TestClient):
        """Test that sweet spot recommendation is provided."""
        response = api_client.post(
            "/v1/cost-performance-tradeoffs",
            json={
                "min_performance": 0.75,
                "max_cost": 10.0,
                "optimization_priority": "balanced",
            }
        )
        
        data = response.json()
        assert "sweet_spot_recommendation" in data


class TestRiskAssessmentEndpoint:
    """Tests for failure risk assessment endpoint."""
    
    def test_risk_assessment_success(
        self,
        api_client: TestClient,
        risk_assessment_request_data: dict[str, Any]
    ):
        """Test successful risk assessment."""
        response = api_client.post(
            "/v1/failure-risk-assessment",
            json=risk_assessment_request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert "failure_probability" in data
        assert "mitigation_steps" in data
    
    def test_risk_assessment_high_risk(self, api_client: TestClient):
        """Test risk assessment with high-risk metrics."""
        response = api_client.post(
            "/v1/failure-risk-assessment",
            json={
                "agent_id": "agent-critical",
                "success_rate": 0.5,  # Very low
                "cpu_usage_percent": 95.0,  # Very high
                "memory_usage_mb": 600.0,  # Very high
                "response_latency_ms": 800.0,  # Very high
                "error_rate": 0.3,
            }
        )
        
        data = response.json()
        assert data["risk_level"] in ["High", "Critical"]
        assert data["risk_score"] > 50
    
    def test_risk_assessment_low_risk(self, api_client: TestClient):
        """Test risk assessment with low-risk metrics."""
        response = api_client.post(
            "/v1/failure-risk-assessment",
            json={
                "agent_id": "agent-stable",
                "success_rate": 0.98,
                "cpu_usage_percent": 30.0,
                "memory_usage_mb": 150.0,
                "response_latency_ms": 100.0,
                "error_rate": 0.01,
            }
        )
        
        data = response.json()
        assert data["risk_level"] in ["Low", "Medium"]
    
    def test_risk_assessment_contributing_factors(
        self,
        api_client: TestClient
    ):
        """Test that contributing factors are identified."""
        response = api_client.post(
            "/v1/failure-risk-assessment",
            json={
                "agent_id": "agent-001",
                "success_rate": 0.65,  # Low - should be flagged
                "cpu_usage_percent": 90.0,  # High - should be flagged
                "memory_usage_mb": 200.0,
                "response_latency_ms": 150.0,
            }
        )
        
        data = response.json()
        factors = data["contributing_factors"]
        
        # Should identify at least CPU and success rate issues
        factor_names = [f["factor"] for f in factors]
        assert any("CPU" in name for name in factor_names)
        assert any("Success" in name for name in factor_names)


class TestRecommendationEndpoint:
    """Tests for agent recommendation endpoint."""
    
    def test_recommendation_success(
        self,
        api_client: TestClient,
        recommendation_request_data: dict[str, Any]
    ):
        """Test successful recommendation request."""
        response = api_client.post(
            "/v1/agent-recommendation-engine",
            json=recommendation_request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "task_profile_summary" in data
    
    def test_recommendation_top_k(self, api_client: TestClient):
        """Test that top_k parameter is respected."""
        response = api_client.post(
            "/v1/agent-recommendation-engine",
            json={
                "task_category": "Data Analysis",
                "required_accuracy": 0.85,
                "required_efficiency": 0.75,
                "max_cost_cents": 5.0,
                "top_k": 2,
            }
        )
        
        data = response.json()
        assert len(data["recommendations"]) <= 2
    
    def test_recommendation_structure(
        self,
        api_client: TestClient,
        recommendation_request_data: dict[str, Any]
    ):
        """Test recommendation response structure."""
        response = api_client.post(
            "/v1/agent-recommendation-engine",
            json=recommendation_request_data
        )
        
        data = response.json()
        
        if data["recommendations"]:
            rec = data["recommendations"][0]
            assert "agent_type" in rec
            assert "model_architecture" in rec
            assert "similarity_score" in rec
            assert "avg_performance" in rec
            assert "recommendation_reason" in rec


class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_missing_required_field(self, api_client: TestClient):
        """Test error response for missing required field."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            json={
                "task_category": "Data Analysis",
                # Missing required fields
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_invalid_json(self, api_client: TestClient):
        """Test error response for invalid JSON."""
        response = api_client.post(
            "/v1/optimize-agent-configuration",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_method_not_allowed(self, api_client: TestClient):
        """Test error response for wrong HTTP method."""
        response = api_client.get("/v1/optimize-agent-configuration")
        
        assert response.status_code == 405


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers(self, api_client: TestClient):
        """Test CORS headers are present."""
        response = api_client.options(
            "/v1/optimize-agent-configuration",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )
        
        # Should allow the request (not 403)
        assert response.status_code != 403
