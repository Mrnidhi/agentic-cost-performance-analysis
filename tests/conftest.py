"""
Pytest configuration and fixtures for test suite.

This module provides shared fixtures, test data factories, and
configuration for all tests.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Generator, Any
from datetime import datetime
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.core.config import Settings, BusinessMetricsConfig, ModelConfig
from src.api.main import create_app
from src.api.dependencies import reset_singletons


# Test data constants
SAMPLE_AGENT_TYPES = [
    "Code Assistant",
    "Data Analyst",
    "Research Assistant",
    "Content Writer",
]

SAMPLE_ARCHITECTURES = [
    "GPT-4",
    "GPT-3.5",
    "Claude",
    "Llama",
]

SAMPLE_ENVIRONMENTS = [
    "Cloud",
    "Edge",
    "Hybrid",
    "Server",
]


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        environment="development",
        debug=True,
        api_host="127.0.0.1",
        api_port=8000,
        model_dir=Path(tempfile.mkdtemp()),
        log_level="DEBUG",
        enable_caching=False,
        enable_metrics=False,
        enable_rate_limiting=False,
    )


@pytest.fixture
def api_client(test_settings: Settings) -> Generator[TestClient, None, None]:
    """Create test client for API testing."""
    reset_singletons()
    app = create_app(settings=test_settings)
    
    with TestClient(app) as client:
        yield client
    
    reset_singletons()


@pytest.fixture
def sample_agent_data() -> pd.DataFrame:
    """Generate sample agent data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "agent_id": [f"agent_{i:03d}" for i in range(n_samples)],
        "timestamp": pd.date_range(
            start="2024-01-01",
            periods=n_samples,
            freq="H"
        ),
        "agent_type": np.random.choice(SAMPLE_AGENT_TYPES, n_samples),
        "model_architecture": np.random.choice(SAMPLE_ARCHITECTURES, n_samples),
        "deployment_environment": np.random.choice(SAMPLE_ENVIRONMENTS, n_samples),
        "task_category": np.random.choice(
            ["Data Analysis", "Code Generation", "Research"],
            n_samples
        ),
        "task_complexity": np.random.randint(1, 11, n_samples),
        "autonomy_level": np.random.randint(1, 11, n_samples),
        "success_rate": np.random.uniform(0.7, 1.0, n_samples),
        "accuracy_score": np.random.uniform(0.7, 1.0, n_samples),
        "efficiency_score": np.random.uniform(0.6, 1.0, n_samples),
        "response_latency_ms": np.random.uniform(50, 500, n_samples),
        "execution_time_seconds": np.random.uniform(1, 60, n_samples),
        "memory_usage_mb": np.random.uniform(100, 500, n_samples),
        "cpu_usage_percent": np.random.uniform(10, 90, n_samples),
        "cost_per_task_cents": np.random.uniform(0.5, 10, n_samples),
        "cost_efficiency_ratio": np.random.uniform(0.5, 1.0, n_samples),
        "human_intervention_required": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "error_recovery_rate": np.random.uniform(0.6, 1.0, n_samples),
        "privacy_compliance_score": np.random.uniform(0.8, 1.0, n_samples),
        "data_quality_score": np.random.uniform(0.7, 1.0, n_samples),
    })


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature array for model testing."""
    np.random.seed(42)
    return np.random.randn(50, 10)


@pytest.fixture
def sample_targets() -> np.ndarray:
    """Generate sample target array for model testing."""
    np.random.seed(42)
    return np.random.uniform(0, 1, 50)


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names for model testing."""
    return [
        "success_rate",
        "accuracy_score",
        "efficiency_score",
        "cost_efficiency_ratio",
        "task_complexity",
        "autonomy_level",
        "cpu_usage_percent",
        "memory_usage_mb",
        "error_recovery_rate",
        "privacy_compliance_score",
    ]


@pytest.fixture
def optimization_request_data() -> dict[str, Any]:
    """Sample optimization request data."""
    return {
        "task_category": "Data Analysis",
        "required_accuracy": 0.85,
        "budget_constraint": 5.0,
        "latency_requirement": 500,
        "privacy_requirements": "medium",
        "business_criticality": "high",
    }


@pytest.fixture
def benchmark_request_data() -> dict[str, Any]:
    """Sample benchmark request data."""
    return {
        "agent_type": "Data Analyst",
        "success_rate": 0.87,
        "accuracy_score": 0.82,
        "efficiency_score": 0.79,
        "cost_per_task_cents": 3.2,
        "response_latency_ms": 180.0,
    }


@pytest.fixture
def risk_assessment_request_data() -> dict[str, Any]:
    """Sample risk assessment request data."""
    return {
        "agent_id": "agent-001",
        "success_rate": 0.75,
        "cpu_usage_percent": 85.0,
        "memory_usage_mb": 450.0,
        "response_latency_ms": 350.0,
        "error_rate": 0.05,
    }


@pytest.fixture
def recommendation_request_data() -> dict[str, Any]:
    """Sample recommendation request data."""
    return {
        "task_category": "Data Analysis",
        "required_accuracy": 0.85,
        "required_efficiency": 0.75,
        "max_cost_cents": 5.0,
        "top_k": 5,
    }


@pytest.fixture
def mock_model_metadata() -> dict[str, Any]:
    """Mock model metadata for testing."""
    return {
        "name": "test_model",
        "version": "1.0.0",
        "created_at": datetime.now(),
        "feature_names": tuple([f"feature_{i}" for i in range(10)]),
        "target_name": "business_value_score",
        "model_type": "XGBRegressor",
        "metrics": {"r2": 0.85, "rmse": 0.12},
    }


class AgentDataFactory:
    """Factory for generating test agent data."""
    
    @staticmethod
    def build(
        n_samples: int = 1,
        success_rate: float = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Build agent data with optional overrides.
        
        Args:
            n_samples: Number of samples to generate.
            success_rate: Optional fixed success rate.
            **kwargs: Additional column overrides.
            
        Returns:
            DataFrame with agent data.
        """
        np.random.seed(42)
        
        data = {
            "agent_id": [f"agent_{i:03d}" for i in range(n_samples)],
            "success_rate": (
                [success_rate] * n_samples
                if success_rate is not None
                else np.random.uniform(0.7, 1.0, n_samples)
            ),
            "accuracy_score": np.random.uniform(0.7, 1.0, n_samples),
            "efficiency_score": np.random.uniform(0.6, 1.0, n_samples),
            "cost_efficiency_ratio": np.random.uniform(0.5, 1.0, n_samples),
            "task_complexity": np.random.randint(1, 11, n_samples),
        }
        
        data.update(kwargs)
        
        return pd.DataFrame(data)


@pytest.fixture
def agent_data_factory() -> type[AgentDataFactory]:
    """Provide agent data factory."""
    return AgentDataFactory
