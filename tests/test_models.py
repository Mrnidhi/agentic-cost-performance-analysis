"""
Tests for ML models module.

This module provides comprehensive tests for model training,
prediction, and service functionality.

Course: DATA 230 (Data Visualization) at SJSU
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile

from src.models.ensemble import (
    PerformanceOptimizationEngine,
    FailurePredictionSystem,
    AgentRecommendationEngine,
)
from src.models.optimization import (
    ParetoFrontierExtractor,
    CostPerformanceTradeoffAnalyzer,
    OptimizationObjective,
    ParetoSolution,
)
from src.models.service import (
    ModelService,
    ModelRegistry,
    InstrumentedModel,
    ModelMetadata,
)
from src.core.exceptions import PredictionError, ModelLoadingError


class TestPerformanceOptimizationEngine:
    """Tests for PerformanceOptimizationEngine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = PerformanceOptimizationEngine()
        
        assert engine.feature_names == []
        assert engine.model_version is not None
    
    def test_fit(
        self,
        sample_features: np.ndarray,
        sample_targets: np.ndarray,
        feature_names: list[str]
    ):
        """Test model fitting."""
        engine = PerformanceOptimizationEngine()
        
        result = engine.fit(
            X=sample_features,
            y=sample_targets,
            feature_names=feature_names,
        )
        
        assert result is engine  # Returns self
        assert engine.feature_names == feature_names
    
    def test_predict_before_fit(self, sample_features: np.ndarray):
        """Test prediction before fitting raises error."""
        engine = PerformanceOptimizationEngine()
        
        with pytest.raises(PredictionError) as exc_info:
            engine.predict(sample_features)
        
        assert "not fitted" in str(exc_info.value).lower()
    
    def test_predict_after_fit(
        self,
        sample_features: np.ndarray,
        sample_targets: np.ndarray,
        feature_names: list[str]
    ):
        """Test prediction after fitting."""
        engine = PerformanceOptimizationEngine()
        engine.fit(sample_features, sample_targets, feature_names)
        
        predictions = engine.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert predictions.dtype == np.float64 or predictions.dtype == np.float32
    
    def test_feature_importance(
        self,
        sample_features: np.ndarray,
        sample_targets: np.ndarray,
        feature_names: list[str]
    ):
        """Test feature importance extraction."""
        engine = PerformanceOptimizationEngine()
        engine.fit(sample_features, sample_targets, feature_names)
        
        importance = engine.feature_importance
        
        assert len(importance) == len(feature_names)
        assert all(name in importance for name in feature_names)
    
    def test_predict_with_confidence(
        self,
        sample_features: np.ndarray,
        sample_targets: np.ndarray,
        feature_names: list[str]
    ):
        """Test prediction with confidence scores."""
        engine = PerformanceOptimizationEngine()
        engine.fit(sample_features, sample_targets, feature_names)
        
        predictions, confidence = engine.predict_with_confidence(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert len(confidence) == len(sample_features)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()


class TestFailurePredictionSystem:
    """Tests for FailurePredictionSystem."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = FailurePredictionSystem()
        
        assert system is not None
    
    def test_fit(
        self,
        sample_features: np.ndarray,
        feature_names: list[str]
    ):
        """Test model fitting."""
        system = FailurePredictionSystem()
        
        result = system.fit(sample_features, feature_names)
        
        assert result is system
    
    def test_detect_anomalies(
        self,
        sample_features: np.ndarray,
        feature_names: list[str]
    ):
        """Test anomaly detection."""
        system = FailurePredictionSystem()
        system.fit(sample_features, feature_names)
        
        labels = system.detect_anomalies(sample_features)
        
        assert len(labels) == len(sample_features)
        assert set(labels).issubset({-1, 1})  # Anomaly or normal
    
    def test_assess_risk(
        self,
        sample_features: np.ndarray,
        feature_names: list[str]
    ):
        """Test risk assessment."""
        system = FailurePredictionSystem()
        system.fit(sample_features, feature_names)
        
        agent_state = {name: 0.5 for name in feature_names}
        agent_state["cpu_usage_percent"] = 85  # High CPU
        agent_state["success_rate"] = 0.7  # Low success
        
        result = system.assess_risk(agent_state)
        
        assert 0 <= result.risk_score <= 100
        assert 0 <= result.failure_probability <= 1
        assert result.risk_level is not None


class TestAgentRecommendationEngine:
    """Tests for AgentRecommendationEngine."""
    
    @pytest.fixture
    def agent_profiles(self) -> pd.DataFrame:
        """Create sample agent profiles."""
        return pd.DataFrame({
            "agent_type": ["Code Assistant", "Data Analyst", "Research Assistant"],
            "model_architecture": ["GPT-4", "GPT-3.5", "Claude"],
            "success_rate": [0.92, 0.88, 0.85],
            "accuracy_score": [0.90, 0.85, 0.87],
            "efficiency_score": [0.85, 0.82, 0.80],
            "cost_per_task_cents": [4.5, 2.5, 3.0],
            "cost_efficiency_ratio": [0.78, 0.85, 0.82],
        })
    
    @pytest.fixture
    def capability_columns(self) -> list[str]:
        """Capability columns for matching."""
        return ["success_rate", "accuracy_score", "efficiency_score", "cost_efficiency_ratio"]
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = AgentRecommendationEngine()
        
        assert engine is not None
    
    def test_fit(
        self,
        agent_profiles: pd.DataFrame,
        capability_columns: list[str]
    ):
        """Test engine fitting."""
        engine = AgentRecommendationEngine()
        
        result = engine.fit(agent_profiles, capability_columns)
        
        assert result is engine
    
    def test_recommend(
        self,
        agent_profiles: pd.DataFrame,
        capability_columns: list[str]
    ):
        """Test agent recommendations."""
        engine = AgentRecommendationEngine()
        engine.fit(agent_profiles, capability_columns)
        
        task_profile = {
            "success_rate": 0.9,
            "accuracy_score": 0.85,
            "efficiency_score": 0.8,
            "cost_efficiency_ratio": 0.8,
        }
        
        recommendations = engine.recommend(task_profile, top_k=2)
        
        assert len(recommendations) == 2
        assert all(rec.similarity_score >= -1 for rec in recommendations)
        assert all(rec.similarity_score <= 1 for rec in recommendations)
    
    def test_calculate_similarity(
        self,
        agent_profiles: pd.DataFrame,
        capability_columns: list[str]
    ):
        """Test similarity calculation."""
        engine = AgentRecommendationEngine()
        engine.fit(agent_profiles, capability_columns)
        
        task_profile = {"success_rate": 0.9, "accuracy_score": 0.85, "efficiency_score": 0.8, "cost_efficiency_ratio": 0.8}
        agent_profile = {"success_rate": 0.92, "accuracy_score": 0.90, "efficiency_score": 0.85, "cost_efficiency_ratio": 0.78}
        
        similarity = engine.calculate_similarity(task_profile, agent_profile)
        
        assert -1 <= similarity <= 1


class TestParetoFrontierExtractor:
    """Tests for ParetoFrontierExtractor."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = ParetoFrontierExtractor()
        
        assert extractor is not None
    
    def test_extract_empty(self):
        """Test extraction with empty configurations."""
        extractor = ParetoFrontierExtractor()
        
        result = extractor.extract([], [])
        
        assert result == []
    
    def test_extract_single_objective(self):
        """Test extraction with single objective."""
        extractor = ParetoFrontierExtractor()
        
        configs = [
            {"performance": 0.8},
            {"performance": 0.9},
            {"performance": 0.7},
        ]
        objectives = [OptimizationObjective("performance", "maximize")]
        
        result = extractor.extract(configs, objectives)
        
        assert len(result) == 3
        # All solutions should be extracted
        performance_values = [s.objective_values["performance"] for s in result]
        assert 0.9 in performance_values
        assert 0.8 in performance_values
        assert 0.7 in performance_values
    
    def test_extract_two_objectives(self):
        """Test extraction with two objectives."""
        extractor = ParetoFrontierExtractor()
        
        configs = [
            {"performance": 0.9, "cost": 5.0},  # High perf, high cost
            {"performance": 0.7, "cost": 2.0},  # Low perf, low cost
            {"performance": 0.8, "cost": 3.0},  # Medium
        ]
        objectives = [
            OptimizationObjective("performance", "maximize"),
            OptimizationObjective("cost", "minimize"),
        ]
        
        result = extractor.extract(configs, objectives)
        
        # All configurations should be returned
        assert len(result) == 3
        
        # Verify all configurations are present in results
        performance_values = [s.objective_values["performance"] for s in result]
        assert 0.9 in performance_values
        assert 0.7 in performance_values
        assert 0.8 in performance_values


class TestCostPerformanceTradeoffAnalyzer:
    """Tests for CostPerformanceTradeoffAnalyzer."""
    
    @pytest.fixture
    def tradeoff_data(self) -> pd.DataFrame:
        """Create sample tradeoff data."""
        return pd.DataFrame({
            "business_value_score": [0.9, 0.8, 0.7, 0.85, 0.75],
            "cost_per_task_cents": [5.0, 3.0, 2.0, 4.0, 2.5],
            "operational_risk_index": [0.2, 0.3, 0.4, 0.25, 0.35],
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CostPerformanceTradeoffAnalyzer()
        
        assert analyzer is not None
    
    def test_analyze(self, tradeoff_data: pd.DataFrame):
        """Test tradeoff analysis."""
        analyzer = CostPerformanceTradeoffAnalyzer()
        
        result = analyzer.analyze(
            data=tradeoff_data,
            performance_col="business_value_score",
            cost_col="cost_per_task_cents",
            risk_col="operational_risk_index",
        )
        
        assert "pareto_solutions" in result
        assert "pareto_optimal_count" in result
        assert "sweet_spots" in result
        assert result["total_configurations"] == len(tradeoff_data)


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    @pytest.fixture
    def temp_model_dir(self) -> Path:
        """Create temporary model directory."""
        return Path(tempfile.mkdtemp())
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple model for testing."""
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    @pytest.fixture
    def sample_metadata(self) -> ModelMetadata:
        """Create sample metadata."""
        return ModelMetadata(
            name="test_model",
            version="1.0.0",
            created_at=datetime.now(),
            feature_names=tuple(["f1", "f2", "f3"]),
            target_name="target",
            model_type="LinearRegression",
            metrics={"r2": 0.85},
        )
    
    def test_initialization(self, temp_model_dir: Path):
        """Test registry initialization."""
        registry = ModelRegistry(model_dir=temp_model_dir)
        
        assert registry is not None
    
    def test_register_and_load(
        self,
        temp_model_dir: Path,
        sample_model,
        sample_metadata: ModelMetadata
    ):
        """Test model registration and loading."""
        registry = ModelRegistry(model_dir=temp_model_dir)
        
        # Register
        path = registry.register(sample_model, sample_metadata)
        assert path.exists()
        
        # Load
        loaded = registry.load(sample_metadata.name, sample_metadata.version)
        assert loaded.metadata.name == sample_metadata.name
    
    def test_load_nonexistent(self, temp_model_dir: Path):
        """Test loading nonexistent model raises error."""
        registry = ModelRegistry(model_dir=temp_model_dir)
        
        with pytest.raises(ModelLoadingError):
            registry.load("nonexistent", "1.0.0")
    
    def test_list_models(
        self,
        temp_model_dir: Path,
        sample_model,
        sample_metadata: ModelMetadata
    ):
        """Test listing registered models."""
        registry = ModelRegistry(model_dir=temp_model_dir)
        registry.register(sample_model, sample_metadata)
        
        models = registry.list_models()
        
        assert sample_metadata.name in models


class TestInstrumentedModel:
    """Tests for InstrumentedModel."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))
        return MockModel()
    
    @pytest.fixture
    def model_metadata(self) -> ModelMetadata:
        """Create model metadata."""
        return ModelMetadata(
            name="mock_model",
            version="1.0.0",
            created_at=datetime.now(),
            feature_names=tuple(["f1", "f2", "f3"]),
            target_name="target",
            model_type="MockModel",
        )
    
    def test_initialization(self, mock_model, model_metadata: ModelMetadata):
        """Test instrumented model initialization."""
        instrumented = InstrumentedModel(mock_model, model_metadata)
        
        assert instrumented.metadata == model_metadata
        assert instrumented.prediction_count == 0
    
    def test_predict(self, mock_model, model_metadata: ModelMetadata):
        """Test prediction with instrumentation."""
        instrumented = InstrumentedModel(mock_model, model_metadata)
        features = np.random.randn(10, 3)
        
        predictions = instrumented.predict(features)
        
        assert len(predictions) == 10
        assert instrumented.prediction_count == 1
    
    def test_predict_invalid_shape(self, mock_model, model_metadata: ModelMetadata):
        """Test prediction with invalid input shape."""
        instrumented = InstrumentedModel(mock_model, model_metadata)
        features = np.random.randn(10)  # 1D instead of 2D
        
        with pytest.raises(PredictionError):
            instrumented.predict(features)
    
    def test_predict_wrong_features(self, mock_model, model_metadata: ModelMetadata):
        """Test prediction with wrong number of features."""
        instrumented = InstrumentedModel(mock_model, model_metadata)
        features = np.random.randn(10, 5)  # 5 features instead of 3
        
        with pytest.raises(PredictionError):
            instrumented.predict(features)


class TestModelService:
    """Tests for ModelService."""
    
    @pytest.fixture
    def model_service(self, temp_model_dir: Path) -> ModelService:
        """Create model service with temp directory."""
        temp_dir = Path(tempfile.mkdtemp())
        registry = ModelRegistry(model_dir=temp_dir)
        return ModelService(registry=registry)
    
    @pytest.fixture
    def temp_model_dir(self) -> Path:
        """Create temporary model directory."""
        return Path(tempfile.mkdtemp())
    
    def test_initialization(self, model_service: ModelService):
        """Test service initialization."""
        assert model_service is not None
    
    def test_health_check(self, model_service: ModelService):
        """Test health check."""
        health = model_service.health_check()
        
        assert "status" in health
        assert health["status"] == "healthy"
