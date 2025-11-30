"""
Protocol definitions for AI Agent Performance Intelligence System.

This module defines abstract interfaces (protocols) that enable dependency
injection, testing with mocks, and clean separation of concerns following
the Dependency Inversion Principle.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Protocol, TypeVar, Generic, Sequence, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.core.types import (
    RiskLevel,
    PerformanceMetrics,
    AgentId,
    ConfidenceScore,
)


# Generic type variables for protocol flexibility
T = TypeVar("T")
InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


class FeatureEngineeringStrategy(Protocol):
    """
    Protocol for feature engineering strategies.
    
    Implementations should transform raw agent data into engineered features
    suitable for ML model consumption. Follows Strategy pattern for
    interchangeable feature engineering approaches.
    
    Example:
        >>> class BusinessMetricsStrategy:
        ...     def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        ...         # Implementation
        ...         return engineered_data
        ...     
        ...     @property
        ...     def feature_names(self) -> list[str]:
        ...         return ["business_value_score", "operational_risk_index"]
    """
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into engineered features.
        
        Args:
            data: Raw agent performance data.
            
        Returns:
            DataFrame with engineered features added.
            
        Raises:
            FeatureEngineeringError: If transformation fails.
        """
        ...
    
    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names produced by this strategy."""
        ...
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.
        
        Args:
            data: Input DataFrame to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        ...


class PerformancePredictor(Protocol):
    """
    Protocol for performance prediction models.
    
    Implementations should predict agent performance metrics given
    input features, with support for confidence estimation.
    """
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predictions array of shape (n_samples,).
        """
        ...
    
    def predict_with_confidence(
        self, 
        features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Tuple of (predictions, confidence_scores).
        """
        ...
    
    @property
    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        ...
    
    @property
    def model_version(self) -> str:
        """Return model version identifier."""
        ...


class RiskAssessor(Protocol):
    """
    Protocol for risk assessment components.
    
    Implementations should evaluate agent risk based on current state
    and historical patterns, providing actionable risk metrics.
    """
    
    def assess_risk(self, agent_state: dict[str, Any]) -> "RiskAssessmentResult":
        """
        Assess risk for given agent state.
        
        Args:
            agent_state: Current agent metrics and state.
            
        Returns:
            Risk assessment result with score, level, and factors.
        """
        ...
    
    def detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """
        Detect anomalous patterns in feature data.
        
        Args:
            features: Feature matrix to analyze.
            
        Returns:
            Array of anomaly labels (-1 for anomaly, 1 for normal).
        """
        ...
    
    def get_risk_factors(
        self, 
        agent_state: dict[str, Any]
    ) -> list["RiskFactor"]:
        """
        Identify contributing risk factors.
        
        Args:
            agent_state: Current agent metrics.
            
        Returns:
            List of identified risk factors with impact levels.
        """
        ...


class AgentRecommender(Protocol):
    """
    Protocol for agent recommendation engines.
    
    Implementations should match task requirements to optimal agent
    configurations using similarity-based or ML-based approaches.
    """
    
    def recommend(
        self, 
        task_profile: dict[str, Any],
        top_k: int = 5
    ) -> list["AgentRecommendation"]:
        """
        Generate agent recommendations for task profile.
        
        Args:
            task_profile: Task requirements and constraints.
            top_k: Number of recommendations to return.
            
        Returns:
            Ranked list of agent recommendations.
        """
        ...
    
    def calculate_similarity(
        self, 
        task_profile: dict[str, Any],
        agent_profile: dict[str, Any]
    ) -> float:
        """
        Calculate similarity between task and agent profiles.
        
        Args:
            task_profile: Task requirements.
            agent_profile: Agent capabilities.
            
        Returns:
            Similarity score between 0 and 1.
        """
        ...


class ConfigValidator(Protocol):
    """Protocol for configuration validation."""
    
    def validate(self, config: dict[str, Any]) -> "ValidationResult":
        """
        Validate configuration against business rules.
        
        Args:
            config: Configuration dictionary to validate.
            
        Returns:
            Validation result with errors and warnings.
        """
        ...


class ModelExplainer(Protocol):
    """Protocol for model explanation/interpretability."""
    
    def explain_prediction(
        self, 
        features: np.ndarray,
        prediction: float
    ) -> "PredictionExplanation":
        """
        Generate explanation for a prediction.
        
        Args:
            features: Input features for the prediction.
            prediction: Model prediction value.
            
        Returns:
            Explanation with feature contributions.
        """
        ...
    
    def get_feature_contributions(
        self, 
        features: np.ndarray
    ) -> dict[str, float]:
        """
        Get feature contributions to prediction.
        
        Args:
            features: Input features.
            
        Returns:
            Dictionary mapping feature names to contribution values.
        """
        ...


# Result dataclasses for protocol return types

@dataclass(frozen=True)
class RiskAssessmentResult:
    """Immutable result of risk assessment."""
    
    risk_score: float
    risk_level: RiskLevel
    failure_probability: float
    contributing_factors: tuple["RiskFactor", ...]
    mitigation_steps: tuple[str, ...]
    model_version: str
    
    def __post_init__(self) -> None:
        if not 0 <= self.risk_score <= 100:
            raise ValueError(f"risk_score must be 0-100, got {self.risk_score}")
        if not 0 <= self.failure_probability <= 1:
            raise ValueError(f"failure_probability must be 0-1, got {self.failure_probability}")


@dataclass(frozen=True)
class RiskFactor:
    """Immutable risk factor description."""
    
    factor: str
    current_value: Optional[float]
    threshold: Optional[float]
    impact: str  # "low", "medium", "high"
    
    @property
    def is_exceeded(self) -> bool:
        """Check if current value exceeds threshold."""
        if self.current_value is None or self.threshold is None:
            return False
        return self.current_value > self.threshold


@dataclass(frozen=True)
class AgentRecommendation:
    """Immutable agent recommendation."""
    
    agent_type: str
    model_architecture: str
    similarity_score: float
    avg_performance: float
    avg_cost_cents: float
    recommendation_reason: str
    
    def __post_init__(self) -> None:
        if not -1 <= self.similarity_score <= 1:
            raise ValueError(f"similarity_score must be -1 to 1, got {self.similarity_score}")


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""
    
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass(frozen=True)
class PredictionExplanation:
    """Immutable prediction explanation."""
    
    prediction: float
    base_value: float
    feature_contributions: dict[str, float]
    top_positive_features: tuple[str, ...]
    top_negative_features: tuple[str, ...]
    confidence: float

