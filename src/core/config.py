"""
Configuration management for AI Agent Performance Intelligence System.

This module provides centralized, validated configuration using Pydantic
for type safety and environment-based settings management.

Course: DATA 230 (Data Visualization) at SJSU
"""

from functools import lru_cache
from typing import Optional, Literal
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class BusinessMetricsConfig(BaseModel):
    """
    Immutable configuration for business metrics calculation.
    
    Weights must sum to 1.0 for proper normalization of business value scores.
    """
    
    success_rate_weight: float = Field(
        default=0.25, 
        ge=0.0, 
        le=1.0,
        description="Weight for success rate in business value calculation"
    )
    cost_efficiency_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for cost efficiency"
    )
    resource_efficiency_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for resource efficiency"
    )
    human_intervention_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for human intervention (inverted)"
    )
    error_recovery_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for error recovery rate"
    )
    privacy_compliance_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for privacy compliance"
    )
    privacy_compliance_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum privacy compliance score"
    )
    
    @model_validator(mode="after")
    def validate_weights_sum(self) -> "BusinessMetricsConfig":
        """Ensure weights sum to 1.0."""
        total = (
            self.success_rate_weight +
            self.cost_efficiency_weight +
            self.resource_efficiency_weight +
            self.human_intervention_weight +
            self.error_recovery_weight +
            self.privacy_compliance_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self

    class Config:
        frozen = True


class RiskConfig(BaseModel):
    """Configuration for risk assessment thresholds."""
    
    low_risk_threshold: float = Field(default=20.0, ge=0.0, le=100.0)
    medium_risk_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    high_risk_threshold: float = Field(default=75.0, ge=0.0, le=100.0)
    
    cpu_warning_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    memory_warning_threshold_mb: float = Field(default=400.0, gt=0.0)
    latency_warning_threshold_ms: float = Field(default=500.0, gt=0.0)
    success_rate_warning_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    anomaly_contamination: float = Field(default=0.1, ge=0.01, le=0.5)
    
    @model_validator(mode="after")
    def validate_thresholds_order(self) -> "RiskConfig":
        """Ensure risk thresholds are in ascending order."""
        if not (self.low_risk_threshold < self.medium_risk_threshold < self.high_risk_threshold):
            raise ValueError("Risk thresholds must be in ascending order")
        return self

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Configuration for ML models."""
    
    xgboost_n_estimators: int = Field(default=100, ge=10, le=1000)
    xgboost_max_depth: int = Field(default=6, ge=2, le=15)
    xgboost_learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    
    random_forest_n_estimators: int = Field(default=100, ge=10, le=500)
    
    isolation_forest_n_estimators: int = Field(default=100, ge=50, le=300)
    
    optuna_n_trials: int = Field(default=50, ge=10, le=200)
    
    model_cache_size: int = Field(default=100, ge=10, le=1000)
    prediction_timeout_seconds: float = Field(default=30.0, gt=0.0)
    
    class Config:
        frozen = True


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""
    
    temporal_window_days: int = Field(default=7, ge=1, le=30)
    performance_quartiles: int = Field(default=4, ge=2, le=10)
    cost_efficiency_tiers: int = Field(default=5, ge=3, le=10)
    
    chunk_size: int = Field(default=10000, ge=1000, le=100000)
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60)
    
    class Config:
        frozen = True


class APIConfig(BaseModel):
    """Configuration for API settings."""
    
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_request_size_bytes: int = Field(default=1_000_000, gt=0)
    
    enable_cors: bool = Field(default=True)
    cors_origins: list[str] = Field(default=["*"])
    
    enable_metrics: bool = Field(default=True)
    enable_health_check: bool = Field(default=True)
    
    class Config:
        frozen = True


class Settings(BaseSettings):
    """
    Centralized application settings with environment variable support.
    
    Settings are loaded from environment variables with fallback defaults.
    Use the get_settings() function to access the cached singleton.
    
    Environment Variables:
        ENVIRONMENT: Deployment environment (development/staging/production)
        API_HOST: API server host
        API_PORT: API server port
        DATABASE_URL: PostgreSQL connection string
        REDIS_URL: Redis connection string
        MODEL_DIR: Directory for model artifacts
        LOG_LEVEL: Logging level
    """
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(default=False)
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    
    # Database
    database_url: Optional[str] = Field(default=None)
    
    # Redis
    redis_url: Optional[str] = Field(default=None)
    
    # Model
    model_dir: Path = Field(default=Path("models"))
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    
    # Security
    secret_key: str = Field(default="change-me-in-production")
    api_key: Optional[str] = Field(default=None)
    
    # Feature flags
    enable_caching: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    enable_rate_limiting: bool = Field(default=False)
    
    # Nested configs
    business_metrics: BusinessMetricsConfig = Field(default_factory=BusinessMetricsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is a known value."""
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Warn if using default secret key in production."""
        # Access other field values through info.data
        if v == "change-me-in-production":
            import warnings
            warnings.warn(
                "Using default secret key. Set SECRET_KEY environment variable.",
                UserWarning
            )
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings singleton.
    
    Returns:
        Settings instance loaded from environment.
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.environment)
        'development'
    """
    return Settings()


def get_model_config() -> ModelConfig:
    """Get model configuration from settings."""
    return get_settings().model


def get_feature_config() -> FeatureConfig:
    """Get feature configuration from settings."""
    return get_settings().feature


def get_risk_config() -> RiskConfig:
    """Get risk configuration from settings."""
    return get_settings().risk


def get_business_metrics_config() -> BusinessMetricsConfig:
    """Get business metrics configuration from settings."""
    return get_settings().business_metrics

