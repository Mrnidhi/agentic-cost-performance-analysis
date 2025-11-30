"""
Pydantic schemas for API request/response validation.

This module provides comprehensive data validation using Pydantic models
with strict typing, validation rules, and example values for documentation.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# Enums for validated string choices

class TaskCategory(str, Enum):
    """Valid task categories."""
    
    TEXT_PROCESSING = "Text Processing"
    DATA_ANALYSIS = "Data Analysis"
    CREATIVE_WRITING = "Creative Writing"
    RESEARCH_SUMMARIZATION = "Research & Summarization"
    PLANNING_SCHEDULING = "Planning & Scheduling"
    CODE_GENERATION = "Code Generation"
    CUSTOMER_SUPPORT = "Customer Support"


class PrivacyLevel(str, Enum):
    """Privacy requirement levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BusinessCriticality(str, Enum):
    """Business criticality levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class CompetitivePosition(str, Enum):
    """Competitive positioning categories."""
    
    LEADER = "Leader"
    CHALLENGER = "Challenger"
    FOLLOWER = "Follower"
    LAGGARD = "Laggard"


# Base configuration models

class AgentConfig(BaseModel):
    """
    Agent configuration parameters.
    
    Represents the recommended or current configuration for an AI agent,
    including deployment settings and expected performance characteristics.
    """
    
    agent_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of AI agent",
        json_schema_extra={"example": "Code Assistant"}
    )
    model_architecture: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Model architecture (e.g., GPT-4, Claude)",
        json_schema_extra={"example": "GPT-4"}
    )
    deployment_environment: str = Field(
        ...,
        description="Deployment environment",
        json_schema_extra={"example": "Cloud"}
    )
    optimal_autonomy_level: int = Field(
        ...,
        ge=1,
        le=10,
        description="Recommended autonomy level (1-10)",
        json_schema_extra={"example": 7}
    )
    expected_cost_per_task: float = Field(
        ...,
        gt=0,
        description="Expected cost per task in cents",
        json_schema_extra={"example": 2.5}
    )

    @field_validator("deployment_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate deployment environment."""
        allowed = {"Cloud", "Edge", "Hybrid", "Server", "Mobile"}
        if v not in allowed:
            raise ValueError(f"deployment_environment must be one of {allowed}")
        return v


class PerformanceMetrics(BaseModel):
    """
    Expected performance metrics for an agent configuration.
    
    All metrics are normalized to 0-1 range where applicable.
    """
    
    expected_success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected task success rate",
        json_schema_extra={"example": 0.92}
    )
    expected_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected accuracy score",
        json_schema_extra={"example": 0.88}
    )
    expected_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected efficiency score",
        json_schema_extra={"example": 0.85}
    )
    expected_latency_ms: float = Field(
        ...,
        gt=0,
        description="Expected response latency in milliseconds",
        json_schema_extra={"example": 150.0}
    )
    expected_cost_efficiency_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected cost efficiency ratio",
        json_schema_extra={"example": 0.78}
    )
    expected_business_value_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected business value score",
        json_schema_extra={"example": 0.82}
    )


class TradeoffOption(BaseModel):
    """
    Alternative configuration option with tradeoff analysis.
    
    Represents an alternative to the primary recommendation with
    explicit tradeoffs and business impact assessment.
    """
    
    configuration: AgentConfig = Field(
        ...,
        description="Alternative agent configuration"
    )
    tradeoffs: dict[str, str] = Field(
        ...,
        description="Tradeoff descriptions (metric -> impact)",
        json_schema_extra={"example": {"cost": "20% higher", "latency": "15% lower"}}
    )
    business_impact: str = Field(
        ...,
        min_length=1,
        description="Business impact assessment",
        json_schema_extra={"example": "Higher cost but better for time-sensitive tasks"}
    )
    risk_assessment: str = Field(
        ...,
        min_length=1,
        description="Risk assessment summary",
        json_schema_extra={"example": "Low risk - well-tested configuration"}
    )


# Request models

class OptimizationRequest(BaseModel):
    """
    Request for agent configuration optimization.
    
    Specifies task requirements, constraints, and business priorities
    for finding the optimal agent configuration.
    """
    
    task_category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Category of task to optimize for",
        json_schema_extra={"example": "Data Analysis"}
    )
    required_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum required accuracy (0-1)",
        json_schema_extra={"example": 0.85}
    )
    budget_constraint: float = Field(
        ...,
        gt=0.0,
        description="Maximum cost per task in cents",
        json_schema_extra={"example": 5.0}
    )
    latency_requirement: int = Field(
        ...,
        gt=0,
        le=60000,
        description="Maximum acceptable latency in milliseconds",
        json_schema_extra={"example": 500}
    )
    privacy_requirements: PrivacyLevel = Field(
        ...,
        description="Privacy requirement level",
        json_schema_extra={"example": "medium"}
    )
    business_criticality: BusinessCriticality = Field(
        ...,
        description="Business criticality level",
        json_schema_extra={"example": "high"}
    )

    @field_validator("budget_constraint")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        """Ensure budget is reasonable."""
        if v < 0.01:
            raise ValueError("Budget constraint too low for viable operation")
        if v > 1000:
            raise ValueError("Budget constraint exceeds maximum allowed")
        return v


class BenchmarkRequest(BaseModel):
    """
    Request for performance benchmarking.
    
    Provides current agent metrics for competitive analysis
    and improvement recommendations.
    """
    
    agent_type: str = Field(
        ...,
        description="Type of agent being benchmarked",
        json_schema_extra={"example": "Data Analyst"}
    )
    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current success rate",
        json_schema_extra={"example": 0.87}
    )
    accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current accuracy score",
        json_schema_extra={"example": 0.82}
    )
    efficiency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current efficiency score",
        json_schema_extra={"example": 0.79}
    )
    cost_per_task_cents: float = Field(
        ...,
        gt=0,
        description="Current cost per task in cents",
        json_schema_extra={"example": 3.2}
    )
    response_latency_ms: float = Field(
        ...,
        gt=0,
        description="Current response latency in ms",
        json_schema_extra={"example": 180.0}
    )


class TradeoffRequest(BaseModel):
    """
    Request for cost-performance tradeoff analysis.
    
    Specifies budget and performance requirements for
    Pareto-optimal configuration discovery.
    """
    
    min_performance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable performance score",
        json_schema_extra={"example": 0.75}
    )
    max_cost: float = Field(
        ...,
        gt=0,
        description="Maximum cost per task in cents",
        json_schema_extra={"example": 10.0}
    )
    max_risk: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable risk score",
        json_schema_extra={"example": 0.3}
    )
    optimization_priority: str = Field(
        default="balanced",
        description="Optimization priority (cost/performance/balanced)",
        json_schema_extra={"example": "balanced"}
    )

    @field_validator("optimization_priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate optimization priority."""
        allowed = {"cost", "performance", "balanced", "risk_averse"}
        if v not in allowed:
            raise ValueError(f"optimization_priority must be one of {allowed}")
        return v


class RiskAssessmentRequest(BaseModel):
    """
    Request for failure risk assessment.
    
    Provides current agent state for risk analysis
    and mitigation recommendations.
    """
    
    agent_id: str = Field(
        ...,
        min_length=1,
        description="Agent identifier",
        json_schema_extra={"example": "agent-001"}
    )
    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current success rate",
        json_schema_extra={"example": 0.75}
    )
    cpu_usage_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current CPU usage percentage",
        json_schema_extra={"example": 85.0}
    )
    memory_usage_mb: float = Field(
        ...,
        ge=0.0,
        description="Current memory usage in MB",
        json_schema_extra={"example": 450.0}
    )
    response_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Current response latency in ms",
        json_schema_extra={"example": 350.0}
    )
    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current error rate",
        json_schema_extra={"example": 0.05}
    )


class RecommendationRequest(BaseModel):
    """
    Request for agent recommendations.
    
    Specifies task requirements for finding the best-matching
    agent configurations.
    """
    
    task_category: str = Field(
        ...,
        description="Task category",
        json_schema_extra={"example": "Data Analysis"}
    )
    required_accuracy: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Required accuracy level",
        json_schema_extra={"example": 0.85}
    )
    required_efficiency: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Required efficiency level",
        json_schema_extra={"example": 0.75}
    )
    max_cost_cents: float = Field(
        default=10.0,
        gt=0,
        description="Maximum cost per task",
        json_schema_extra={"example": 5.0}
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return",
        json_schema_extra={"example": 5}
    )


# Response models

class OptimizationResponse(BaseModel):
    """
    Response with optimized agent configuration.
    
    Includes the recommended configuration, expected performance,
    alternative options, and optimization insights.
    """
    
    recommended_config: AgentConfig = Field(
        ...,
        description="Recommended agent configuration"
    )
    expected_performance: PerformanceMetrics = Field(
        ...,
        description="Expected performance metrics"
    )
    alternative_options: list[TradeoffOption] = Field(
        default_factory=list,
        description="Alternative configuration options"
    )
    optimization_insights: list[str] = Field(
        default_factory=list,
        description="Optimization insights and recommendations"
    )
    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for predictions"
    )
    model_version: str = Field(
        ...,
        description="Version of the optimization model used"
    )


class BenchmarkResponse(BaseModel):
    """
    Response with benchmarking results.
    
    Includes competitive positioning, percentile rankings,
    and improvement recommendations.
    """
    
    competitive_position: CompetitivePosition = Field(
        ...,
        description="Competitive positioning"
    )
    percentile_rankings: dict[str, float] = Field(
        ...,
        description="Percentile rankings by metric"
    )
    improvement_areas: list[str] = Field(
        ...,
        description="Areas for improvement"
    )
    recommendations: list[str] = Field(
        ...,
        description="Specific recommendations"
    )
    peer_comparison: dict[str, Any] = Field(
        default_factory=dict,
        description="Comparison with peer agents"
    )


class TradeoffResponse(BaseModel):
    """
    Response with cost-performance tradeoff analysis.
    
    Includes Pareto-optimal options with business impact
    and risk assessments.
    """
    
    pareto_optimal_options: list[TradeoffOption] = Field(
        ...,
        description="Pareto-optimal configuration options"
    )
    sweet_spot_recommendation: Optional[TradeoffOption] = Field(
        None,
        description="Recommended sweet spot configuration"
    )
    tradeoff_summary: str = Field(
        ...,
        description="Summary of tradeoffs"
    )
    total_options_analyzed: int = Field(
        ...,
        description="Total configurations analyzed"
    )


class RiskFactor(BaseModel):
    """Individual risk factor."""
    
    factor: str = Field(..., description="Risk factor name")
    current_value: Optional[float] = Field(None, description="Current value")
    threshold: Optional[float] = Field(None, description="Warning threshold")
    impact: str = Field(..., description="Impact level (low/medium/high)")


class RiskAssessmentResponse(BaseModel):
    """
    Response with risk assessment results.
    
    Includes risk score, failure probability, contributing factors,
    and mitigation recommendations.
    """
    
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall risk score (0-100)"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk classification"
    )
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated failure probability"
    )
    contributing_factors: list[RiskFactor] = Field(
        ...,
        description="Contributing risk factors"
    )
    mitigation_steps: list[str] = Field(
        ...,
        description="Recommended mitigation steps"
    )
    model_version: str = Field(
        ...,
        description="Risk model version"
    )


class AgentRecommendation(BaseModel):
    """Individual agent recommendation."""
    
    agent_type: str = Field(..., description="Agent type")
    model_architecture: str = Field(..., description="Model architecture")
    similarity_score: float = Field(..., description="Similarity score")
    avg_performance: float = Field(..., description="Average performance")
    avg_cost_cents: float = Field(..., description="Average cost")
    recommendation_reason: str = Field(..., description="Recommendation reason")


class RecommendationResponse(BaseModel):
    """
    Response with agent recommendations.
    
    Includes ranked recommendations with similarity scores
    and reasoning.
    """
    
    recommendations: list[AgentRecommendation] = Field(
        ...,
        description="Ranked agent recommendations"
    )
    task_profile_summary: str = Field(
        ...,
        description="Summary of task requirements"
    )
    total_agents_evaluated: int = Field(
        ...,
        description="Total agents evaluated"
    )


class HealthResponse(BaseModel):
    """
    Health check response.
    
    Provides service health status and component information.
    """
    
    status: str = Field(
        ...,
        description="Overall health status",
        json_schema_extra={"example": "healthy"}
    )
    version: str = Field(
        ...,
        description="API version",
        json_schema_extra={"example": "1.0.0"}
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Component health statuses"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response.
    
    Provides structured error information for API errors.
    """
    
    error: str = Field(
        ...,
        description="Error code",
        json_schema_extra={"example": "VALIDATION_ERROR"}
    )
    message: str = Field(
        ...,
        description="Error message",
        json_schema_extra={"example": "Invalid input parameters"}
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing"
    )
