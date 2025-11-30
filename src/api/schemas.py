from pydantic import BaseModel, Field
from typing import Optional


class AgentInput(BaseModel):
    """Input schema for agent performance prediction."""
    agent_type: str = Field(..., description="Type of AI agent")
    model_architecture: str = Field(..., description="Model architecture used")
    deployment_environment: str = Field(..., description="Deployment environment")
    task_category: str = Field(..., description="Category of task")
    task_complexity: int = Field(..., ge=1, le=10, description="Task complexity (1-10)")
    autonomy_level: int = Field(..., ge=1, le=10, description="Autonomy level (1-10)")
    memory_usage_mb: float = Field(..., gt=0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., gt=0, le=100, description="CPU usage percentage")
    execution_time_seconds: float = Field(..., gt=0, description="Execution time in seconds")
    response_latency_ms: float = Field(..., gt=0, description="Response latency in ms")
    multimodal_capability: bool = Field(default=False, description="Has multimodal capability")
    edge_compatibility: bool = Field(default=False, description="Edge compatible")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_type": "Code Assistant",
                "model_architecture": "GPT-4",
                "deployment_environment": "Cloud",
                "task_category": "Code Generation",
                "task_complexity": 7,
                "autonomy_level": 5,
                "memory_usage_mb": 512.0,
                "cpu_usage_percent": 45.0,
                "execution_time_seconds": 2.5,
                "response_latency_ms": 150.0,
                "multimodal_capability": False,
                "edge_compatibility": False
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for model predictions."""
    predicted_success_rate: float = Field(..., description="Predicted success rate")
    predicted_cost_per_task: float = Field(..., description="Predicted cost per task in cents")
    recommended_model: str = Field(..., description="Recommended LLM model")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    optimization_suggestions: list[str] = Field(default=[], description="Suggestions to improve performance")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_success_rate": 0.85,
                "predicted_cost_per_task": 0.012,
                "recommended_model": "GPT-4-Turbo",
                "confidence_score": 0.92,
                "optimization_suggestions": [
                    "Consider reducing memory allocation",
                    "Task complexity could be lowered"
                ]
            }
        }

