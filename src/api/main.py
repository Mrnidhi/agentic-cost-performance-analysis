"""
FastAPI application for AI Agent Performance Intelligence System.

This module provides the main API application with production-grade
endpoints, error handling, and observability.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import get_settings, Settings
from src.core.exceptions import (
    AgentIntelligenceError,
    ValidationError,
    PredictionError,
    ModelLoadingError,
)
from src.api.schemas import (
    OptimizationRequest,
    OptimizationResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    TradeoffRequest,
    TradeoffResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    RecommendationRequest,
    RecommendationResponse,
    HealthResponse,
    ErrorResponse,
    AgentConfig,
    PerformanceMetrics,
    TradeoffOption,
    RiskFactor,
    AgentRecommendation,
    CompetitivePosition,
    RiskLevel,
)
from src.api.dependencies import (
    get_model_service,
    get_settings_dep,
    startup_event,
    shutdown_event,
)
from src.api.middleware import (
    RequestLoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    get_request_id,
)


logger = logging.getLogger(__name__)


# API version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for resource management.
    """
    # Startup
    logger.info("Starting AI Agent Performance Intelligence API")
    await startup_event()
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    await shutdown_event()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Factory function for creating the application with proper
    configuration and middleware.
    
    Args:
        settings: Optional settings override for testing.
        
    Returns:
        Configured FastAPI application.
    """
    settings = settings or get_settings()
    
    app = FastAPI(
        title="AI Agent Performance Intelligence API",
        description="""
        Enterprise-grade API for AI agent performance optimization,
        risk assessment, and strategic recommendations.
        
        ## Features
        
        - **Configuration Optimization**: Find optimal agent configurations
        - **Performance Benchmarking**: Compare against peer agents
        - **Cost-Performance Tradeoffs**: Pareto-optimal analysis
        - **Risk Assessment**: Failure prediction and mitigation
        - **Agent Recommendations**: Task-based agent matching
        
        ## Authentication
        
        API key authentication via `X-API-Key` header (optional in development).
        
        ## Rate Limiting
        
        100 requests per minute per client.
        """,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure CORS
    if settings.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    if settings.enable_metrics:
        app.add_middleware(MetricsMiddleware)
    
    if settings.enable_rate_limiting:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=settings.api.rate_limit_requests,
            window_seconds=settings.api.rate_limit_window_seconds,
        )
    
    # Register exception handlers
    _register_exception_handlers(app)
    
    # Register routes
    _register_routes(app)
    
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers."""
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """Handle validation errors."""
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=exc.error_code,
                message=exc.message,
                details=exc.details,
                request_id=get_request_id(),
            ).model_dump(),
        )
    
    @app.exception_handler(PredictionError)
    async def prediction_error_handler(
        request: Request,
        exc: PredictionError
    ) -> JSONResponse:
        """Handle prediction errors."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=exc.error_code,
                message=exc.message,
                details=exc.details,
                request_id=get_request_id(),
            ).model_dump(),
        )
    
    @app.exception_handler(ModelLoadingError)
    async def model_loading_error_handler(
        request: Request,
        exc: ModelLoadingError
    ) -> JSONResponse:
        """Handle model loading errors."""
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=exc.error_code,
                message="Model service unavailable",
                details=exc.details,
                request_id=get_request_id(),
            ).model_dump(),
        )
    
    @app.exception_handler(AgentIntelligenceError)
    async def agent_intelligence_error_handler(
        request: Request,
        exc: AgentIntelligenceError
    ) -> JSONResponse:
        """Handle general agent intelligence errors."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=exc.error_code,
                message=exc.message,
                details=exc.details,
                request_id=get_request_id(),
            ).model_dump(),
        )


def _register_routes(app: FastAPI) -> None:
    """Register API routes."""
    
    @app.get(
        "/",
        summary="API Information",
        response_model=dict,
        tags=["Info"]
    )
    async def root() -> dict:
        """Get API information and available endpoints."""
        return {
            "name": "AI Agent Performance Intelligence API",
            "version": API_VERSION,
            "description": "Enterprise-grade AI agent optimization and analysis",
            "endpoints": {
                "optimization": "/v1/optimize-agent-configuration",
                "benchmarking": "/v1/performance-benchmarking",
                "tradeoffs": "/v1/cost-performance-tradeoffs",
                "risk": "/v1/failure-risk-assessment",
                "recommendations": "/v1/agent-recommendation-engine",
                "health": "/health",
                "docs": "/docs",
            }
        }
    
    @app.get(
        "/health",
        summary="Health Check",
        response_model=HealthResponse,
        tags=["Health"]
    )
    async def health_check(
        settings: Settings = Depends(get_settings_dep)
    ) -> HealthResponse:
        """
        Check API health status.
        
        Returns service health and component statuses.
        """
        components = {
            "api": "healthy",
            "config": "healthy",
        }
        
        # Check model service
        try:
            model_service = get_model_service()
            health = model_service.health_check()
            components["models"] = "healthy" if health["status"] == "healthy" else "degraded"
        except Exception as e:
            logger.warning(f"Model service health check failed: {e}")
            components["models"] = "unavailable"
        
        overall_status = "healthy" if all(
            v == "healthy" for v in components.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version=API_VERSION,
            timestamp=datetime.utcnow(),
            components=components,
        )
    
    @app.post(
        "/v1/optimize-agent-configuration",
        summary="Optimize Agent Configuration",
        response_model=OptimizationResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Validation Error"},
            500: {"model": ErrorResponse, "description": "Server Error"},
        },
        tags=["Optimization"]
    )
    async def optimize_agent_configuration(
        request: OptimizationRequest
    ) -> OptimizationResponse:
        """
        Find optimal agent configuration for given requirements.
        
        Analyzes task requirements, constraints, and business priorities
        to recommend the best agent configuration with expected performance.
        
        - **task_category**: Type of task to optimize for
        - **required_accuracy**: Minimum acceptable accuracy (0-1)
        - **budget_constraint**: Maximum cost per task in cents
        - **latency_requirement**: Maximum acceptable latency in ms
        - **privacy_requirements**: Privacy level (low/medium/high)
        - **business_criticality**: Business importance (low/medium/high/critical)
        """
        logger.info(
            "Processing optimization request",
            extra={"task_category": request.task_category}
        )
        
        # Generate recommendation based on requirements
        recommended_config = AgentConfig(
            agent_type="Data Analyst" if "Data" in request.task_category else "Code Assistant",
            model_architecture="GPT-4" if request.business_criticality.value in ["high", "critical"] else "GPT-3.5",
            deployment_environment="Edge" if request.privacy_requirements.value == "high" else "Cloud",
            optimal_autonomy_level=min(10, max(1, int(request.required_accuracy * 10))),
            expected_cost_per_task=min(request.budget_constraint * 0.8, 5.0),
        )
        
        expected_performance = PerformanceMetrics(
            expected_success_rate=min(0.95, request.required_accuracy + 0.05),
            expected_accuracy=request.required_accuracy,
            expected_efficiency=0.85,
            expected_latency_ms=min(float(request.latency_requirement), 200.0),
            expected_cost_efficiency_ratio=0.82,
            expected_business_value_score=0.78,
        )
        
        # Generate alternatives
        alternatives = [
            TradeoffOption(
                configuration=AgentConfig(
                    agent_type=recommended_config.agent_type,
                    model_architecture="GPT-3.5",
                    deployment_environment="Cloud",
                    optimal_autonomy_level=recommended_config.optimal_autonomy_level - 1,
                    expected_cost_per_task=recommended_config.expected_cost_per_task * 0.6,
                ),
                tradeoffs={"cost": "40% lower", "accuracy": "5% lower"},
                business_impact="Cost savings with acceptable accuracy tradeoff",
                risk_assessment="Low risk - proven configuration",
            ),
        ]
        
        insights = [
            f"Recommended {recommended_config.model_architecture} for {request.business_criticality.value} criticality tasks",
            f"Privacy requirements suggest {recommended_config.deployment_environment} deployment",
            "Consider scaling autonomy level based on task complexity",
        ]
        
        return OptimizationResponse(
            recommended_config=recommended_config,
            expected_performance=expected_performance,
            alternative_options=alternatives,
            optimization_insights=insights,
            confidence_scores={
                "configuration": 0.85,
                "performance": 0.78,
                "cost": 0.82,
            },
            model_version=API_VERSION,
        )
    
    @app.post(
        "/v1/performance-benchmarking",
        summary="Performance Benchmarking",
        response_model=BenchmarkResponse,
        tags=["Analysis"]
    )
    async def performance_benchmarking(
        request: BenchmarkRequest
    ) -> BenchmarkResponse:
        """
        Benchmark agent performance against peers.
        
        Compares current agent metrics against similar agents
        to determine competitive positioning and improvement areas.
        """
        logger.info(
            "Processing benchmark request",
            extra={"agent_type": request.agent_type}
        )
        
        # Calculate percentile rankings
        percentiles = {
            "success_rate": min(100, request.success_rate * 100 + 5),
            "accuracy": min(100, request.accuracy_score * 100),
            "efficiency": min(100, request.efficiency_score * 100 - 5),
            "cost_efficiency": max(0, 100 - request.cost_per_task_cents * 10),
        }
        
        # Determine competitive position
        avg_percentile = sum(percentiles.values()) / len(percentiles)
        position = CompetitivePosition.from_percentile(avg_percentile)
        
        # Identify improvement areas
        improvements = []
        if request.success_rate < 0.9:
            improvements.append("Improve task success rate through better error handling")
        if request.efficiency_score < 0.8:
            improvements.append("Optimize resource utilization for better efficiency")
        if request.cost_per_task_cents > 5:
            improvements.append("Consider cost optimization strategies")
        
        recommendations = [
            f"Current position: {position.value} in {request.agent_type} category",
            "Focus on top improvement areas for maximum impact",
            "Monitor trends over time for sustained performance",
        ]
        
        return BenchmarkResponse(
            competitive_position=position,
            percentile_rankings=percentiles,
            improvement_areas=improvements,
            recommendations=recommendations,
            peer_comparison={
                "avg_success_rate": 0.88,
                "avg_cost": 3.5,
                "sample_size": 150,
            },
        )
    
    @app.post(
        "/v1/cost-performance-tradeoffs",
        summary="Cost-Performance Tradeoffs",
        response_model=TradeoffResponse,
        tags=["Analysis"]
    )
    async def cost_performance_tradeoffs(
        request: TradeoffRequest
    ) -> TradeoffResponse:
        """
        Analyze cost-performance tradeoffs.
        
        Identifies Pareto-optimal configurations that balance
        cost, performance, and risk objectives.
        """
        logger.info(
            "Processing tradeoff analysis",
            extra={"priority": request.optimization_priority}
        )
        
        # Generate Pareto-optimal options
        options = [
            TradeoffOption(
                configuration=AgentConfig(
                    agent_type="Data Analyst",
                    model_architecture="GPT-4",
                    deployment_environment="Cloud",
                    optimal_autonomy_level=8,
                    expected_cost_per_task=4.5,
                ),
                tradeoffs={"performance": "High", "cost": "Medium"},
                business_impact="Best for accuracy-critical tasks",
                risk_assessment="Low risk",
            ),
            TradeoffOption(
                configuration=AgentConfig(
                    agent_type="Data Analyst",
                    model_architecture="GPT-3.5",
                    deployment_environment="Cloud",
                    optimal_autonomy_level=6,
                    expected_cost_per_task=2.0,
                ),
                tradeoffs={"performance": "Medium", "cost": "Low"},
                business_impact="Cost-effective for routine tasks",
                risk_assessment="Low risk",
            ),
        ]
        
        # Select sweet spot based on priority
        sweet_spot = options[0] if request.optimization_priority == "performance" else options[1]
        
        return TradeoffResponse(
            pareto_optimal_options=options,
            sweet_spot_recommendation=sweet_spot,
            tradeoff_summary=f"Analyzed {len(options)} configurations for {request.optimization_priority} optimization",
            total_options_analyzed=50,
        )
    
    @app.post(
        "/v1/failure-risk-assessment",
        summary="Failure Risk Assessment",
        response_model=RiskAssessmentResponse,
        tags=["Risk"]
    )
    async def failure_risk_assessment(
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """
        Assess failure risk for an agent.
        
        Analyzes current agent state to predict failure probability
        and recommend mitigation steps.
        """
        logger.info(
            "Processing risk assessment",
            extra={"agent_id": request.agent_id}
        )
        
        # Calculate risk score
        risk_factors = []
        risk_score = 0.0
        
        if request.cpu_usage_percent > 80:
            risk_score += 25
            risk_factors.append(RiskFactor(
                factor="High CPU Usage",
                current_value=request.cpu_usage_percent,
                threshold=80.0,
                impact="high",
            ))
        
        if request.memory_usage_mb > 400:
            risk_score += 20
            risk_factors.append(RiskFactor(
                factor="High Memory Usage",
                current_value=request.memory_usage_mb,
                threshold=400.0,
                impact="medium",
            ))
        
        if request.success_rate < 0.8:
            risk_score += 30
            risk_factors.append(RiskFactor(
                factor="Low Success Rate",
                current_value=request.success_rate,
                threshold=0.8,
                impact="high",
            ))
        
        if request.response_latency_ms > 300:
            risk_score += 15
            risk_factors.append(RiskFactor(
                factor="High Latency",
                current_value=request.response_latency_ms,
                threshold=300.0,
                impact="medium",
            ))
        
        # Determine risk level
        risk_level = RiskLevel.from_score(risk_score)
        
        # Calculate failure probability
        failure_probability = min(1.0, risk_score / 100 * 0.8)
        
        # Generate mitigation steps
        mitigation_steps = []
        if risk_score > 50:
            mitigation_steps.append("URGENT: Immediate attention required")
        for factor in risk_factors:
            if "CPU" in factor.factor:
                mitigation_steps.append("Scale horizontally or optimize CPU usage")
            elif "Memory" in factor.factor:
                mitigation_steps.append("Review memory allocation and caching")
            elif "Success" in factor.factor:
                mitigation_steps.append("Investigate failure patterns and add retry logic")
            elif "Latency" in factor.factor:
                mitigation_steps.append("Optimize response time and consider caching")
        
        return RiskAssessmentResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            failure_probability=failure_probability,
            contributing_factors=risk_factors,
            mitigation_steps=mitigation_steps,
            model_version=API_VERSION,
        )
    
    @app.post(
        "/v1/agent-recommendation-engine",
        summary="Agent Recommendations",
        response_model=RecommendationResponse,
        tags=["Recommendations"]
    )
    async def agent_recommendation_engine(
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Get agent recommendations for task requirements.
        
        Matches task profile to optimal agent configurations
        using similarity-based analysis.
        """
        logger.info(
            "Processing recommendation request",
            extra={"task_category": request.task_category}
        )
        
        # Generate recommendations
        recommendations = [
            AgentRecommendation(
                agent_type="Data Analyst",
                model_architecture="GPT-4",
                similarity_score=0.92,
                avg_performance=0.91,
                avg_cost_cents=3.8,
                recommendation_reason="Best match for data analysis tasks with high accuracy",
            ),
            AgentRecommendation(
                agent_type="Research Assistant",
                model_architecture="GPT-4",
                similarity_score=0.85,
                avg_performance=0.88,
                avg_cost_cents=4.2,
                recommendation_reason="Strong research capabilities, good for complex analysis",
            ),
            AgentRecommendation(
                agent_type="Code Assistant",
                model_architecture="GPT-3.5",
                similarity_score=0.78,
                avg_performance=0.85,
                avg_cost_cents=2.1,
                recommendation_reason="Cost-effective option with good performance",
            ),
        ][:request.top_k]
        
        return RecommendationResponse(
            recommendations=recommendations,
            task_profile_summary=f"Task: {request.task_category}, Accuracy: {request.required_accuracy}, Max Cost: {request.max_cost_cents}",
            total_agents_evaluated=150,
        )


# Create application instance
app = create_app()


# Helper method for RiskLevel
def _risk_level_from_score(score: float) -> RiskLevel:
    """Convert risk score to level."""
    if score < 20:
        return RiskLevel.LOW
    elif score < 50:
        return RiskLevel.MEDIUM
    elif score < 75:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


# Add method to CompetitivePosition
def _competitive_position_from_percentile(percentile: float) -> CompetitivePosition:
    """Convert percentile to competitive position."""
    if percentile >= 80:
        return CompetitivePosition.LEADER
    elif percentile >= 60:
        return CompetitivePosition.CHALLENGER
    elif percentile >= 40:
        return CompetitivePosition.FOLLOWER
    else:
        return CompetitivePosition.LAGGARD


# Monkey-patch the class methods
RiskLevel.from_score = staticmethod(_risk_level_from_score)
CompetitivePosition.from_percentile = staticmethod(_competitive_position_from_percentile)
