"""
Custom exceptions for AI Agent Performance Intelligence System.

This module defines a hierarchy of domain-specific exceptions that provide
clear error context and enable precise error handling throughout the application.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Any


class AgentIntelligenceError(Exception):
    """
    Base exception for all Agent Intelligence System errors.
    
    All custom exceptions inherit from this class, enabling catch-all
    handling while maintaining specific exception types.
    
    Attributes:
        message: Human-readable error description.
        details: Additional error context.
        error_code: Machine-readable error identifier.
    """
    
    def __init__(
        self, 
        message: str,
        details: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error description.
            details: Additional context as key-value pairs.
            error_code: Machine-readable error identifier.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or "AGENT_INTELLIGENCE_ERROR"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ValidationError(AgentIntelligenceError):
    """
    Raised when input validation fails.
    
    Used for invalid request parameters, out-of-range values,
    or business rule violations in input data.
    """
    
    def __init__(
        self, 
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Description of validation failure.
            field: Name of the invalid field.
            value: The invalid value received.
            constraint: The constraint that was violated.
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if constraint:
            details["constraint"] = constraint
            
        super().__init__(
            message=message,
            details=details,
            error_code="VALIDATION_ERROR"
        )
        self.field = field
        self.value = value
        self.constraint = constraint


class PredictionError(AgentIntelligenceError):
    """
    Raised when model prediction fails.
    
    Used for inference failures, invalid model state,
    or prediction pipeline errors.
    """
    
    def __init__(
        self, 
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple[int, ...]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """
        Initialize prediction error.
        
        Args:
            message: Description of prediction failure.
            model_name: Name of the failing model.
            input_shape: Shape of input that caused failure.
            cause: Underlying exception that caused this error.
        """
        details = {}
        if model_name:
            details["model_name"] = model_name
        if input_shape:
            details["input_shape"] = str(input_shape)
        if cause:
            details["cause"] = str(cause)
            
        super().__init__(
            message=message,
            details=details,
            error_code="PREDICTION_ERROR"
        )
        self.model_name = model_name
        self.input_shape = input_shape
        self.__cause__ = cause


class ModelLoadingError(AgentIntelligenceError):
    """
    Raised when model loading fails.
    
    Used for missing model files, incompatible versions,
    or corrupted model artifacts.
    """
    
    def __init__(
        self, 
        message: str,
        model_path: Optional[str] = None,
        expected_version: Optional[str] = None,
        actual_version: Optional[str] = None
    ) -> None:
        """
        Initialize model loading error.
        
        Args:
            message: Description of loading failure.
            model_path: Path to the model file.
            expected_version: Expected model version.
            actual_version: Actual model version found.
        """
        details = {}
        if model_path:
            details["model_path"] = model_path
        if expected_version:
            details["expected_version"] = expected_version
        if actual_version:
            details["actual_version"] = actual_version
            
        super().__init__(
            message=message,
            details=details,
            error_code="MODEL_LOADING_ERROR"
        )
        self.model_path = model_path


class FeatureEngineeringError(AgentIntelligenceError):
    """
    Raised when feature engineering fails.
    
    Used for missing columns, invalid transformations,
    or data quality issues during feature creation.
    """
    
    def __init__(
        self, 
        message: str,
        missing_columns: Optional[list[str]] = None,
        invalid_values: Optional[dict[str, Any]] = None,
        strategy_name: Optional[str] = None
    ) -> None:
        """
        Initialize feature engineering error.
        
        Args:
            message: Description of feature engineering failure.
            missing_columns: List of required but missing columns.
            invalid_values: Dictionary of columns with invalid values.
            strategy_name: Name of the failing strategy.
        """
        details = {}
        if missing_columns:
            details["missing_columns"] = missing_columns
        if invalid_values:
            details["invalid_values"] = invalid_values
        if strategy_name:
            details["strategy_name"] = strategy_name
            
        super().__init__(
            message=message,
            details=details,
            error_code="FEATURE_ENGINEERING_ERROR"
        )
        self.missing_columns = missing_columns or []


class ConfigurationError(AgentIntelligenceError):
    """
    Raised when configuration is invalid.
    
    Used for missing environment variables, invalid settings,
    or incompatible configuration combinations.
    """
    
    def __init__(
        self, 
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None
    ) -> None:
        """
        Initialize configuration error.
        
        Args:
            message: Description of configuration failure.
            config_key: Name of the invalid configuration key.
            expected_type: Expected type or format.
            actual_value: The invalid value received.
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)
            
        super().__init__(
            message=message,
            details=details,
            error_code="CONFIGURATION_ERROR"
        )


class RiskAssessmentError(AgentIntelligenceError):
    """
    Raised when risk assessment fails.
    
    Used for invalid agent state, missing metrics,
    or risk calculation failures.
    """
    
    def __init__(
        self, 
        message: str,
        agent_id: Optional[str] = None,
        missing_metrics: Optional[list[str]] = None
    ) -> None:
        """
        Initialize risk assessment error.
        
        Args:
            message: Description of risk assessment failure.
            agent_id: ID of the agent being assessed.
            missing_metrics: List of required but missing metrics.
        """
        details = {}
        if agent_id:
            details["agent_id"] = agent_id
        if missing_metrics:
            details["missing_metrics"] = missing_metrics
            
        super().__init__(
            message=message,
            details=details,
            error_code="RISK_ASSESSMENT_ERROR"
        )


class RecommendationError(AgentIntelligenceError):
    """
    Raised when recommendation generation fails.
    
    Used for invalid task profiles, missing agent data,
    or similarity calculation failures.
    """
    
    def __init__(
        self, 
        message: str,
        task_profile: Optional[dict[str, Any]] = None,
        available_agents: Optional[int] = None
    ) -> None:
        """
        Initialize recommendation error.
        
        Args:
            message: Description of recommendation failure.
            task_profile: The task profile that caused failure.
            available_agents: Number of available agents.
        """
        details = {}
        if task_profile:
            details["task_profile_keys"] = list(task_profile.keys())
        if available_agents is not None:
            details["available_agents"] = available_agents
            
        super().__init__(
            message=message,
            details=details,
            error_code="RECOMMENDATION_ERROR"
        )


class DataQualityError(AgentIntelligenceError):
    """
    Raised when data quality issues are detected.
    
    Used for missing values, outliers, or data integrity
    violations that prevent processing.
    """
    
    def __init__(
        self, 
        message: str,
        null_columns: Optional[list[str]] = None,
        outlier_columns: Optional[list[str]] = None,
        duplicate_count: Optional[int] = None
    ) -> None:
        """
        Initialize data quality error.
        
        Args:
            message: Description of data quality issue.
            null_columns: Columns with null values.
            outlier_columns: Columns with outliers.
            duplicate_count: Number of duplicate rows.
        """
        details = {}
        if null_columns:
            details["null_columns"] = null_columns
        if outlier_columns:
            details["outlier_columns"] = outlier_columns
        if duplicate_count is not None:
            details["duplicate_count"] = duplicate_count
            
        super().__init__(
            message=message,
            details=details,
            error_code="DATA_QUALITY_ERROR"
        )


class RateLimitError(AgentIntelligenceError):
    """
    Raised when rate limit is exceeded.
    
    Used for API throttling and resource protection.
    """
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None
    ) -> None:
        """
        Initialize rate limit error.
        
        Args:
            message: Description of rate limit violation.
            limit: Maximum requests allowed.
            window_seconds: Time window in seconds.
            retry_after: Seconds until retry is allowed.
        """
        details = {}
        if limit:
            details["limit"] = limit
        if window_seconds:
            details["window_seconds"] = window_seconds
        if retry_after:
            details["retry_after"] = retry_after
            
        super().__init__(
            message=message,
            details=details,
            error_code="RATE_LIMIT_ERROR"
        )
        self.retry_after = retry_after

