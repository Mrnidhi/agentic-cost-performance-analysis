"""
Domain types and enumerations for AI Agent Performance Intelligence System.

This module defines type-safe enums and domain-specific types used throughout
the application, ensuring consistency and compile-time type checking.

Course: DATA 230 (Data Visualization) at SJSU
"""

from enum import Enum, auto
from typing import TypeVar, Generic, NewType
from dataclasses import dataclass


class AgentType(str, Enum):
    """Type-safe enumeration of AI agent categories."""
    
    CODE_ASSISTANT = "Code Assistant"
    DATA_ANALYST = "Data Analyst"
    QA_TESTER = "QA Tester"
    RESEARCH_ASSISTANT = "Research Assistant"
    CONTENT_WRITER = "Content Writer"
    PROJECT_MANAGER = "Project Manager"
    MARKETING_ASSISTANT = "Marketing Assistant"
    CUSTOMER_SUPPORT = "Customer Support"
    
    @classmethod
    def from_string(cls, value: str) -> "AgentType":
        """
        Convert string to AgentType with fuzzy matching.
        
        Args:
            value: String representation of agent type.
            
        Returns:
            Matching AgentType enum value.
            
        Raises:
            ValueError: If no matching agent type found.
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value.lower() == normalized:
                return member
        raise ValueError(f"Unknown agent type: {value}")


class DeploymentEnvironment(str, Enum):
    """Type-safe enumeration of deployment environments."""
    
    CLOUD = "Cloud"
    EDGE = "Edge"
    HYBRID = "Hybrid"
    SERVER = "Server"
    MOBILE = "Mobile"
    
    @property
    def is_distributed(self) -> bool:
        """Check if environment supports distributed deployment."""
        return self in {self.CLOUD, self.HYBRID}
    
    @property
    def supports_edge_optimization(self) -> bool:
        """Check if environment supports edge optimization."""
        return self in {self.EDGE, self.HYBRID, self.MOBILE}


class RiskLevel(str, Enum):
    """Risk classification levels with associated thresholds."""
    
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """
        Classify risk level from numerical score.
        
        Args:
            score: Risk score between 0 and 100.
            
        Returns:
            Appropriate RiskLevel classification.
        """
        if score < 20:
            return cls.LOW
        elif score < 50:
            return cls.MEDIUM
        elif score < 75:
            return cls.HIGH
        else:
            return cls.CRITICAL
    
    @property
    def requires_immediate_action(self) -> bool:
        """Check if risk level requires immediate mitigation."""
        return self in {self.HIGH, self.CRITICAL}
    
    @property
    def color_code(self) -> str:
        """Get associated color for visualization."""
        return {
            self.LOW: "#4caf50",
            self.MEDIUM: "#ff9800",
            self.HIGH: "#f44336",
            self.CRITICAL: "#9c27b0",
        }[self]


class OptimizationPriority(str, Enum):
    """Optimization strategy priorities."""
    
    COST = "cost"
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    RISK_AVERSE = "risk_averse"
    
    @property
    def cost_weight(self) -> float:
        """Get cost weighting factor for this priority."""
        return {
            self.COST: 0.7,
            self.PERFORMANCE: 0.2,
            self.BALANCED: 0.4,
            self.RISK_AVERSE: 0.3,
        }[self]
    
    @property
    def performance_weight(self) -> float:
        """Get performance weighting factor for this priority."""
        return {
            self.COST: 0.2,
            self.PERFORMANCE: 0.7,
            self.BALANCED: 0.4,
            self.RISK_AVERSE: 0.4,
        }[self]


class StrategicImportance(str, Enum):
    """Strategic importance classification for agents."""
    
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    
    @classmethod
    def classify(cls, business_value: float, complexity: int) -> "StrategicImportance":
        """
        Classify strategic importance based on business value and complexity.
        
        Args:
            business_value: Business value score (0-1).
            complexity: Task complexity (1-10).
            
        Returns:
            Strategic importance classification.
        """
        if business_value >= 0.7 and complexity >= 7:
            return cls.CRITICAL
        elif business_value >= 0.5 and complexity >= 5:
            return cls.HIGH
        elif business_value >= 0.3:
            return cls.MEDIUM
        else:
            return cls.LOW


class CostEfficiencyTier(str, Enum):
    """Cost efficiency tier classification."""
    
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"
    
    @classmethod
    def from_percentile(cls, percentile: float) -> "CostEfficiencyTier":
        """
        Get tier from percentile ranking.
        
        Args:
            percentile: Percentile ranking (0-100).
            
        Returns:
            Corresponding cost efficiency tier.
        """
        if percentile < 20:
            return cls.VERY_LOW
        elif percentile < 40:
            return cls.LOW
        elif percentile < 60:
            return cls.MEDIUM
        elif percentile < 80:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


class CompetitivePosition(str, Enum):
    """Competitive positioning classification."""
    
    LEADER = "Leader"
    CHALLENGER = "Challenger"
    FOLLOWER = "Follower"
    LAGGARD = "Laggard"
    
    @classmethod
    def from_percentile(cls, percentile: float) -> "CompetitivePosition":
        """
        Determine competitive position from performance percentile.
        
        Args:
            percentile: Performance percentile (0-100).
            
        Returns:
            Competitive position classification.
        """
        if percentile >= 80:
            return cls.LEADER
        elif percentile >= 60:
            return cls.CHALLENGER
        elif percentile >= 40:
            return cls.FOLLOWER
        else:
            return cls.LAGGARD


# Type aliases for domain-specific values
AgentId = NewType("AgentId", str)
PerformanceScore = NewType("PerformanceScore", float)
CostCents = NewType("CostCents", float)
RiskScore = NewType("RiskScore", float)
ConfidenceScore = NewType("ConfidenceScore", float)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable container for performance metrics."""
    
    success_rate: float
    accuracy_score: float
    efficiency_score: float
    response_latency_ms: float
    cost_efficiency_ratio: float
    
    def __post_init__(self) -> None:
        """Validate metrics are within expected bounds."""
        if not 0 <= self.success_rate <= 1:
            raise ValueError(f"success_rate must be 0-1, got {self.success_rate}")
        if not 0 <= self.accuracy_score <= 1:
            raise ValueError(f"accuracy_score must be 0-1, got {self.accuracy_score}")
        if not 0 <= self.efficiency_score <= 1:
            raise ValueError(f"efficiency_score must be 0-1, got {self.efficiency_score}")
    
    @property
    def overall_performance(self) -> float:
        """Calculate weighted overall performance score."""
        return (
            self.success_rate * 0.35 +
            self.accuracy_score * 0.35 +
            self.efficiency_score * 0.30
        )


@dataclass(frozen=True)
class ResourceUsage:
    """Immutable container for resource utilization metrics."""
    
    memory_usage_mb: float
    cpu_usage_percent: float
    execution_time_seconds: float
    
    def __post_init__(self) -> None:
        """Validate resource values are positive."""
        if self.memory_usage_mb < 0:
            raise ValueError("memory_usage_mb must be non-negative")
        if not 0 <= self.cpu_usage_percent <= 100:
            raise ValueError("cpu_usage_percent must be 0-100")
        if self.execution_time_seconds < 0:
            raise ValueError("execution_time_seconds must be non-negative")
    
    @property
    def is_resource_constrained(self) -> bool:
        """Check if resource usage is above recommended thresholds."""
        return self.cpu_usage_percent > 80 or self.memory_usage_mb > 400

