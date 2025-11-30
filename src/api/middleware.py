"""
Custom middleware for FastAPI application.

This module provides middleware for logging, metrics, rate limiting,
and request/response processing.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Callable, Optional
import time
import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException

from src.core.config import get_settings
from src.core.exceptions import RateLimitError


logger = logging.getLogger(__name__)


# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.
    
    Logs request details, response status, and timing information
    with correlation IDs for tracing.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with logging."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request_id_var.set(request_id)
        
        # Extract request details
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
            }
        )
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_seconds": round(duration, 4),
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "error": str(e),
                    "duration_seconds": round(duration, 4),
                },
                exc_info=True
            )
            
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    
    Tracks request counts, latencies, and error rates
    for monitoring and alerting.
    """
    
    def __init__(self, app, enable_prometheus: bool = True):
        """
        Initialize metrics middleware.
        
        Args:
            app: FastAPI application.
            enable_prometheus: Whether to use Prometheus metrics.
        """
        super().__init__(app)
        self._enable_prometheus = enable_prometheus
        self._request_count = 0
        self._error_count = 0
        self._latencies: list[float] = []
        
        if enable_prometheus:
            try:
                from prometheus_client import Counter, Histogram
                
                self._http_requests = Counter(
                    "http_requests_total",
                    "Total HTTP requests",
                    ["method", "path", "status"]
                )
                self._http_latency = Histogram(
                    "http_request_duration_seconds",
                    "HTTP request latency",
                    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
                )
            except ImportError:
                self._enable_prometheus = False
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with metrics collection."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = str(response.status_code)
            
        except Exception as e:
            status = "500"
            self._error_count += 1
            raise
            
        finally:
            duration = time.time() - start_time
            self._request_count += 1
            self._latencies.append(duration)
            
            if self._enable_prometheus:
                self._http_requests.labels(
                    method=request.method,
                    path=request.url.path,
                    status=status
                ).inc()
                self._http_latency.observe(duration)
        
        return response
    
    @property
    def stats(self) -> dict:
        """Get current metrics stats."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "avg_latency": sum(self._latencies) / len(self._latencies) if self._latencies else 0,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    
    Implements token bucket algorithm for request throttling.
    """
    
    def __init__(
        self,
        app,
        requests_per_window: int = 100,
        window_seconds: int = 60
    ):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application.
            requests_per_window: Max requests per time window.
            window_seconds: Time window in seconds.
        """
        super().__init__(app)
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._request_counts: dict[str, list[float]] = {}
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with rate limiting."""
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            logger.warning(
                "Rate limit exceeded",
                extra={"client_id": client_id}
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(self._window_seconds)}
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use API key if present, otherwise IP address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        if request.client:
            return f"ip:{request.client.host}"
        
        return "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        window_start = current_time - self._window_seconds
        
        # Get request timestamps for client
        if client_id not in self._request_counts:
            self._request_counts[client_id] = []
        
        # Remove old timestamps
        self._request_counts[client_id] = [
            ts for ts in self._request_counts[client_id]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._request_counts[client_id]) >= self._requests_per_window:
            return False
        
        # Record this request
        self._request_counts[client_id].append(current_time)
        
        return True


def get_request_id() -> str:
    """
    Get the current request ID.
    
    Returns:
        Current request ID or empty string if not in request context.
    """
    return request_id_var.get()

