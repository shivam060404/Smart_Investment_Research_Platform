from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    
    LOW = "low"           # Minor issues, operation can continue
    MEDIUM = "medium"     # Partial failure, degraded functionality
    HIGH = "high"         # Significant failure, operation failed
    CRITICAL = "critical" # System-level failure


class PlatformError(Exception):
    """Base exception for all platform errors.
    
    All custom exceptions should inherit from this class to ensure
    consistent error handling and response formatting.
    
    Attributes:
        message: Human-readable error message.
        error_code: Machine-readable error code.
        details: Additional error context.
        severity: Error severity level.
        timestamp: When the error occurred.
        recoverable: Whether the operation can be retried.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "PLATFORM_ERROR",
        details: Optional[dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
    ):
        """Initialize platform error.
        
        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code.
            details: Additional error context.
            severity: Error severity level.
            recoverable: Whether the operation can be retried.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.severity = severity
        self.timestamp = datetime.utcnow()
        self.recoverable = recoverable
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API response.
        
        Returns:
            Dictionary representation of the error.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details if self.details else None,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(PlatformError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})},
            severity=ErrorSeverity.LOW,
            recoverable=True,
        )
        self.field = field


class InvalidTickerError(ValidationError):
    """Raised when a ticker symbol is invalid or not found."""
    
    def __init__(self, ticker: str, reason: Optional[str] = None):
        message = f"Invalid ticker symbol: {ticker}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message=message,
            field="ticker",
            details={"ticker": ticker, "reason": reason},
        )
        self.error_code = "INVALID_TICKER"
        self.ticker = ticker


# =============================================================================
# Agent Errors
# =============================================================================

class AgentError(PlatformError):
    """Base exception for agent-related errors."""
    
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        error_code: str = "AGENT_ERROR",
        details: Optional[dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details={"agent_name": agent_name, **(details or {})},
            severity=severity,
            recoverable=recoverable,
        )
        self.agent_name = agent_name


class AgentTimeoutError(AgentError):
    """Raised when an agent analysis times out."""
    
    def __init__(
        self,
        agent_name: str,
        timeout_seconds: float,
        ticker: Optional[str] = None,
    ):
        super().__init__(
            message=f"{agent_name} agent timed out after {timeout_seconds}s",
            agent_name=agent_name,
            error_code="AGENT_TIMEOUT",
            details={
                "timeout_seconds": timeout_seconds,
                "ticker": ticker,
            },
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
        )
        self.timeout_seconds = timeout_seconds
        self.ticker = ticker


class AgentAnalysisError(AgentError):
    """Raised when agent analysis fails."""
    
    def __init__(
        self,
        agent_name: str,
        reason: str,
        ticker: Optional[str] = None,
    ):
        super().__init__(
            message=f"{agent_name} analysis failed: {reason}",
            agent_name=agent_name,
            error_code="AGENT_ANALYSIS_ERROR",
            details={"reason": reason, "ticker": ticker},
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
        )


class InsufficientSignalsError(AgentError):
    """Raised when not enough agent signals are available for synthesis."""
    
    def __init__(
        self,
        ticker: str,
        available_agents: list[str],
        failed_agents: list[str],
        minimum_required: int = 1,
    ):
        super().__init__(
            message=f"Insufficient signals for {ticker}: {len(available_agents)} available, {minimum_required} required",
            agent_name="Orchestrator",
            error_code="INSUFFICIENT_SIGNALS",
            details={
                "ticker": ticker,
                "available_agents": available_agents,
                "failed_agents": failed_agents,
                "minimum_required": minimum_required,
            },
            severity=ErrorSeverity.HIGH,
            recoverable=True,
        )
        self.ticker = ticker
        self.available_agents = available_agents
        self.failed_agents = failed_agents


class PartialAgentFailureError(AgentError):
    """Raised when some agents fail but analysis can continue with degraded results."""
    
    def __init__(
        self,
        ticker: str,
        failed_agents: list[str],
        successful_agents: list[str],
    ):
        super().__init__(
            message=f"Partial agent failure for {ticker}: {len(failed_agents)} agents failed",
            agent_name="Orchestrator",
            error_code="PARTIAL_AGENT_FAILURE",
            details={
                "ticker": ticker,
                "failed_agents": failed_agents,
                "successful_agents": successful_agents,
            },
            severity=ErrorSeverity.LOW,
            recoverable=False,  # Already handled gracefully
        )
        self.ticker = ticker
        self.failed_agents = failed_agents
        self.successful_agents = successful_agents


# =============================================================================
# External Service Errors
# =============================================================================

class ExternalServiceError(PlatformError):
    """Base exception for external service errors."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        error_code: str = "EXTERNAL_SERVICE_ERROR",
        details: Optional[dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details={"service_name": service_name, **(details or {})},
            severity=severity,
            recoverable=recoverable,
        )
        self.service_name = service_name


class DataFetcherError(ExternalServiceError):
    """Raised when data fetching from external APIs fails."""
    
    def __init__(
        self,
        message: str,
        source: str,
        ticker: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service_name=source,
            error_code="DATA_FETCHER_ERROR",
            details={"ticker": ticker, **(details or {})},
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
        )
        self.source = source
        self.ticker = ticker


class TickerNotFoundError(DataFetcherError):
    """Raised when a ticker is not found in external data sources."""
    
    def __init__(self, ticker: str, source: str = "Unknown"):
        super().__init__(
            message=f"Ticker '{ticker}' not found",
            source=source,
            ticker=ticker,
        )
        self.error_code = "TICKER_NOT_FOUND"


class CircuitBreakerOpenError(ExternalServiceError):
    """Raised when a circuit breaker is open for an external service."""
    
    def __init__(self, service_name: str, recovery_time_seconds: Optional[float] = None):
        message = f"Service '{service_name}' is temporarily unavailable (circuit breaker open)"
        if recovery_time_seconds:
            message += f", retry in {recovery_time_seconds}s"
        
        super().__init__(
            message=message,
            service_name=service_name,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={"recovery_time_seconds": recovery_time_seconds},
            severity=ErrorSeverity.HIGH,
            recoverable=True,
        )


class MistralServiceError(ExternalServiceError):
    """Base exception for Mistral API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MISTRAL_ERROR",
        details: Optional[dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            service_name="Mistral API",
            error_code=error_code,
            details=details,
            severity=ErrorSeverity.HIGH,
            recoverable=recoverable,
        )


class MistralUnavailableError(MistralServiceError):
    """Raised when Mistral API is unavailable."""
    
    def __init__(self, reason: Optional[str] = None):
        message = "Mistral API is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            error_code="MISTRAL_UNAVAILABLE",
            details={"reason": reason},
            recoverable=True,
        )


class MistralTimeoutError(MistralServiceError):
    """Raised when Mistral API request times out."""
    
    def __init__(self, timeout_seconds: Optional[float] = None):
        message = "Mistral API request timed out"
        if timeout_seconds:
            message += f" after {timeout_seconds}s"
        super().__init__(
            message=message,
            error_code="MISTRAL_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
            recoverable=True,
        )


class MistralResponseError(MistralServiceError):
    """Raised when Mistral returns an invalid response."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid Mistral response: {reason}",
            error_code="MISTRAL_RESPONSE_ERROR",
            details={"reason": reason},
            recoverable=True,
        )


# =============================================================================
# Database Errors
# =============================================================================

class DatabaseError(PlatformError):
    """Base exception for database errors."""
    
    def __init__(
        self,
        message: str,
        database: str,
        error_code: str = "DATABASE_ERROR",
        details: Optional[dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details={"database": database, **(details or {})},
            severity=ErrorSeverity.HIGH,
            recoverable=recoverable,
        )
        self.database = database


class Neo4jError(DatabaseError):
    """Raised when Neo4j operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            database="Neo4j",
            error_code="NEO4J_ERROR",
            details={"operation": operation},
            recoverable=True,
        )
        self.operation = operation


class CacheError(DatabaseError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            database="Redis",
            error_code="CACHE_ERROR",
            details={"operation": operation},
            recoverable=True,
        )
        self.operation = operation


# =============================================================================
# Rate Limiting Errors
# =============================================================================

class RateLimitError(PlatformError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after_seconds,
            },
            severity=ErrorSeverity.LOW,
            recoverable=True,
        )
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds


# =============================================================================
# Authentication Errors
# =============================================================================

class AuthenticationError(PlatformError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
        )


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid or missing."""
    
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message=message)
        self.error_code = "INVALID_API_KEY"
