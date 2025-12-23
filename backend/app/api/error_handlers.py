import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Coroutine

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.exceptions import (
    PlatformError,
    ValidationError,
    InvalidTickerError,
    AgentError,
    AgentTimeoutError,
    InsufficientSignalsError,
    PartialAgentFailureError,
    ExternalServiceError,
    DataFetcherError,
    TickerNotFoundError,
    CircuitBreakerOpenError,
    MistralServiceError,
    MistralUnavailableError,
    DatabaseError,
    Neo4jError,
    CacheError,
    RateLimitError,
    AuthenticationError,
    InvalidAPIKeyError,
    ErrorSeverity,
)
from app.models.responses import ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)


def _create_error_response(
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Create a standardized error response dictionary.
    
    Args:
        error_code: Machine-readable error code.
        message: Human-readable error message.
        details: Additional error context.
        request_id: Request tracking ID.
        
    Returns:
        Error response dictionary.
    """
    response = {
        "error_code": error_code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if details:
        response["details"] = details
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def _get_request_id(request: Request) -> str | None:
    """Extract request ID from request state or headers.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        Request ID if available.
    """
    # Try to get from request state (set by middleware)
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    
    # Try to get from headers
    return request.headers.get("X-Request-ID")


def _log_error(
    request: Request,
    exc: Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    include_traceback: bool = False,
) -> None:
    """Log error with request context.
    
    Args:
        request: FastAPI request object.
        exc: Exception that occurred.
        severity: Error severity level.
        include_traceback: Whether to include full traceback.
    """
    request_id = _get_request_id(request)
    
    log_data = {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host if request.client else "unknown",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    
    log_message = (
        f"[{request_id or 'no-id'}] {request.method} {request.url.path} - "
        f"{type(exc).__name__}: {str(exc)}"
    )
    
    if severity == ErrorSeverity.CRITICAL:
        logger.critical(log_message, extra=log_data, exc_info=include_traceback)
    elif severity == ErrorSeverity.HIGH:
        logger.error(log_message, extra=log_data, exc_info=include_traceback)
    elif severity == ErrorSeverity.MEDIUM:
        logger.warning(log_message, extra=log_data, exc_info=include_traceback)
    else:
        logger.info(log_message, extra=log_data)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers.
    
    Args:
        app: FastAPI application instance.
    """
    settings = get_settings()
    
    # -------------------------------------------------------------------------
    # Platform-specific exception handlers
    # -------------------------------------------------------------------------
    
    @app.exception_handler(InvalidTickerError)
    async def invalid_ticker_handler(
        request: Request, exc: InvalidTickerError
    ) -> JSONResponse:
        """Handle invalid ticker errors."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_create_error_response(
                error_code=ErrorCode.INVALID_TICKER.value,
                message=exc.message,
                details={"ticker": exc.ticker} if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(TickerNotFoundError)
    async def ticker_not_found_handler(
        request: Request, exc: TickerNotFoundError
    ) -> JSONResponse:
        """Handle ticker not found errors."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_create_error_response(
                error_code=ErrorCode.INVALID_TICKER.value,
                message=exc.message,
                details={"ticker": exc.ticker, "source": exc.source} if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle custom validation errors."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR.value,
                message=exc.message,
                details=exc.details if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(AgentTimeoutError)
    async def agent_timeout_handler(
        request: Request, exc: AgentTimeoutError
    ) -> JSONResponse:
        """Handle agent timeout errors."""
        _log_error(request, exc, ErrorSeverity.MEDIUM)
        
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=_create_error_response(
                error_code=ErrorCode.AGENT_TIMEOUT.value,
                message=exc.message,
                details={
                    "agent": exc.agent_name,
                    "timeout_seconds": exc.timeout_seconds,
                } if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(InsufficientSignalsError)
    async def insufficient_signals_handler(
        request: Request, exc: InsufficientSignalsError
    ) -> JSONResponse:
        """Handle insufficient signals errors."""
        _log_error(request, exc, ErrorSeverity.HIGH)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.AGENT_TIMEOUT.value,
                message=f"Unable to analyze {exc.ticker}: insufficient agent responses",
                details={
                    "available_agents": exc.available_agents,
                    "failed_agents": exc.failed_agents,
                } if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(AgentError)
    async def agent_error_handler(
        request: Request, exc: AgentError
    ) -> JSONResponse:
        """Handle general agent errors."""
        _log_error(request, exc, exc.severity)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.EXTERNAL_API_ERROR.value,
                message=exc.message,
                details=exc.details if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(MistralUnavailableError)
    async def mistral_unavailable_handler(
        request: Request, exc: MistralUnavailableError
    ) -> JSONResponse:
        """Handle Mistral unavailable errors."""
        _log_error(request, exc, ErrorSeverity.HIGH)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.MISTRAL_UNAVAILABLE.value,
                message="AI analysis service is temporarily unavailable",
                request_id=_get_request_id(request),
            ),
            headers={"Retry-After": "30"},
        )
    
    @app.exception_handler(MistralServiceError)
    async def mistral_service_handler(
        request: Request, exc: MistralServiceError
    ) -> JSONResponse:
        """Handle Mistral service errors."""
        _log_error(request, exc, ErrorSeverity.HIGH)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.MISTRAL_UNAVAILABLE.value,
                message="AI analysis service error",
                details=exc.details if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(CircuitBreakerOpenError)
    async def circuit_breaker_handler(
        request: Request, exc: CircuitBreakerOpenError
    ) -> JSONResponse:
        """Handle circuit breaker open errors."""
        _log_error(request, exc, ErrorSeverity.HIGH)
        
        headers = {}
        if exc.details.get("recovery_time_seconds"):
            headers["Retry-After"] = str(int(exc.details["recovery_time_seconds"]))
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.EXTERNAL_API_ERROR.value,
                message=exc.message,
                request_id=_get_request_id(request),
            ),
            headers=headers if headers else None,
        )
    
    @app.exception_handler(DataFetcherError)
    async def data_fetcher_handler(
        request: Request, exc: DataFetcherError
    ) -> JSONResponse:
        """Handle data fetcher errors."""
        _log_error(request, exc, ErrorSeverity.MEDIUM)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.EXTERNAL_API_ERROR.value,
                message="Failed to fetch market data",
                details={"source": exc.source} if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(ExternalServiceError)
    async def external_service_handler(
        request: Request, exc: ExternalServiceError
    ) -> JSONResponse:
        """Handle external service errors."""
        _log_error(request, exc, exc.severity)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.EXTERNAL_API_ERROR.value,
                message="External service error",
                details={"service": exc.service_name} if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(Neo4jError)
    async def neo4j_error_handler(
        request: Request, exc: Neo4jError
    ) -> JSONResponse:
        """Handle Neo4j database errors."""
        _log_error(request, exc, ErrorSeverity.HIGH, include_traceback=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.NEO4J_ERROR.value,
                message="Knowledge graph service error",
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(CacheError)
    async def cache_error_handler(
        request: Request, exc: CacheError
    ) -> JSONResponse:
        """Handle cache errors - these are typically non-fatal."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        # Cache errors shouldn't fail the request, but we log them
        # This handler is mainly for cases where cache is critical
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.INTERNAL_ERROR.value,
                message="Cache service error",
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(DatabaseError)
    async def database_error_handler(
        request: Request, exc: DatabaseError
    ) -> JSONResponse:
        """Handle general database errors."""
        _log_error(request, exc, ErrorSeverity.HIGH, include_traceback=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_create_error_response(
                error_code=ErrorCode.INTERNAL_ERROR.value,
                message="Database service error",
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(
        request: Request, exc: RateLimitError
    ) -> JSONResponse:
        """Handle rate limit errors."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        headers = {}
        if exc.retry_after_seconds:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        if exc.limit:
            headers["X-RateLimit-Limit"] = str(exc.limit)
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=_create_error_response(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED.value,
                message=exc.message,
                details={
                    "retry_after_seconds": exc.retry_after_seconds,
                } if exc.retry_after_seconds else None,
                request_id=_get_request_id(request),
            ),
            headers=headers if headers else None,
        )
    
    @app.exception_handler(InvalidAPIKeyError)
    async def invalid_api_key_handler(
        request: Request, exc: InvalidAPIKeyError
    ) -> JSONResponse:
        """Handle invalid API key errors."""
        _log_error(request, exc, ErrorSeverity.MEDIUM)
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=_create_error_response(
                error_code="INVALID_API_KEY",
                message=exc.message,
                request_id=_get_request_id(request),
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        """Handle authentication errors."""
        _log_error(request, exc, ErrorSeverity.MEDIUM)
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=_create_error_response(
                error_code="AUTHENTICATION_ERROR",
                message=exc.message,
                request_id=_get_request_id(request),
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    @app.exception_handler(PlatformError)
    async def platform_error_handler(
        request: Request, exc: PlatformError
    ) -> JSONResponse:
        """Handle general platform errors."""
        _log_error(request, exc, exc.severity)
        
        # Map severity to status code
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if exc.severity == ErrorSeverity.LOW:
            status_code = status.HTTP_400_BAD_REQUEST
        elif exc.severity == ErrorSeverity.MEDIUM:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=_create_error_response(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
    
    # -------------------------------------------------------------------------
    # FastAPI/Starlette built-in exception handlers
    # -------------------------------------------------------------------------
    
    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic request validation errors."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        # Format validation errors
        errors = []
        for error in exc.errors():
            loc = " -> ".join(str(l) for l in error.get("loc", []))
            errors.append({
                "field": loc,
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR.value,
                message="Request validation failed",
                details={"errors": errors} if settings.debug else {"error_count": len(errors)},
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        severity = ErrorSeverity.LOW if exc.status_code < 500 else ErrorSeverity.HIGH
        _log_error(request, exc, severity)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=_create_error_response(
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail) if exc.detail else "HTTP error",
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """Handle value errors (often from invalid input)."""
        _log_error(request, exc, ErrorSeverity.LOW)
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR.value,
                message=str(exc),
                request_id=_get_request_id(request),
            ),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        _log_error(request, exc, ErrorSeverity.CRITICAL, include_traceback=True)
        
        # Log full traceback for debugging
        logger.error(
            f"Unhandled exception: {traceback.format_exc()}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_create_error_response(
                error_code=ErrorCode.INTERNAL_ERROR.value,
                message="An unexpected error occurred",
                details={"error": str(exc)} if settings.debug else None,
                request_id=_get_request_id(request),
            ),
        )
