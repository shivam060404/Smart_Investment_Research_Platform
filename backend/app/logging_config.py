import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional

from app.config import get_settings

# Context variable for request tracing
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context.
    
    Returns:
        Current request ID or None.
    """
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the request ID in context.
    
    Args:
        request_id: Request ID to set. Generates new UUID if not provided.
        
    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the request ID from context."""
    request_id_var.set(None)


class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging.
    
    Outputs log records as JSON objects with consistent fields for
    easy parsing and analysis by log aggregation systems.
    """
    
    def __init__(self, include_extra: bool = True):
        """Initialize the formatter.
        
        Args:
            include_extra: Whether to include extra fields from log records.
        """
        super().__init__()
        self.include_extra = include_extra
        self._skip_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "taskName",
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        # Base log structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info[2] else None,
            }
        
        # Add extra fields from record
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self._skip_fields and not key.startswith("_"):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, default=str)


class ConsoleLogFormatter(logging.Formatter):
    """Human-readable formatter for console output.
    
    Provides colored, readable log output for development environments.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console.
        
        Args:
            record: Log record to format.
            
        Returns:
            Formatted log string with colors.
        """
        # Get color for level
        color = self.COLORS.get(record.levelname, "")
        
        # Build prefix with request ID if available
        request_id = get_request_id()
        prefix = f"[{request_id}] " if request_id else ""
        
        # Format timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build message
        message = record.getMessage()
        
        # Format with color
        formatted = (
            f"{timestamp} {color}{record.levelname:8}{self.RESET} "
            f"{prefix}{record.name}: {message}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


class AgentLogger:
    """Specialized logger for agent interactions.
    
    Provides structured logging for agent analysis operations with
    consistent fields for tracking and debugging.
    """
    
    def __init__(self, agent_name: str):
        """Initialize agent logger.
        
        Args:
            agent_name: Name of the agent.
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"app.agents.{agent_name.lower()}")
    
    def log_analysis_start(self, ticker: str, **extra: Any) -> None:
        """Log the start of an analysis.
        
        Args:
            ticker: Stock ticker being analyzed.
            **extra: Additional context fields.
        """
        self.logger.info(
            f"Starting analysis for {ticker}",
            extra={
                "event": "analysis_start",
                "agent": self.agent_name,
                "ticker": ticker,
                **extra,
            }
        )
    
    def log_analysis_complete(
        self,
        ticker: str,
        score: float,
        signal_type: str,
        confidence: float,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log successful analysis completion.
        
        Args:
            ticker: Stock ticker analyzed.
            score: Analysis score.
            signal_type: Signal type (bullish/bearish/neutral).
            confidence: Confidence score.
            duration_ms: Analysis duration in milliseconds.
            **extra: Additional context fields.
        """
        self.logger.info(
            f"Analysis complete for {ticker}: {signal_type} (score={score:.1f}, confidence={confidence:.1f}%)",
            extra={
                "event": "analysis_complete",
                "agent": self.agent_name,
                "ticker": ticker,
                "score": score,
                "signal_type": signal_type,
                "confidence": confidence,
                "duration_ms": duration_ms,
                **extra,
            }
        )
    
    def log_analysis_error(
        self,
        ticker: str,
        error: Exception,
        duration_ms: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log analysis error.
        
        Args:
            ticker: Stock ticker being analyzed.
            error: Exception that occurred.
            duration_ms: Duration before error in milliseconds.
            **extra: Additional context fields.
        """
        self.logger.error(
            f"Analysis failed for {ticker}: {type(error).__name__}: {str(error)}",
            extra={
                "event": "analysis_error",
                "agent": self.agent_name,
                "ticker": ticker,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "duration_ms": duration_ms,
                **extra,
            },
            exc_info=True,
        )
    
    def log_external_call(
        self,
        service: str,
        operation: str,
        ticker: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log external service call.
        
        Args:
            service: External service name.
            operation: Operation being performed.
            ticker: Stock ticker if applicable.
            **extra: Additional context fields.
        """
        self.logger.debug(
            f"Calling {service}: {operation}",
            extra={
                "event": "external_call",
                "agent": self.agent_name,
                "service": service,
                "operation": operation,
                "ticker": ticker,
                **extra,
            }
        )
    
    def log_external_response(
        self,
        service: str,
        operation: str,
        success: bool,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log external service response.
        
        Args:
            service: External service name.
            operation: Operation performed.
            success: Whether the call succeeded.
            duration_ms: Call duration in milliseconds.
            **extra: Additional context fields.
        """
        level = logging.DEBUG if success else logging.WARNING
        status = "success" if success else "failed"
        
        self.logger.log(
            level,
            f"{service} {operation}: {status} ({duration_ms:.0f}ms)",
            extra={
                "event": "external_response",
                "agent": self.agent_name,
                "service": service,
                "operation": operation,
                "success": success,
                "duration_ms": duration_ms,
                **extra,
            }
        )


class APILogger:
    """Specialized logger for API interactions.
    
    Provides structured logging for API requests and responses.
    """
    
    def __init__(self):
        """Initialize API logger."""
        self.logger = logging.getLogger("app.api")
    
    def log_request(
        self,
        method: str,
        path: str,
        client_ip: str,
        **extra: Any,
    ) -> None:
        """Log incoming API request.
        
        Args:
            method: HTTP method.
            path: Request path.
            client_ip: Client IP address.
            **extra: Additional context fields.
        """
        self.logger.info(
            f"{method} {path}",
            extra={
                "event": "api_request",
                "method": method,
                "path": path,
                "client_ip": client_ip,
                **extra,
            }
        )
    
    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log API response.
        
        Args:
            method: HTTP method.
            path: Request path.
            status_code: Response status code.
            duration_ms: Request duration in milliseconds.
            **extra: Additional context fields.
        """
        level = logging.INFO if status_code < 400 else logging.WARNING
        
        self.logger.log(
            level,
            f"{method} {path} -> {status_code} ({duration_ms:.0f}ms)",
            extra={
                "event": "api_response",
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                **extra,
            }
        )


def configure_logging() -> None:
    """Configure application logging based on settings.
    
    Sets up structured JSON logging for production and readable
    console logging for development.
    """
    settings = get_settings()
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler based on environment
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Use JSON formatter for production, console formatter for debug
    if settings.debug:
        formatter = ConsoleLogFormatter()
    else:
        formatter = StructuredLogFormatter()
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={settings.log_level}, debug={settings.debug}",
        extra={"event": "logging_configured"}
    )


def log_function_call(logger: Optional[logging.Logger] = None):
    """Decorator to log function calls with timing.
    
    Args:
        logger: Logger to use. Uses function's module logger if not provided.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            func_name = func.__name__
            
            logger.debug(
                f"Calling {func_name}",
                extra={"event": "function_call", "function": func_name}
            )
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.debug(
                    f"{func_name} completed ({duration_ms:.0f}ms)",
                    extra={
                        "event": "function_complete",
                        "function": func_name,
                        "duration_ms": duration_ms,
                    }
                )
                return result
            except Exception as e:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.error(
                    f"{func_name} failed: {e}",
                    extra={
                        "event": "function_error",
                        "function": func_name,
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            func_name = func.__name__
            
            logger.debug(
                f"Calling {func_name}",
                extra={"event": "function_call", "function": func_name}
            )
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.debug(
                    f"{func_name} completed ({duration_ms:.0f}ms)",
                    extra={
                        "event": "function_complete",
                        "function": func_name,
                        "duration_ms": duration_ms,
                    }
                )
                return result
            except Exception as e:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.error(
                    f"{func_name} failed: {e}",
                    extra={
                        "event": "function_error",
                        "function": func_name,
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
