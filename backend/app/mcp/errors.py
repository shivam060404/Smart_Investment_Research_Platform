import asyncio
import logging
from enum import IntEnum
from typing import Any, Callable, Optional, TypeVar

from app.exceptions import PlatformError, ErrorSeverity


logger = logging.getLogger(__name__)

T = TypeVar("T")


class MCPErrorCode(IntEnum):
    """Standard JSON-RPC and MCP error codes."""

    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom MCP errors
    SERVER_ERROR = -32000
    TIMEOUT = -32001
    RATE_LIMITED = -32002
    CONNECTION_FAILED = -32003
    SERVER_UNAVAILABLE = -32004


class MCPError(PlatformError):
    """Base exception for MCP-related errors."""

    def __init__(
        self,
        message: str,
        code: MCPErrorCode = MCPErrorCode.SERVER_ERROR,
        details: Optional[dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            error_code=f"MCP_{code.name}",
            details={"mcp_code": code.value, **(details or {})},
            severity=ErrorSeverity.MEDIUM,
            recoverable=recoverable,
        )
        self.code = code


class MCPParseError(MCPError):
    """Raised when JSON parsing fails."""

    def __init__(self, message: str = "Invalid JSON"):
        super().__init__(
            message=message,
            code=MCPErrorCode.PARSE_ERROR,
            recoverable=False,
        )


class MCPInvalidRequestError(MCPError):
    """Raised when the request is invalid."""

    def __init__(self, message: str = "Invalid JSON-RPC request"):
        super().__init__(
            message=message,
            code=MCPErrorCode.INVALID_REQUEST,
            recoverable=False,
        )


class MCPMethodNotFoundError(MCPError):
    """Raised when the requested method/tool doesn't exist."""

    def __init__(self, method: str):
        super().__init__(
            message=f"Method '{method}' not found",
            code=MCPErrorCode.METHOD_NOT_FOUND,
            details={"method": method},
            recoverable=False,
        )


class MCPInvalidParamsError(MCPError):
    """Raised when tool parameters are invalid."""

    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(
            message=message,
            code=MCPErrorCode.INVALID_PARAMS,
            details={"param": param} if param else None,
            recoverable=False,
        )


class MCPTimeoutError(MCPError):
    """Raised when an MCP request times out."""

    def __init__(self, server: str, timeout_seconds: float):
        super().__init__(
            message=f"MCP server '{server}' timed out after {timeout_seconds}s",
            code=MCPErrorCode.TIMEOUT,
            details={"server": server, "timeout_seconds": timeout_seconds},
            recoverable=True,
        )
        self.server = server
        self.timeout_seconds = timeout_seconds


class MCPRateLimitError(MCPError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        server: str,
        retry_after_seconds: Optional[float] = None,
    ):
        message = f"Rate limit exceeded for MCP server '{server}'"
        if retry_after_seconds:
            message += f", retry after {retry_after_seconds}s"
        super().__init__(
            message=message,
            code=MCPErrorCode.RATE_LIMITED,
            details={
                "server": server,
                "retry_after_seconds": retry_after_seconds,
            },
            recoverable=True,
        )
        self.server = server
        self.retry_after_seconds = retry_after_seconds


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""

    def __init__(self, server: str, reason: Optional[str] = None):
        message = f"Failed to connect to MCP server '{server}'"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            code=MCPErrorCode.CONNECTION_FAILED,
            details={"server": server, "reason": reason},
            recoverable=True,
        )
        self.server = server


class MCPServerUnavailableError(MCPError):
    """Raised when MCP server is unavailable."""

    def __init__(self, server: str):
        super().__init__(
            message=f"MCP server '{server}' is unavailable",
            code=MCPErrorCode.SERVER_UNAVAILABLE,
            details={"server": server},
            recoverable=True,
        )
        self.server = server


class MCPErrorHandler:
    """Handler for MCP errors with retry and backoff logic."""

    def __init__(
        self,
        max_retries: int = 1,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        exponential_base: float = 2.0,
    ):
        """Initialize the error handler.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay_seconds: Initial delay between retries.
            max_delay_seconds: Maximum delay between retries.
            exponential_base: Base for exponential backoff calculation.
        """
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.exponential_base = exponential_base

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff.

        Args:
            attempt: Current retry attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay_seconds * (self.exponential_base ** attempt)
        return min(delay, self.max_delay_seconds)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable.

        Args:
            error: The exception to check.

        Returns:
            True if the error can be retried.
        """
        if isinstance(error, MCPError):
            return error.recoverable and error.code in (
                MCPErrorCode.TIMEOUT,
                MCPErrorCode.RATE_LIMITED,
                MCPErrorCode.CONNECTION_FAILED,
                MCPErrorCode.SERVER_UNAVAILABLE,
                MCPErrorCode.INTERNAL_ERROR,
            )
        return False

    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "MCP operation",
    ) -> T:
        """Execute an operation with retry logic.

        Args:
            operation: Async callable to execute.
            operation_name: Name for logging purposes.

        Returns:
            Result of the operation.

        Raises:
            MCPError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e

                if not self._is_retryable(e):
                    logger.warning(
                        f"{operation_name} failed with non-retryable error: {e}"
                    )
                    raise

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)

                    # Handle rate limit with specific retry-after
                    if isinstance(e, MCPRateLimitError) and e.retry_after_seconds:
                        delay = max(delay, e.retry_after_seconds)

                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {self.max_retries + 1} attempts: {e}"
                    )

        raise last_error or MCPError("Unknown error during retry")


def mcp_error_from_code(code: int, message: str, data: Any = None) -> MCPError:
    """Create an MCPError from a JSON-RPC error code.

    Args:
        code: JSON-RPC error code.
        message: Error message.
        data: Additional error data.

    Returns:
        Appropriate MCPError subclass.
    """
    error_map = {
        MCPErrorCode.PARSE_ERROR: MCPParseError,
        MCPErrorCode.INVALID_REQUEST: MCPInvalidRequestError,
        MCPErrorCode.METHOD_NOT_FOUND: lambda: MCPMethodNotFoundError(
            data.get("method", "unknown") if isinstance(data, dict) else "unknown"
        ),
        MCPErrorCode.INVALID_PARAMS: lambda: MCPInvalidParamsError(message),
        MCPErrorCode.TIMEOUT: lambda: MCPTimeoutError(
            data.get("server", "unknown") if isinstance(data, dict) else "unknown",
            data.get("timeout_seconds", 0) if isinstance(data, dict) else 0,
        ),
        MCPErrorCode.RATE_LIMITED: lambda: MCPRateLimitError(
            data.get("server", "unknown") if isinstance(data, dict) else "unknown",
            data.get("retry_after_seconds") if isinstance(data, dict) else None,
        ),
    }

    error_class = error_map.get(code)
    if error_class:
        if callable(error_class) and not isinstance(error_class, type):
            return error_class()
        return error_class(message)

    return MCPError(message=message, code=MCPErrorCode(code) if code in MCPErrorCode._value2member_map_ else MCPErrorCode.SERVER_ERROR)
