import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.logging_config import (
    set_request_id,
    clear_request_id,
    APILogger,
)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and logging.
    
    Adds a unique request ID to each request for tracing across
    logs and services. Also logs request/response details.
    """
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware.
        
        Args:
            app: ASGI application.
        """
        super().__init__(app)
        self.api_logger = APILogger()
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request with tracing.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler in chain.
            
        Returns:
            Response from the handler.
        """
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        # Set request ID in context for logging
        set_request_id(request_id)
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Log request
        self.api_logger.log_request(
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            query_params=str(request.query_params) if request.query_params else None,
        )
        
        # Track timing
        start_time = time.perf_counter()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            self.api_logger.log_response(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            
            # Add tracing headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.0f}ms"
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log error response
            self.api_logger.log_response(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=duration_ms,
                error=str(e),
            )
            
            raise
            
        finally:
            # Clear request ID from context
            clear_request_id()
