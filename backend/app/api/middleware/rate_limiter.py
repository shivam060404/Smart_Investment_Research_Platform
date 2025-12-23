import logging
import time
from typing import Callable, Optional

import redis.asyncio as redis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import get_settings
from app.models.responses import ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based distributed rate limiter using sliding window algorithm."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        requests_per_window: Optional[int] = None,
        window_seconds: Optional[int] = None
    ):
        """Initialize the rate limiter.
        
        Args:
            redis_url: Redis connection URL. Defaults to settings value.
            requests_per_window: Max requests allowed per window. Defaults to settings.
            window_seconds: Window duration in seconds. Defaults to settings.
        """
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._requests_per_window = requests_per_window or settings.api_rate_limit
        self._window_seconds = window_seconds or settings.rate_limit_window_seconds
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is None:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Rate limiter connected to Redis")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Rate limiter disconnected from Redis")

    async def _ensure_connected(self) -> redis.Redis:
        """Ensure Redis connection is established."""
        if self._client is None:
            await self.connect()
        return self._client

    def _get_client_identifier(self, request: Request) -> str:
        """Extract client identifier from request.
        
        Uses X-Forwarded-For header if present (for proxied requests),
        otherwise falls back to client host.
        
        Args:
            request: FastAPI request object.
            
        Returns:
            Client identifier string.
        """
        # Check for forwarded header (common in proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Check for X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client host
        if request.client:
            return request.client.host
        
        return "unknown"

    def _generate_key(self, client_id: str) -> str:
        """Generate Redis key for rate limiting.
        
        Args:
            client_id: Client identifier.
            
        Returns:
            Redis key string.
        """
        return f"rate_limit:{client_id}"

    async def is_allowed(self, request: Request) -> tuple[bool, int, int]:
        """Check if request is allowed under rate limit.
        
        Uses a sliding window counter algorithm for accurate rate limiting.
        
        Args:
            request: FastAPI request object.
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time_seconds).
        """
        try:
            client = await self._ensure_connected()
            client_id = self._get_client_identifier(request)
            key = self._generate_key(client_id)
            
            current_time = int(time.time())
            window_start = current_time - self._window_seconds
            
            # Use Redis pipeline for atomic operations
            pipe = client.pipeline()
            
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request with timestamp as score
            pipe.zadd(key, {f"{current_time}:{id(request)}": current_time})
            
            # Set expiry on the key
            pipe.expire(key, self._window_seconds)
            
            results = await pipe.execute()
            
            # results[1] is the count before adding current request
            current_count = results[1]
            
            remaining = max(0, self._requests_per_window - current_count - 1)
            reset_time = self._window_seconds
            
            if current_count >= self._requests_per_window:
                # Remove the request we just added since it's over limit
                await client.zrem(key, f"{current_time}:{id(request)}")
                logger.warning(f"Rate limit exceeded for client {client_id}")
                return False, 0, reset_time
            
            return True, remaining, reset_time
            
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {e}")
            # Fail open - allow request if Redis is unavailable
            return True, self._requests_per_window, self._window_seconds
        except Exception as e:
            logger.error(f"Unexpected error in rate limiter: {e}")
            # Fail open
            return True, self._requests_per_window, self._window_seconds

    async def get_usage(self, request: Request) -> tuple[int, int]:
        """Get current rate limit usage for a client.
        
        Args:
            request: FastAPI request object.
            
        Returns:
            Tuple of (current_count, limit).
        """
        try:
            client = await self._ensure_connected()
            client_id = self._get_client_identifier(request)
            key = self._generate_key(client_id)
            
            current_time = int(time.time())
            window_start = current_time - self._window_seconds
            
            # Clean old entries and count
            await client.zremrangebyscore(key, 0, window_start)
            count = await client.zcard(key)
            
            return count, self._requests_per_window
            
        except Exception as e:
            logger.error(f"Error getting rate limit usage: {e}")
            return 0, self._requests_per_window


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting requests."""

    # Paths excluded from rate limiting
    EXCLUDED_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """Initialize the rate limit middleware.
        
        Args:
            app: ASGI application.
            rate_limiter: Optional RateLimiter instance. Creates new one if not provided.
        """
        super().__init__(app)
        self._rate_limiter = rate_limiter or RateLimiter()

    def _should_rate_limit(self, path: str) -> bool:
        """Check if path should be rate limited.
        
        Args:
            path: Request path.
            
        Returns:
            True if path should be rate limited.
        """
        return path not in self.EXCLUDED_PATHS

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response]
    ) -> Response:
        """Process request through rate limiter.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler in chain.
            
        Returns:
            Response from handler or rate limit error response.
        """
        # Skip rate limiting for excluded paths
        if not self._should_rate_limit(request.url.path):
            return await call_next(request)
        
        # Check rate limit
        is_allowed, remaining, reset_time = await self._rate_limiter.is_allowed(request)
        
        if not is_allowed:
            error_response = ErrorResponse(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message=f"Rate limit exceeded. Try again in {reset_time} seconds."
            )
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response.model_dump(mode="json")
            )
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self._rate_limiter._requests_per_window)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(reset_time)
            response.headers["Retry-After"] = str(reset_time)
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(self._rate_limiter._requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.
    
    Returns:
        RateLimiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
