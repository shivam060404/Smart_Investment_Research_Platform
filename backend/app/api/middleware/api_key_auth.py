import logging
import secrets
from typing import Callable, Optional, Set

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import get_settings
from app.models.responses import ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API key authentication.
    
    Validates API key from X-API-Key header for protected endpoints.
    Excludes certain paths like /health, /docs, /redoc from authentication.
    """

    # Paths excluded from API key authentication
    EXCLUDED_PATHS: Set[str] = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(
        self,
        app: ASGIApp,
        api_keys: Optional[Set[str]] = None,
    ):
        """Initialize the API key auth middleware.
        
        Args:
            app: ASGI application.
            api_keys: Optional set of valid API keys. If not provided,
                      uses API_KEYS from settings.
        """
        super().__init__(app)
        settings = get_settings()
        
        # Load API keys from settings or use provided keys
        if api_keys is not None:
            self._api_keys = api_keys
        else:
            # Parse API keys from comma-separated string in settings
            keys_str = getattr(settings, "api_keys", "")
            if keys_str:
                self._api_keys = {k.strip() for k in keys_str.split(",") if k.strip()}
            else:
                self._api_keys = set()
        
        # Check if authentication is enabled
        self._auth_enabled = getattr(settings, "api_key_auth_enabled", True)
        
        if self._auth_enabled and not self._api_keys:
            logger.warning(
                "API key authentication is enabled but no API keys are configured. "
                "All requests to protected endpoints will be rejected."
            )

    def _should_authenticate(self, path: str) -> bool:
        """Check if path requires authentication.
        
        Args:
            path: Request path.
            
        Returns:
            True if path requires authentication.
        """
        # Check exact match first
        if path in self.EXCLUDED_PATHS:
            return False
        
        # Check if path starts with any excluded prefix
        for excluded in self.EXCLUDED_PATHS:
            if path.startswith(excluded):
                return False
        
        return True


    def _validate_api_key(self, api_key: Optional[str]) -> bool:
        """Validate the provided API key.
        
        Uses constant-time comparison to prevent timing attacks.
        
        Args:
            api_key: API key to validate.
            
        Returns:
            True if API key is valid.
        """
        if not api_key:
            return False
        
        if not self._api_keys:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        for valid_key in self._api_keys:
            if secrets.compare_digest(api_key, valid_key):
                return True
        
        return False

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response]
    ) -> Response:
        """Process request through API key authentication.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler in chain.
            
        Returns:
            Response from handler or authentication error response.
        """
        # Skip authentication if disabled
        if not self._auth_enabled:
            return await call_next(request)
        
        # Skip authentication for excluded paths
        if not self._should_authenticate(request.url.path):
            return await call_next(request)
        
        # Extract API key from header
        api_key = request.headers.get("X-API-Key")
        
        # Validate API key
        if not self._validate_api_key(api_key):
            logger.warning(
                f"Unauthorized request to {request.url.path} from "
                f"{request.client.host if request.client else 'unknown'}"
            )
            error_response = ErrorResponse(
                error_code=ErrorCode.VALIDATION_ERROR,
                message="Invalid or missing API key. Provide a valid X-API-Key header."
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(mode="json"),
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        # API key is valid, proceed with request
        return await call_next(request)


# Global middleware instance
_api_key_middleware: Optional[APIKeyAuthMiddleware] = None


def get_api_key_middleware() -> APIKeyAuthMiddleware:
    """Get the global API key middleware instance.
    
    Returns:
        APIKeyAuthMiddleware instance.
    """
    global _api_key_middleware
    if _api_key_middleware is None:
        _api_key_middleware = APIKeyAuthMiddleware(app=None)
    return _api_key_middleware
