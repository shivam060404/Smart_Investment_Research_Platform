import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware.rate_limiter import RateLimitMiddleware
from app.api.middleware.api_key_auth import APIKeyAuthMiddleware
from app.api.middleware.request_tracing import RequestTracingMiddleware
from app.api.error_handlers import register_exception_handlers
from app.config import get_settings
from app.logging_config import configure_logging
from app.models.responses import ErrorCode, ErrorResponse
from app.services.cache_service import get_cache_service

# Configure structured logging
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting AI Investment Research Platform...")
    settings = get_settings()
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize cache service connection
    cache_service = get_cache_service()
    try:
        await cache_service.connect()
        logger.info("Cache service connected")
    except Exception as e:
        logger.warning(f"Failed to connect cache service: {e}")
    
    # Initialize MCP servers if enabled
    if settings.mcp_enabled:
        try:
            from app.mcp.registry import get_mcp_registry
            
            registry = get_mcp_registry()
            config_path = settings.get_mcp_config_path()
            
            # Load configuration from YAML file
            registry.load_config_from_yaml(config_path)
            
            # Connect to all configured servers
            results = await registry.connect_all()
            
            connected = sum(1 for v in results.values() if v)
            total = len(results)
            logger.info(f"MCP servers initialized: {connected}/{total} connected")
            
            for name, success in results.items():
                if success:
                    logger.info(f"MCP server '{name}' connected")
                else:
                    logger.warning(f"MCP server '{name}' failed to connect")
                    
        except Exception as e:
            logger.error(f"Failed to initialize MCP servers: {e}")
    else:
        logger.info("MCP servers disabled (MCP_ENABLED=false)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Investment Research Platform...")
    
    # Disconnect MCP servers if enabled
    if settings.mcp_enabled:
        try:
            from app.mcp.registry import get_mcp_registry
            registry = get_mcp_registry()
            await registry.disconnect_all()
            logger.info("MCP servers disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting MCP servers: {e}")
    
    # Disconnect cache service
    try:
        await cache_service.disconnect()
        logger.info("Cache service disconnected")
    except Exception as e:
        logger.warning(f"Error disconnecting cache service: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Multi-agent AI system for stock analysis and investment recommendations",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Add request tracing middleware (must be early in chain)
    app.add_middleware(RequestTracingMiddleware)

    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Add API key authentication middleware
    app.add_middleware(APIKeyAuthMiddleware)

    # Register comprehensive exception handlers
    register_exception_handlers(app)

    # Include API routers
    register_routers(app)

    return app


def register_routers(app: FastAPI) -> None:
    """Register API routers.
    
    Args:
        app: FastAPI application instance.
    """
    from app.api.routes import analyze, backtest, health, chat

    # Health check endpoint (no prefix, no auth required)
    app.include_router(health.router, tags=["Health"])

    # API endpoints
    app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
    app.include_router(backtest.router, prefix="/api", tags=["Backtest"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])


# Create the application instance
app = create_app()
