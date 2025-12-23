import logging
from datetime import datetime

from fastapi import APIRouter

from app.config import get_settings
from app.models.responses import HealthResponse, ServiceStatus
from app.services.cache_service import get_cache_service
from app.services.neo4j_service import get_neo4j_service
from app.services.mistral_service import get_mistral_service
from app.mcp.health import get_mcp_health_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the system and all dependent services.",
)
async def health_check() -> HealthResponse:
    """Check system health status.
    
    Performs connectivity checks against all dependent services:
    - Neo4j Knowledge Graph
    - Redis Cache
    - Mistral AI API
    - MCP Servers (YFinance, News, Technical, Neo4j, Backtesting)
    
    Returns:
        HealthResponse: Current health status of the system and its dependencies.
    """
    settings = get_settings()
    start_time = datetime.utcnow()
    
    # Track service statuses
    services: dict[str, ServiceStatus] = {}
    unhealthy_count = 0
    
    # Check Redis health
    try:
        cache_service = get_cache_service()
        redis_healthy, redis_latency = await cache_service.health_check()
        services["redis"] = ServiceStatus(
            name="Redis",
            status="healthy" if redis_healthy else "unhealthy",
            latency_ms=redis_latency,
            message="Connected" if redis_healthy else "Connection failed"
        )
        if not redis_healthy:
            unhealthy_count += 1
    except Exception as e:
        logger.error(f"Redis health check error: {e}")
        services["redis"] = ServiceStatus(
            name="Redis",
            status="unhealthy",
            latency_ms=None,
            message=f"Health check failed: {str(e)}"
        )
        unhealthy_count += 1
    
    # Check Neo4j health
    try:
        neo4j_service = get_neo4j_service()
        neo4j_result = await neo4j_service.health_check()
        services["neo4j"] = ServiceStatus(
            name="Neo4j",
            status=neo4j_result.get("status", "unknown"),
            latency_ms=neo4j_result.get("latency_ms"),
            message=neo4j_result.get("message", "")
        )
        if neo4j_result.get("status") != "healthy":
            unhealthy_count += 1
    except Exception as e:
        logger.error(f"Neo4j health check error: {e}")
        services["neo4j"] = ServiceStatus(
            name="Neo4j",
            status="unhealthy",
            latency_ms=None,
            message=f"Health check failed: {str(e)}"
        )
        unhealthy_count += 1
    
    # Check Mistral API health
    try:
        mistral_service = get_mistral_service()
        mistral_start = datetime.utcnow()
        mistral_healthy, mistral_error = await mistral_service.health_check()
        mistral_latency = (datetime.utcnow() - mistral_start).total_seconds() * 1000
        
        services["mistral"] = ServiceStatus(
            name="Mistral API",
            status="healthy" if mistral_healthy else "unhealthy",
            latency_ms=round(mistral_latency, 2) if mistral_healthy else None,
            message="Connected" if mistral_healthy else (mistral_error or "Connection failed")
        )
        if not mistral_healthy:
            unhealthy_count += 1
    except Exception as e:
        logger.error(f"Mistral health check error: {e}")
        services["mistral"] = ServiceStatus(
            name="Mistral API",
            status="unhealthy",
            latency_ms=None,
            message=f"Health check failed: {str(e)}"
        )
        unhealthy_count += 1
    
    # Check MCP servers health
    try:
        settings = get_settings()
        
        if not settings.mcp_enabled:
            # MCP is disabled - show informative status
            services["mcp"] = ServiceStatus(
                name="MCP Servers",
                status="unknown",
                latency_ms=None,
                message="MCP disabled (set MCP_ENABLED=true to enable)"
            )
        else:
            mcp_health_service = get_mcp_health_service()
            mcp_summary = await mcp_health_service.check_all_servers()
            
            # Add individual MCP server statuses
            for server_name, server_status in mcp_summary.servers.items():
                service_key = f"mcp_{server_name}"
                services[service_key] = ServiceStatus(
                    name=f"MCP {server_name.title()}",
                    status=server_status.status,
                    latency_ms=server_status.latency_ms,
                    message=server_status.error_message or "Connected"
                )
                if server_status.status != "healthy":
                    unhealthy_count += 1
            
            # Add aggregated MCP status
            services["mcp"] = ServiceStatus(
                name="MCP Servers",
                status=mcp_summary.overall_status,
                latency_ms=None,
                message=f"{mcp_summary.healthy_count}/{mcp_summary.total_servers} servers healthy"
            )
            
            # Attempt reconnection for unhealthy servers
            if mcp_summary.unhealthy_count > 0:
                logger.warning(
                    f"MCP health check: {mcp_summary.unhealthy_count} unhealthy servers, "
                    "attempting reconnection"
                )
                await mcp_health_service.reconnect_unhealthy_servers()
            
    except Exception as e:
        logger.error(f"MCP health check error: {e}")
        services["mcp"] = ServiceStatus(
            name="MCP Servers",
            status="unhealthy",
            latency_ms=None,
            message=f"Health check failed: {str(e)}"
        )
        unhealthy_count += 1
    
    # Determine overall status
    total_services = len(services)
    if unhealthy_count == 0:
        overall_status = "healthy"
    elif unhealthy_count < total_services:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    logger.info(
        f"Health check complete: {overall_status} "
        f"({total_services - unhealthy_count}/{total_services} services healthy)"
    )
    
    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        services=services,
    )


@router.get(
    "/health/mcp",
    summary="MCP servers health check",
    description="Returns detailed health status of all MCP servers.",
)
async def mcp_health_check() -> dict:
    """Check MCP servers health status.
    
    Performs health checks on all registered MCP servers and returns
    detailed status information including latency and error messages.
    
    Returns:
        dict: Detailed health status of all MCP servers.
    """
    try:
        mcp_health_service = get_mcp_health_service()
        return await mcp_health_service.get_health_for_api()
    except Exception as e:
        logger.error(f"MCP health check endpoint error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_servers": 0,
            "healthy_count": 0,
            "unhealthy_count": 0,
            "servers": {},
        }
