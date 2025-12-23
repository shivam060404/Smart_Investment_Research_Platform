import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.mcp.base import HealthStatus
from app.mcp.registry import get_mcp_registry, MCPRegistry


logger = logging.getLogger(__name__)


class MCPServerHealthStatus(BaseModel):
    """Health status for a single MCP server."""

    name: str
    status: str  # "healthy", "unhealthy", "unknown"
    latency_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    reconnect_attempts: int = 0


class MCPHealthSummary(BaseModel):
    """Aggregated health summary for all MCP servers."""

    overall_status: str  # "healthy", "degraded", "unhealthy"
    total_servers: int
    healthy_count: int
    unhealthy_count: int
    servers: dict[str, MCPServerHealthStatus] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class MCPHealthService:
    """Service for monitoring MCP server health.

    Provides health check aggregation and automatic reconnection
    for unhealthy MCP servers.
    """

    def __init__(self, registry: Optional[MCPRegistry] = None):
        """Initialize the MCP health service.

        Args:
            registry: Optional MCP registry instance.
        """
        self._registry = registry

    @property
    def registry(self) -> MCPRegistry:
        """Get the MCP registry."""
        if self._registry is None:
            self._registry = get_mcp_registry()
        return self._registry

    async def check_server_health(self, server_name: str) -> MCPServerHealthStatus:
        """Check health of a specific MCP server.

        Args:
            server_name: Name of the server to check.

        Returns:
            Health status for the server.
        """
        try:
            # Check if server is registered locally
            server = self.registry.get_server(server_name)
            if server:
                health = await server.health_check()
                return MCPServerHealthStatus(
                    name=server_name,
                    status="healthy" if health.healthy else "unhealthy",
                    latency_ms=health.latency_ms,
                    last_check=health.last_check,
                    error_message=health.error_message,
                )

            # Check client-connected servers
            client = self.registry.client
            if server_name in client.servers:
                health = await client.health_check(server_name)
                connection = client.servers.get(server_name)
                return MCPServerHealthStatus(
                    name=server_name,
                    status="healthy" if health.healthy else "unhealthy",
                    latency_ms=health.latency_ms,
                    last_check=health.last_check,
                    error_message=health.error_message,
                    reconnect_attempts=connection.reconnect_attempts if connection else 0,
                )

            return MCPServerHealthStatus(
                name=server_name,
                status="unknown",
                error_message="Server not found",
            )

        except Exception as e:
            logger.error(f"Error checking health for MCP server '{server_name}': {e}")
            return MCPServerHealthStatus(
                name=server_name,
                status="unhealthy",
                error_message=str(e),
            )

    async def check_all_servers(self) -> MCPHealthSummary:
        """Check health of all registered MCP servers.

        Returns:
            Aggregated health summary.
        """
        servers: dict[str, MCPServerHealthStatus] = {}
        healthy_count = 0
        unhealthy_count = 0

        # Get all server names from registry and client
        server_names = set(self.registry.list_servers())
        server_names.update(self.registry.client.servers.keys())

        for server_name in server_names:
            status = await self.check_server_health(server_name)
            servers[server_name] = status

            if status.status == "healthy":
                healthy_count += 1
            else:
                unhealthy_count += 1

        # Determine overall status
        total_servers = len(servers)
        if total_servers == 0:
            overall_status = "unknown"
        elif unhealthy_count == 0:
            overall_status = "healthy"
        elif healthy_count == 0:
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        return MCPHealthSummary(
            overall_status=overall_status,
            total_servers=total_servers,
            healthy_count=healthy_count,
            unhealthy_count=unhealthy_count,
            servers=servers,
        )

    async def reconnect_unhealthy_servers(self) -> dict[str, bool]:
        """Attempt to reconnect all unhealthy servers.

        Returns:
            Dictionary mapping server names to reconnection success.
        """
        results = {}

        # Check and reconnect client-connected servers
        client_results = await self.registry.client.check_and_reconnect_unhealthy()
        results.update(client_results)

        # Log results with appropriate severity
        for server_name, success in results.items():
            if success:
                logger.info(f"Successfully reconnected MCP server '{server_name}'")
            else:
                logger.warning(
                    f"Failed to reconnect MCP server '{server_name}'. "
                    "Service may be degraded."
                )

        return results

    async def log_health_warnings(self) -> None:
        """Log warnings for any unhealthy MCP servers.

        This method checks all servers and logs appropriate warnings
        for any that are unhealthy or experiencing issues.
        """
        summary = await self.check_all_servers()

        if summary.overall_status == "unhealthy":
            logger.error(
                f"MCP infrastructure is unhealthy: "
                f"{summary.unhealthy_count}/{summary.total_servers} servers down"
            )
        elif summary.overall_status == "degraded":
            logger.warning(
                f"MCP infrastructure is degraded: "
                f"{summary.unhealthy_count}/{summary.total_servers} servers unhealthy"
            )

        # Log individual server issues
        for server_name, status in summary.servers.items():
            if status.status == "unhealthy":
                logger.warning(
                    f"MCP server '{server_name}' is unhealthy: {status.error_message}"
                )
                if status.reconnect_attempts > 0:
                    logger.warning(
                        f"MCP server '{server_name}' has failed "
                        f"{status.reconnect_attempts} reconnection attempts"
                    )
            elif status.latency_ms and status.latency_ms > 5000:
                # Warn about high latency (> 5 seconds)
                logger.warning(
                    f"MCP server '{server_name}' has high latency: "
                    f"{status.latency_ms:.0f}ms"
                )

        return results

    async def get_health_for_api(self) -> dict[str, Any]:
        """Get health status formatted for API response.

        Returns:
            Dictionary suitable for API response.
        """
        summary = await self.check_all_servers()

        return {
            "status": summary.overall_status,
            "total_servers": summary.total_servers,
            "healthy_count": summary.healthy_count,
            "unhealthy_count": summary.unhealthy_count,
            "servers": {
                name: {
                    "status": status.status,
                    "latency_ms": status.latency_ms,
                    "error_message": status.error_message,
                }
                for name, status in summary.servers.items()
            },
        }


# Global health service instance
_health_service: Optional[MCPHealthService] = None


def get_mcp_health_service() -> MCPHealthService:
    """Get the global MCP health service instance.

    Returns:
        Global MCPHealthService instance.
    """
    global _health_service
    if _health_service is None:
        _health_service = MCPHealthService()
    return _health_service
