import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.mcp.base import (
    ContentBlock,
    HealthStatus,
    JSONRPCRequest,
    JSONRPCResponse,
    ToolDefinition,
    ToolResult,
)
from app.mcp.errors import (
    MCPConnectionError,
    MCPErrorHandler,
    MCPServerUnavailableError,
    MCPTimeoutError,
    mcp_error_from_code,
)


logger = logging.getLogger(__name__)

# Health check constants
HEALTH_CHECK_INTERVAL_SECONDS = 60
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SECONDS = 5


@dataclass
class ServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: str = "stdio"  # "stdio" or "http"
    command: Optional[str] = None  # For stdio transport
    url: Optional[str] = None  # For http transport
    timeout_seconds: float = 30.0
    enabled: bool = True


@dataclass
class MCPServerConnection:
    """Represents a connection to an MCP server."""

    config: ServerConfig
    connected: bool = False
    last_health_check: Optional[datetime] = None
    health_status: Optional[HealthStatus] = None
    reconnect_attempts: int = 0
    last_reconnect_attempt: Optional[datetime] = None
    _tools_cache: list[ToolDefinition] = field(default_factory=list)


class MCPClient:
    """Client for connecting to and invoking MCP servers.

    This client manages connections to multiple MCP servers and provides
    methods for tool invocation and resource access with timeout handling.
    """

    def __init__(self, default_timeout: float = 30.0):
        """Initialize the MCP client.

        Args:
            default_timeout: Default timeout for server operations in seconds.
        """
        self.default_timeout = default_timeout
        self._servers: dict[str, MCPServerConnection] = {}
        self._error_handler = MCPErrorHandler(max_retries=1)

    @property
    def servers(self) -> dict[str, MCPServerConnection]:
        """Get all registered server connections."""
        return self._servers

    async def connect(self, config: ServerConfig) -> None:
        """Connect to an MCP server.

        Args:
            config: Server configuration.

        Raises:
            MCPConnectionError: If connection fails.
        """
        if not config.enabled:
            logger.info(f"MCP server '{config.name}' is disabled, skipping connection")
            return

        logger.info(f"Connecting to MCP server '{config.name}'...")

        connection = MCPServerConnection(config=config)

        try:
            # For now, we simulate connection establishment
            # In a full implementation, this would establish stdio/http connection
            connection.connected = True
            connection.last_health_check = datetime.utcnow()

            # Cache available tools
            connection._tools_cache = await self._fetch_tools(connection)

            self._servers[config.name] = connection
            logger.info(
                f"Connected to MCP server '{config.name}' with "
                f"{len(connection._tools_cache)} tools"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{config.name}': {e}")
            raise MCPConnectionError(config.name, str(e))

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            server_name: Name of the server to disconnect.
        """
        if server_name in self._servers:
            connection = self._servers[server_name]
            connection.connected = False
            del self._servers[server_name]
            logger.info(f"Disconnected from MCP server '{server_name}'")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        server_names = list(self._servers.keys())
        for name in server_names:
            await self.disconnect(name)

    def _get_connection(self, server_name: str) -> MCPServerConnection:
        """Get a server connection by name.

        Args:
            server_name: Name of the server.

        Returns:
            Server connection.

        Raises:
            MCPServerUnavailableError: If server is not connected.
        """
        connection = self._servers.get(server_name)
        if not connection or not connection.connected:
            raise MCPServerUnavailableError(server_name)
        return connection

    async def _send_request(
        self,
        connection: MCPServerConnection,
        request: JSONRPCRequest,
    ) -> JSONRPCResponse:
        """Send a JSON-RPC request to a server.

        Args:
            connection: Server connection.
            request: JSON-RPC request.

        Returns:
            JSON-RPC response.

        Raises:
            MCPTimeoutError: If request times out.
        """
        timeout = connection.config.timeout_seconds

        try:
            # This is a placeholder for actual transport implementation
            # In production, this would use stdio or HTTP transport
            async with asyncio.timeout(timeout):
                # Simulate request/response for now
                # Real implementation would send via transport
                return JSONRPCResponse(
                    id=request.id,
                    result={"status": "ok"},
                )
        except asyncio.TimeoutError:
            raise MCPTimeoutError(connection.config.name, timeout)

    async def _fetch_tools(
        self, connection: MCPServerConnection
    ) -> list[ToolDefinition]:
        """Fetch available tools from a server.

        Args:
            connection: Server connection.

        Returns:
            List of tool definitions.
        """
        # Placeholder - in real implementation, this would query the server
        return []

    async def call_tool(
        self,
        server: str,
        tool: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Call a tool on an MCP server.

        Args:
            server: Server name.
            tool: Tool name.
            params: Tool parameters.
            timeout: Optional timeout override.

        Returns:
            Tool result.

        Raises:
            MCPServerUnavailableError: If server is not connected.
            MCPTimeoutError: If request times out.
        """
        connection = self._get_connection(server)

        if timeout:
            original_timeout = connection.config.timeout_seconds
            connection.config.timeout_seconds = timeout

        try:
            request = JSONRPCRequest(
                method="tools/call",
                params={"name": tool, "arguments": params or {}},
            )

            async def _execute():
                response = await self._send_request(connection, request)
                if response.error:
                    raise mcp_error_from_code(
                        response.error.code,
                        response.error.message,
                        response.error.data,
                    )
                return ToolResult(
                    content=[ContentBlock(data=response.result)],
                    is_error=False,
                )

            return await self._error_handler.execute_with_retry(
                _execute,
                f"call_tool({server}/{tool})",
            )

        finally:
            if timeout:
                connection.config.timeout_seconds = original_timeout

    async def get_resource(
        self,
        server: str,
        uri: str,
        timeout: Optional[float] = None,
    ) -> ContentBlock:
        """Get a resource from an MCP server.

        Args:
            server: Server name.
            uri: Resource URI.
            timeout: Optional timeout override.

        Returns:
            Resource content.

        Raises:
            MCPServerUnavailableError: If server is not connected.
            MCPTimeoutError: If request times out.
        """
        connection = self._get_connection(server)

        if timeout:
            original_timeout = connection.config.timeout_seconds
            connection.config.timeout_seconds = timeout

        try:
            request = JSONRPCRequest(
                method="resources/read",
                params={"uri": uri},
            )

            async def _execute():
                response = await self._send_request(connection, request)
                if response.error:
                    raise mcp_error_from_code(
                        response.error.code,
                        response.error.message,
                        response.error.data,
                    )
                return ContentBlock(data=response.result)

            return await self._error_handler.execute_with_retry(
                _execute,
                f"get_resource({server}/{uri})",
            )

        finally:
            if timeout:
                connection.config.timeout_seconds = original_timeout

    async def list_available_tools(self) -> dict[str, list[ToolDefinition]]:
        """List all available tools from all connected servers.

        Returns:
            Dictionary mapping server names to their tool definitions.
        """
        result = {}
        for name, connection in self._servers.items():
            if connection.connected:
                result[name] = connection._tools_cache
        return result

    async def discover_servers(self) -> list[str]:
        """Discover all connected server names.

        Returns:
            List of connected server names.
        """
        return [
            name
            for name, conn in self._servers.items()
            if conn.connected
        ]

    async def health_check(self, server_name: str) -> HealthStatus:
        """Check health of a specific server.

        Args:
            server_name: Name of the server to check.

        Returns:
            Health status of the server.
        """
        connection = self._servers.get(server_name)

        if not connection:
            return HealthStatus(
                healthy=False,
                server_name=server_name,
                error_message="Server not registered",
            )

        if not connection.connected:
            return HealthStatus(
                healthy=False,
                server_name=server_name,
                error_message="Server not connected",
            )

        start_time = time.time()

        try:
            request = JSONRPCRequest(method="tools/list")
            await self._send_request(connection, request)

            latency_ms = (time.time() - start_time) * 1000

            status = HealthStatus(
                healthy=True,
                server_name=server_name,
                latency_ms=latency_ms,
            )
            connection.health_status = status
            connection.last_health_check = datetime.utcnow()

            return status

        except Exception as e:
            return HealthStatus(
                healthy=False,
                server_name=server_name,
                error_message=str(e),
            )

    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Check health of all registered servers.

        Returns:
            Dictionary mapping server names to their health status.
        """
        results = {}
        for name in self._servers:
            results[name] = await self.health_check(name)
        return results

    async def reconnect(self, server_name: str) -> bool:
        """Attempt to reconnect to an unhealthy server.

        Args:
            server_name: Name of the server to reconnect.

        Returns:
            True if reconnection was successful, False otherwise.
        """
        connection = self._servers.get(server_name)
        if not connection:
            logger.warning(f"Cannot reconnect: server '{server_name}' not registered")
            return False

        # Check if we've exceeded max reconnect attempts
        if connection.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.error(
                f"MCP server '{server_name}' exceeded max reconnect attempts "
                f"({MAX_RECONNECT_ATTEMPTS})"
            )
            return False

        # Check if we should wait before reconnecting
        if connection.last_reconnect_attempt:
            time_since_last = (
                datetime.utcnow() - connection.last_reconnect_attempt
            ).total_seconds()
            if time_since_last < RECONNECT_DELAY_SECONDS:
                logger.debug(
                    f"Waiting before reconnect attempt for '{server_name}'"
                )
                return False

        connection.reconnect_attempts += 1
        connection.last_reconnect_attempt = datetime.utcnow()

        logger.warning(
            f"Attempting to reconnect to MCP server '{server_name}' "
            f"(attempt {connection.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})"
        )

        try:
            # Attempt reconnection
            connection.connected = True
            connection._tools_cache = await self._fetch_tools(connection)

            # Verify health after reconnection
            health = await self.health_check(server_name)
            if health.healthy:
                connection.reconnect_attempts = 0  # Reset on success
                logger.info(f"Successfully reconnected to MCP server '{server_name}'")
                return True
            else:
                connection.connected = False
                logger.warning(
                    f"Reconnection to '{server_name}' succeeded but health check failed: "
                    f"{health.error_message}"
                )
                return False

        except Exception as e:
            connection.connected = False
            logger.error(f"Failed to reconnect to MCP server '{server_name}': {e}")
            return False

    async def check_and_reconnect_unhealthy(self) -> dict[str, bool]:
        """Check all servers and attempt to reconnect unhealthy ones.

        Returns:
            Dictionary mapping server names to reconnection success status.
        """
        results = {}
        health_statuses = await self.health_check_all()

        for server_name, health in health_statuses.items():
            if not health.healthy:
                logger.warning(
                    f"MCP server '{server_name}' is unhealthy: {health.error_message}. "
                    "Attempting automatic reconnection."
                )
                success = await self.reconnect(server_name)
                results[server_name] = success
                
                if not success:
                    connection = self._servers.get(server_name)
                    if connection and connection.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                        logger.error(
                            f"MCP server '{server_name}' has exceeded maximum reconnection "
                            f"attempts ({MAX_RECONNECT_ATTEMPTS}). Manual intervention may be required."
                        )
            else:
                # Reset reconnect attempts for healthy servers
                connection = self._servers.get(server_name)
                if connection:
                    if connection.reconnect_attempts > 0:
                        logger.info(
                            f"MCP server '{server_name}' recovered after "
                            f"{connection.reconnect_attempts} reconnection attempts"
                        )
                    connection.reconnect_attempts = 0

        return results
