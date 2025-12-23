import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from app.mcp.base import BaseMCPServer, HealthStatus
from app.mcp.client import MCPClient, ServerConfig


logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    enabled: bool = True
    transport: str = "stdio"
    command: Optional[str] = None
    url: Optional[str] = None
    timeout_seconds: float = 30.0


class MCPConfig(BaseModel):
    """Root configuration for all MCP servers."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class MCPRegistry:
    """Registry for MCP servers.

    Manages server registration, discovery, and configuration loading
    from YAML files and environment variables.
    """

    def __init__(self):
        """Initialize the MCP registry."""
        self._servers: dict[str, BaseMCPServer] = {}
        self._client: Optional[MCPClient] = None
        self._config: Optional[MCPConfig] = None

    @property
    def client(self) -> MCPClient:
        """Get or create the MCP client."""
        if self._client is None:
            self._client = MCPClient()
        return self._client

    def register_server(self, server: BaseMCPServer) -> None:
        """Register an MCP server instance.

        Args:
            server: MCP server instance to register.
        """
        self._servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")

    def unregister_server(self, name: str) -> None:
        """Unregister an MCP server.

        Args:
            name: Name of the server to unregister.
        """
        if name in self._servers:
            del self._servers[name]
            logger.info(f"Unregistered MCP server: {name}")

    def get_server(self, name: str) -> Optional[BaseMCPServer]:
        """Get a registered server by name.

        Args:
            name: Server name.

        Returns:
            Server instance or None if not found.
        """
        return self._servers.get(name)

    def list_servers(self) -> list[str]:
        """List all registered server names.

        Returns:
            List of server names.
        """
        return list(self._servers.keys())

    def load_config_from_yaml(self, config_path: str | Path) -> MCPConfig:
        """Load MCP configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Loaded MCP configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file is invalid.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"MCP config file not found: {config_path}")
            return MCPConfig()

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            self._config = MCPConfig(**data)
            logger.info(
                f"Loaded MCP config from {config_path} with "
                f"{len(self._config.servers)} servers"
            )
            return self._config

        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
            raise ValueError(f"Invalid MCP config: {e}")

    def load_config_from_env(self) -> MCPConfig:
        """Load MCP configuration from environment variables.

        Environment variables follow the pattern:
        - MCP_{SERVER}_ENABLED: Enable/disable server
        - MCP_{SERVER}_TRANSPORT: Transport type (stdio/http)
        - MCP_{SERVER}_COMMAND: Command for stdio transport
        - MCP_{SERVER}_URL: URL for http transport
        - MCP_{SERVER}_TIMEOUT: Timeout in seconds

        Returns:
            Loaded MCP configuration.
        """
        servers: dict[str, MCPServerConfig] = {}

        # Known server names to check
        known_servers = ["YFINANCE", "NEWS", "TECHNICAL", "NEO4J", "BACKTESTING"]

        for server_name in known_servers:
            prefix = f"MCP_{server_name}"

            # Check if server is configured via env
            enabled_env = os.getenv(f"{prefix}_ENABLED")
            if enabled_env is None:
                continue

            config = MCPServerConfig(
                enabled=enabled_env.lower() in ("true", "1", "yes"),
                transport=os.getenv(f"{prefix}_TRANSPORT", "stdio"),
                command=os.getenv(f"{prefix}_COMMAND"),
                url=os.getenv(f"{prefix}_URL"),
                timeout_seconds=float(
                    os.getenv(f"{prefix}_TIMEOUT", "30")
                ),
            )

            servers[server_name.lower()] = config

        self._config = MCPConfig(servers=servers)
        logger.info(
            f"Loaded MCP config from environment with "
            f"{len(servers)} servers"
        )
        return self._config

    def get_config(self) -> MCPConfig:
        """Get the current configuration.

        Returns:
            Current MCP configuration or empty config if not loaded.
        """
        return self._config or MCPConfig()

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured and enabled servers.

        Returns:
            Dictionary mapping server names to connection success status.
        """
        if not self._config:
            logger.warning("No MCP config loaded, skipping connections")
            return {}

        results: dict[str, bool] = {}

        for name, server_config in self._config.servers.items():
            if not server_config.enabled:
                logger.info(f"Skipping disabled MCP server: {name}")
                results[name] = False
                continue

            try:
                config = ServerConfig(
                    name=name,
                    transport=server_config.transport,
                    command=server_config.command,
                    url=server_config.url,
                    timeout_seconds=server_config.timeout_seconds,
                    enabled=server_config.enabled,
                )
                await self.client.connect(config)
                results[name] = True

            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{name}': {e}")
                results[name] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        await self.client.disconnect_all()

    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Check health of all registered servers.

        Returns:
            Dictionary mapping server names to health status.
        """
        results: dict[str, HealthStatus] = {}

        # Check client-connected servers
        client_health = await self.client.health_check_all()
        results.update(client_health)

        # Check locally registered servers
        for name, server in self._servers.items():
            if name not in results:
                try:
                    results[name] = await server.health_check()
                except Exception as e:
                    results[name] = HealthStatus(
                        healthy=False,
                        server_name=name,
                        error_message=str(e),
                    )

        return results


# Global registry instance
_registry: Optional[MCPRegistry] = None


def get_mcp_registry() -> MCPRegistry:
    """Get the global MCP registry instance.

    Returns:
        Global MCPRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry
