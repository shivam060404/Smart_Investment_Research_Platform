from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# JSON-RPC Protocol Models
# =============================================================================


class JSONRPCError(BaseModel):
    """JSON-RPC error object."""

    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC request object."""

    jsonrpc: str = "2.0"
    id: str | int = Field(default_factory=lambda: str(uuid4()))
    method: str
    params: Optional[dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC response object."""

    jsonrpc: str = "2.0"
    id: str | int
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


# =============================================================================
# MCP Tool and Resource Definitions
# =============================================================================


class ToolDefinition(BaseModel):
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ResourceDefinition(BaseModel):
    """Definition of an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


class ContentType(str, Enum):
    """Content block types."""

    TEXT = "text"
    JSON = "json"
    IMAGE = "image"


class ContentBlock(BaseModel):
    """Content block in tool results."""

    type: ContentType = ContentType.JSON
    data: Any


class ToolResult(BaseModel):
    """Result from an MCP tool invocation."""

    content: list[ContentBlock] = Field(default_factory=list)
    is_error: bool = False


class HealthStatus(BaseModel):
    """Health status of an MCP server."""

    healthy: bool
    server_name: str
    latency_ms: Optional[float] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


# =============================================================================
# Base MCP Server
# =============================================================================


class BaseMCPServer(ABC):
    """Abstract base class for MCP servers.

    All MCP servers should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize the MCP server.

        Args:
            name: Server name identifier.
            version: Server version string.
        """
        self.name = name
        self.version = version
        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {}
        self._resources: dict[str, ResourceDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the server.

        Args:
            name: Tool name.
            description: Tool description.
            handler: Async function to handle tool calls.
            input_schema: JSON Schema for tool input parameters.
        """
        tool_def = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema or {},
        )
        self._tools[name] = (tool_def, handler)

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "application/json",
    ) -> None:
        """Register a resource with the server.

        Args:
            uri: Resource URI pattern.
            name: Resource name.
            description: Resource description.
            mime_type: Resource MIME type.
        """
        resource_def = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
        )
        self._resources[uri] = resource_def

    async def list_tools(self) -> list[ToolDefinition]:
        """List all available tools.

        Returns:
            List of tool definitions.
        """
        return [tool_def for tool_def, _ in self._tools.values()]

    async def list_resources(self) -> list[ResourceDefinition]:
        """List all available resources.

        Returns:
            List of resource definitions.
        """
        return list(self._resources.values())

    async def call_tool(self, name: str, params: Optional[dict[str, Any]] = None) -> ToolResult:
        """Call a registered tool.

        Args:
            name: Tool name to call.
            params: Tool parameters.

        Returns:
            Tool result with content blocks.

        Raises:
            KeyError: If tool is not found.
        """
        if name not in self._tools:
            return ToolResult(
                content=[
                    ContentBlock(
                        type=ContentType.TEXT,
                        data=f"Tool '{name}' not found",
                    )
                ],
                is_error=True,
            )

        _, handler = self._tools[name]
        try:
            result = await handler(**(params or {}))
            return ToolResult(
                content=[ContentBlock(type=ContentType.JSON, data=result)],
                is_error=False,
            )
        except Exception as e:
            return ToolResult(
                content=[ContentBlock(type=ContentType.TEXT, data=str(e))],
                is_error=True,
            )

    @abstractmethod
    async def read_resource(self, uri: str) -> ContentBlock:
        """Read a resource by URI.

        Args:
            uri: Resource URI to read.

        Returns:
            Content block with resource data.
        """
        pass

    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle an incoming JSON-RPC request.

        Args:
            request: The JSON-RPC request to handle.

        Returns:
            JSON-RPC response with result or error.
        """
        from app.mcp.errors import MCPErrorCode

        try:
            method = request.method
            params = request.params or {}

            if method == "tools/list":
                tools = await self.list_tools()
                return JSONRPCResponse(
                    id=request.id,
                    result={"tools": [t.model_dump() for t in tools]},
                )

            elif method == "resources/list":
                resources = await self.list_resources()
                return JSONRPCResponse(
                    id=request.id,
                    result={"resources": [r.model_dump() for r in resources]},
                )

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})
                if not tool_name:
                    return JSONRPCResponse(
                        id=request.id,
                        error=JSONRPCError(
                            code=MCPErrorCode.INVALID_PARAMS,
                            message="Missing tool name",
                        ),
                    )
                result = await self.call_tool(tool_name, tool_params)
                return JSONRPCResponse(
                    id=request.id,
                    result=result.model_dump(),
                )

            elif method == "resources/read":
                uri = params.get("uri")
                if not uri:
                    return JSONRPCResponse(
                        id=request.id,
                        error=JSONRPCError(
                            code=MCPErrorCode.INVALID_PARAMS,
                            message="Missing resource URI",
                        ),
                    )
                content = await self.read_resource(uri)
                return JSONRPCResponse(
                    id=request.id,
                    result=content.model_dump(),
                )

            else:
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=MCPErrorCode.METHOD_NOT_FOUND,
                        message=f"Method '{method}' not found",
                    ),
                )

        except Exception as e:
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message=str(e),
                ),
            )

    async def health_check(self) -> HealthStatus:
        """Check server health.

        Returns:
            Health status of the server.
        """
        return HealthStatus(
            healthy=True,
            server_name=self.name,
        )
