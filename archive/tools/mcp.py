"""
Model Context Protocol (MCP) Support for Company AGI.

Implements MCP client to connect to external tool servers:
- HTTP/SSE remote servers
- Stdio local servers
- Tool discovery and invocation
- OAuth authentication support

Reference: https://modelcontextprotocol.io/
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    """MCP transport types."""
    HTTP = "http"
    SSE = "sse"  # Server-Sent Events
    STDIO = "stdio"


class MCPCapability(Enum):
    """MCP server capabilities."""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    SAMPLING = "sampling"


@dataclass
class MCPTool:
    """An MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
            "metadata": self.metadata,
        }


@dataclass
class MCPResource:
    """An MCP resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "server_name": self.server_name,
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: MCPTransport
    command: Optional[str] = None  # For stdio
    args: Optional[List[str]] = None
    url: Optional[str] = None  # For HTTP/SSE
    headers: Optional[Dict[str, str]] = None
    env: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None
    timeout: int = 30
    enabled: bool = True
    scope: str = "project"  # local, project, user

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "MCPServerConfig":
        transport = MCPTransport(data.get("transport", "stdio"))
        return cls(
            name=name,
            transport=transport,
            command=data.get("command"),
            args=data.get("args", []),
            url=data.get("url"),
            headers=data.get("headers", {}),
            env=data.get("env", {}),
            auth_token=data.get("auth_token"),
            timeout=data.get("timeout", 30),
            enabled=data.get("enabled", True),
            scope=data.get("scope", "project"),
        )


@dataclass
class MCPResult:
    """Result from an MCP operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPConnection(ABC):
    """Abstract base class for MCP connections."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.connected = False
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self.capabilities: List[MCPCapability] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool on the server."""
        pass

    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        pass

    @abstractmethod
    async def list_resources(self) -> List[MCPResource]:
        """List available resources."""
        pass

    @abstractmethod
    async def read_resource(self, uri: str) -> MCPResult:
        """Read a resource."""
        pass


class StdioMCPConnection(MCPConnection):
    """MCP connection over stdio (local process)."""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Start the MCP server process."""
        if not self.config.command:
            return False

        try:
            # Prepare environment
            env = os.environ.copy()
            if self.config.env:
                env.update(self.config.env)

            # Start process
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *(self.config.args or []),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Start reader task
            self._reader_task = asyncio.create_task(self._read_responses())

            # Initialize connection
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                },
                "clientInfo": {
                    "name": "multi-agent-llm-company-system",
                    "version": "1.0.0",
                }
            })

            if init_result.success:
                self.connected = True
                # Discover tools and resources
                self.tools = await self.list_tools()
                self.resources = await self.list_resources()

            return self.connected

        except Exception as e:
            logger.warning("Failed to connect to MCP server %s: %s", self.config.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()

        self.connected = False

    async def _send_request(self, method: str, params: Dict[str, Any]) -> MCPResult:
        """Send a JSON-RPC request."""
        if not self.process or not self.process.stdin:
            return MCPResult(success=False, error="Not connected")

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line.encode())
            await self.process.stdin.drain()

            # Wait for response
            response = await asyncio.wait_for(future, timeout=self.config.timeout)
            return response

        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            return MCPResult(success=False, error="Request timeout")
        except Exception as e:
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
            return MCPResult(success=False, error=str(e))

    async def _read_responses(self) -> None:
        """Read responses from the process."""
        if not self.process or not self.process.stdout:
            return

        while True:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break

                response = json.loads(line.decode())
                request_id = response.get("id")

                if request_id in self._pending_requests:
                    future = self._pending_requests.pop(request_id)

                    if "error" in response:
                        error = response["error"]
                        future.set_result(MCPResult(
                            success=False,
                            error=f"{error.get('code')}: {error.get('message')}"
                        ))
                    else:
                        future.set_result(MCPResult(
                            success=True,
                            data=response.get("result")
                        ))

            except json.JSONDecodeError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool."""
        return await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

    async def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        result = await self._send_request("tools/list", {})
        if not result.success:
            return []

        tools = []
        for tool_data in result.data.get("tools", []):
            tools.append(MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            ))

        return tools

    async def list_resources(self) -> List[MCPResource]:
        """List available resources."""
        result = await self._send_request("resources/list", {})
        if not result.success:
            return []

        resources = []
        for res_data in result.data.get("resources", []):
            resources.append(MCPResource(
                uri=res_data["uri"],
                name=res_data["name"],
                description=res_data.get("description"),
                mime_type=res_data.get("mimeType"),
                server_name=self.config.name,
            ))

        return resources

    async def read_resource(self, uri: str) -> MCPResult:
        """Read a resource."""
        return await self._send_request("resources/read", {"uri": uri})


class HTTPMCPConnection(MCPConnection):
    """MCP connection over HTTP."""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._session = None

    async def connect(self) -> bool:
        """Connect to HTTP MCP server."""
        try:
            import aiohttp
        except ImportError:
            return False

        if not self.config.url:
            return False

        try:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

            # Test connection
            async with self._session.get(f"{self.config.url}/health") as resp:
                if resp.status == 200:
                    self.connected = True
                    self.tools = await self.list_tools()
                    self.resources = await self.list_resources()
                    return True

        except Exception as e:
            logger.warning("Failed to connect to HTTP MCP server %s: %s", self.config.name, e)

        return False

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = self.config.headers.copy() if self.config.headers else {}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
        self.connected = False

    async def _post(self, endpoint: str, data: Dict[str, Any]) -> MCPResult:
        """Make a POST request."""
        if not self._session or not self.config.url:
            return MCPResult(success=False, error="Not connected")

        try:
            async with self._session.post(
                f"{self.config.url}{endpoint}",
                json=data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return MCPResult(success=True, data=result)
                else:
                    error = await resp.text()
                    return MCPResult(success=False, error=f"HTTP {resp.status}: {error}")

        except Exception as e:
            return MCPResult(success=False, error=str(e))

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool via HTTP."""
        return await self._post("/tools/call", {
            "name": name,
            "arguments": arguments,
        })

    async def list_tools(self) -> List[MCPTool]:
        """List tools via HTTP."""
        result = await self._post("/tools/list", {})
        if not result.success:
            return []

        tools = []
        for tool_data in result.data.get("tools", []):
            tools.append(MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            ))

        return tools

    async def list_resources(self) -> List[MCPResource]:
        """List resources via HTTP."""
        result = await self._post("/resources/list", {})
        if not result.success:
            return []

        resources = []
        for res_data in result.data.get("resources", []):
            resources.append(MCPResource(
                uri=res_data["uri"],
                name=res_data["name"],
                description=res_data.get("description"),
                mime_type=res_data.get("mimeType"),
                server_name=self.config.name,
            ))

        return resources

    async def read_resource(self, uri: str) -> MCPResult:
        """Read a resource via HTTP."""
        return await self._post("/resources/read", {"uri": uri})


class MCPManager:
    """Manages MCP server connections."""

    def __init__(self, config_file: Optional[str] = None):
        self.connections: Dict[str, MCPConnection] = {}
        self.config_file = config_file or ".mcp.json"
        self._tool_cache: Dict[str, MCPTool] = {}

    async def load_config(self) -> None:
        """Load MCP configuration from file."""
        config_paths = [
            Path(self.config_file),  # Project config
            Path.home() / ".claude.json",  # User config
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    servers = config.get("mcpServers", {})
                    for name, server_config in servers.items():
                        await self.add_server(
                            MCPServerConfig.from_dict(name, server_config)
                        )

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to load MCP config from %s: %s", config_path, e)

    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and connect to an MCP server."""
        if not config.enabled:
            return False

        # Create appropriate connection type
        if config.transport == MCPTransport.STDIO:
            connection = StdioMCPConnection(config)
        elif config.transport in (MCPTransport.HTTP, MCPTransport.SSE):
            connection = HTTPMCPConnection(config)
        else:
            return False

        # Connect
        if await connection.connect():
            self.connections[config.name] = connection

            # Cache tools
            for tool in connection.tools:
                self._tool_cache[f"{config.name}:{tool.name}"] = tool

            return True

        return False

    async def remove_server(self, name: str) -> bool:
        """Disconnect and remove an MCP server."""
        if name in self.connections:
            await self.connections[name].disconnect()
            del self.connections[name]

            # Remove from tool cache
            self._tool_cache = {
                k: v for k, v in self._tool_cache.items()
                if not k.startswith(f"{name}:")
            }

            return True
        return False

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for connection in self.connections.values():
            await connection.disconnect()
        self.connections.clear()
        self._tool_cache.clear()

    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools from all servers."""
        return list(self._tool_cache.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name (server:tool_name or just tool_name)."""
        if ":" in name:
            return self._tool_cache.get(name)

        # Search by tool name only
        for key, tool in self._tool_cache.items():
            if tool.name == name:
                return tool

        return None

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return MCPResult(success=False, error=f"Tool not found: {name}")

        connection = self.connections.get(tool.server_name)
        if not connection:
            return MCPResult(success=False, error=f"Server not connected: {tool.server_name}")

        return await connection.call_tool(tool.name, arguments)

    def get_all_resources(self) -> List[MCPResource]:
        """Get all available resources from all servers."""
        resources = []
        for connection in self.connections.values():
            resources.extend(connection.resources)
        return resources

    async def read_resource(self, uri: str) -> MCPResult:
        """Read a resource by URI."""
        # Find the server that owns this resource
        for connection in self.connections.values():
            for resource in connection.resources:
                if resource.uri == uri:
                    return await connection.read_resource(uri)

        return MCPResult(success=False, error=f"Resource not found: {uri}")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all MCP connections."""
        return {
            "servers": {
                name: {
                    "connected": conn.connected,
                    "transport": conn.config.transport.value,
                    "tool_count": len(conn.tools),
                    "resource_count": len(conn.resources),
                }
                for name, conn in self.connections.items()
            },
            "total_tools": len(self._tool_cache),
            "total_resources": sum(len(c.resources) for c in self.connections.values()),
        }


# Convenience function to create an MCP wrapper for UnifiedTools
class MCPToolWrapper:
    """Wrapper that integrates MCP tools with UnifiedTools."""

    def __init__(self, mcp_manager: MCPManager):
        self.mcp = mcp_manager

    async def call(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call an MCP tool."""
        result = await self.mcp.call_tool(tool_name, kwargs)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools."""
        return [tool.to_dict() for tool in self.mcp.get_all_tools()]


# Global MCP manager instance
_global_mcp_manager: Optional[MCPManager] = None


async def get_mcp_manager(config_file: Optional[str] = None) -> MCPManager:
    """Get or create the global MCP manager."""
    global _global_mcp_manager
    if _global_mcp_manager is None:
        _global_mcp_manager = MCPManager(config_file)
        await _global_mcp_manager.load_config()
    return _global_mcp_manager


def reset_mcp_manager() -> None:
    """Reset the global MCP manager."""
    global _global_mcp_manager
    _global_mcp_manager = None
