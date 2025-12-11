import asyncio
import atexit
import threading
from typing import Callable, Dict, List, Optional, Union

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.types import Message

from .models import MCPServerConfig, TransportType
from .mcp_tool_wrapper import MCPToolWrapper
from .transports.base import BaseTransport
from .transports.stdio import StdioTransport
from .transports.sse import SSETransport


class MCPEnv(ToolEnv):
    """Environment for MCP-based tools using the official MCP SDK.

    This environment supports connecting to MCP servers via different
    transports (stdio, SSE) and exposes their tools for multi-turn
    conversations.
    """

    def __init__(
        self,
        mcp_servers: Optional[List[Union[MCPServerConfig, dict]]] = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs,
    ):
        """Initialize the MCP environment.

        Args:
            mcp_servers: List of MCP server configurations.
            max_turns: Maximum number of conversation turns.
            error_formatter: Function to format tool errors.
            **kwargs: Additional arguments passed to ToolEnv.
        """
        self.mcp_servers: List[MCPServerConfig] = []
        if mcp_servers:
            for server in mcp_servers:
                if isinstance(server, dict):
                    # Handle transport as string
                    if "transport" in server and isinstance(server["transport"], str):
                        server = dict(server)
                        server["transport"] = TransportType(server["transport"])
                    self.mcp_servers.append(MCPServerConfig(**server))
                else:
                    self.mcp_servers.append(server)

        self.transports: Dict[str, BaseTransport] = {}
        self.mcp_tools: Dict[str, MCPToolWrapper] = {}

        self.error_formatter = error_formatter
        self._setup_complete = False
        self._init_kwargs = kwargs
        self._max_turns = max_turns

        super().__init__(
            tools=[], max_turns=max_turns, error_formatter=error_formatter, **kwargs
        )

        # Start a persistent background event loop and connect synchronously
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._run_loop, args=(self._bg_loop,), daemon=True
        )
        self._bg_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_servers(), self._bg_loop)
        fut.result()
        self._setup_complete = True

        # Cleanup on exit
        atexit.register(
            lambda: (
                asyncio.run_coroutine_threadsafe(self.cleanup(), self._bg_loop).result(
                    timeout=5
                ),
                self._shutdown_loop(),
            )
        )

    def _run_loop(self, loop: asyncio.AbstractEventLoop):
        """Run the background event loop."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _create_transport(self, config: MCPServerConfig) -> BaseTransport:
        """Create the appropriate transport for a server config."""
        if config.transport == TransportType.STDIO:
            if not config.command:
                raise ValueError(
                    f"Stdio transport requires 'command' for server '{config.name}'"
                )
            return StdioTransport(
                name=config.name,
                command=config.command,
                args=config.args,
                env=config.env,
                logger=self.logger,
            )
        elif config.transport == TransportType.SSE:
            if not config.url:
                raise ValueError(
                    f"SSE transport requires 'url' for server '{config.name}'"
                )
            return SSETransport(
                name=config.name,
                url=config.url,
                headers=config.headers,
                logger=self.logger,
            )
        else:
            raise ValueError(f"Unsupported transport type: {config.transport}")

    async def _connect_servers(self):
        """Connect to all configured MCP servers."""
        wrapper_tools = []

        for server_config in self.mcp_servers:
            transport = self._create_transport(server_config)
            tools = await transport.connect()

            self.transports[server_config.name] = transport

            for tool in tools.values():
                wrapper = MCPToolWrapper(server_config.name, tool, transport)
                wrapper_tools.append(wrapper)
                self.mcp_tools[wrapper.__name__] = wrapper
                self.logger.info(
                    f"Registered MCP tool: {wrapper.__name__} from server '{server_config.name}'"
                )

        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        """Call an MCP tool."""
        if tool_name in self.tool_map:
            tool_wrapper = self.tool_map[tool_name]
            try:
                result = await tool_wrapper(**tool_args)
                return {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call_id,
                }
            except Exception as e:
                return {
                    "role": "tool",
                    "content": self.error_formatter(e),
                    "tool_call_id": tool_call_id,
                }
        else:
            return {
                "role": "tool",
                "content": f"Error: Tool '{tool_name}' not found",
                "tool_call_id": tool_call_id,
            }

    async def cleanup(self):
        """Disconnect from all MCP servers."""
        for transport in self.transports.values():
            try:
                await transport.disconnect()
            except Exception as e:
                self.logger.warning(f"Error disconnecting transport: {e}")

        self.transports.clear()
        self.mcp_tools.clear()

    def _shutdown_loop(self):
        """Shutdown the background event loop."""
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)
