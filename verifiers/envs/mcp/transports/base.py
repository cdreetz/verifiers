import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mcp import ClientSession
from mcp.types import TextContent, Tool


class BaseTransport(ABC):
    """Base class for MCP transport implementations.

    A transport handles the connection lifecycle and communication
    with an MCP server. Different transports support different
    connection methods (stdio, SSE, etc.).
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}
        self._connection_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._error: Optional[Exception] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    @abstractmethod
    async def _create_connection(self):
        """Create the underlying connection. Must be implemented by subclasses.

        This method should:
        1. Establish the connection to the MCP server
        2. Create a ClientSession
        3. Call session.initialize()
        4. List tools and populate self.tools
        5. Set self._ready when ready
        6. Keep the connection alive until cancelled
        """
        pass

    async def connect(self) -> Dict[str, Tool]:
        """Connect to the MCP server and return available tools."""
        self.loop = asyncio.get_running_loop()
        self._connection_task = asyncio.create_task(self._create_connection())

        await self._ready.wait()

        if self._error:
            raise self._error

        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server."""
        assert self.session is not None, f"Server '{self.name}' not connected"
        assert self.loop is not None, "Connection loop not initialized"

        fut = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(tool_name, arguments=arguments), self.loop
        )
        result = await asyncio.wrap_future(fut)

        if result.content:
            text_parts = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    assert isinstance(content_item, TextContent)
                    text_parts.append(content_item.text)
                elif hasattr(content_item, "type") and content_item.type == "text":
                    text_parts.append(getattr(content_item, "text", str(content_item)))
                else:
                    text_parts.append(str(content_item))
            return "\n".join(text_parts)

        return "No result returned from tool"

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self._connection_task is not None:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        self.logger.info(f"MCP server '{self.name}' disconnected")
