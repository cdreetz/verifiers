import asyncio
import logging
from typing import Dict, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool

from .base import BaseTransport


class SSETransport(BaseTransport):
    """Transport using Server-Sent Events for HTTP-based communication.

    This transport connects to an MCP server via HTTP/SSE, which is
    useful for remote servers or servers running in sandboxes with
    exposed ports.
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout

    async def _create_connection(self):
        """Create SSE connection to MCP server."""
        try:
            async with sse_client(
                url=self.url,
                headers=self.headers,
                timeout=self.timeout,
                sse_read_timeout=self.sse_read_timeout,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    await session.initialize()

                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        self.tools[tool.name] = tool

                    self.logger.info(
                        f"Connected to MCP server '{self.name}' at {self.url} "
                        f"with {len(self.tools)} tools"
                    )
                    self._ready.set()

                    # Keep connection alive
                    while True:
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server '{self.name}': {e}")
            self._error = e
            self._ready.set()
        finally:
            self.session = None
            self.tools = {}
