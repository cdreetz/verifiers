import asyncio
import logging
from typing import Dict, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool

from .base import BaseTransport


class SandboxSSETransport(BaseTransport):
    """Transport for connecting to MCP servers running inside sandboxes.

    This transport is specifically designed for MCP servers running in
    remote sandboxes with exposed ports. It includes retry logic to
    handle DNS propagation delays after port exposure.
    """

    def __init__(
        self,
        name: str,
        url: str,
        sandbox_id: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        connection_retries: int = 10,
        retry_delay: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize sandbox SSE transport.

        Args:
            name: Unique identifier for the transport.
            url: The exposed URL for the sandbox MCP server.
            sandbox_id: ID of the sandbox running the MCP server.
            headers: Optional HTTP headers for the SSE connection.
            timeout: HTTP request timeout in seconds.
            sse_read_timeout: SSE read timeout in seconds.
            connection_retries: Number of retries for initial connection.
            retry_delay: Delay between retries in seconds.
            logger: Optional logger instance.
        """
        super().__init__(name, logger)
        self.url = url
        self.sandbox_id = sandbox_id
        self.headers = headers
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.connection_retries = connection_retries
        self.retry_delay = retry_delay

    async def _create_connection(self):
        """Create SSE connection to MCP server in sandbox with retries."""
        last_error = None

        for attempt in range(self.connection_retries):
            try:
                self.logger.debug(
                    f"Attempting connection to '{self.name}' at {self.url} "
                    f"(attempt {attempt + 1}/{self.connection_retries})"
                )

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
                            f"Connected to sandbox MCP server '{self.name}' "
                            f"(sandbox: {self.sandbox_id}) with {len(self.tools)} tools"
                        )
                        self._ready.set()

                        # Keep connection alive
                        while True:
                            await asyncio.sleep(1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.connection_retries - 1:
                    self.logger.warning(
                        f"Connection attempt {attempt + 1} failed for '{self.name}': {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(
                        f"All connection attempts failed for '{self.name}': {e}"
                    )

        # If we get here, all retries failed
        self._error = last_error or Exception("Failed to connect after all retries")
        self._ready.set()
        self.session = None
        self.tools = {}
