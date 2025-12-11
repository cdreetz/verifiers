import asyncio
import logging
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

from .base import BaseTransport


class StdioTransport(BaseTransport):
    """Transport using stdio for local process communication.

    This transport starts an MCP server as a child process and
    communicates via stdin/stdout.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)
        self.command = command
        self.args = args or []
        self.env = env

    async def _create_connection(self):
        """Create stdio connection to MCP server."""
        try:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    await session.initialize()

                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        self.tools[tool.name] = tool

                    self._ready.set()

                    # Keep connection alive
                    while True:
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._error = e
            self._ready.set()
        finally:
            self.session = None
            self.tools = {}
