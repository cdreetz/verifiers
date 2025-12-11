from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TransportType(Enum):
    """Transport type for MCP server connections."""

    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Attributes:
        name: Unique identifier for the server.
        transport: Transport type to use (stdio or sse).
        command: Command to run for stdio transport (e.g., "npx", "uvx").
        args: Arguments to pass to the command.
        env: Environment variables to set.
        url: URL for SSE transport connections.
        description: Human-readable description of the server.
    """

    name: str
    transport: TransportType = TransportType.STDIO
    # For stdio transport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # For SSE transport
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    # General
    description: str = ""


@dataclass
class SandboxMCPServerConfig:
    """Configuration for running an MCP server inside a sandbox.

    Attributes:
        name: Unique identifier for the server.
        setup_commands: Shell commands to run during sandbox setup (e.g., pip install).
        start_command: Command to start the MCP server.
        port: Port the MCP server listens on inside the sandbox.
        env: Environment variables to pass to the server.
        description: Human-readable description.
    """

    name: str
    setup_commands: List[str] = field(default_factory=list)
    start_command: str = ""
    port: int = 8000
    env: Optional[Dict[str, str]] = None
    description: str = ""
