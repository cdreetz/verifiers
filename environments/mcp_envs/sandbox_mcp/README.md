# Sandbox MCP Environment

This example demonstrates running MCP (Model Context Protocol) servers inside isolated sandboxes using the `SandboxMCPEnv` class.

## Overview

The Sandbox MCP Environment combines:
- **Sandbox isolation**: Each rollout gets a fresh, isolated sandbox container
- **MCP connectivity**: Connect to MCP servers via SSE (Server-Sent Events)
- **Port exposure**: Expose sandbox ports for external connectivity

## How It Works

1. **Sandbox Creation**: A new sandbox is created for each rollout
2. **Server Setup**: Install dependencies and deploy the MCP server code
3. **Server Start**: Start the MCP server listening on a configured port
4. **Port Exposure**: Expose the port to get an external URL
5. **SSE Connection**: Connect to the MCP server via SSE transport
6. **Tool Discovery**: Discover and register available tools
7. **Execution**: The model can now call MCP tools
8. **Cleanup**: Sandbox is destroyed after the rollout

## Example: Calculator Server

The default example provides a calculator MCP server with basic math operations:

```python
from sandbox_mcp import load_environment

env = load_environment()
```

Available tools:
- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract b from a
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide a by b

## Example: File Operations Server

An alternative example provides file operation tools:

```python
from sandbox_mcp import load_file_ops_environment

env = load_file_ops_environment()
```

Available tools:
- `write_file(path, content)` - Write content to a file
- `read_file(path)` - Read content from a file
- `list_directory(path)` - List files in a directory

## Custom MCP Servers

To use your own MCP server:

```python
from verifiers.envs.mcp import SandboxMCPEnv
from verifiers.envs.mcp.models import SandboxMCPServerConfig

mcp_server = SandboxMCPServerConfig(
    name="my-server",
    setup_commands=[
        "pip install my-mcp-server",
    ],
    start_command="my-mcp-server --port 8000",
    port=8000,
    description="My custom MCP server",
)

env = SandboxMCPEnv(
    mcp_server=mcp_server,
    dataset=my_dataset,
    # ... other options
)
```

## Configuration Options

### SandboxMCPServerConfig

- `name`: Unique identifier for the server
- `setup_commands`: Shell commands to run during setup (e.g., pip install)
- `start_command`: Command to start the MCP server
- `port`: Port the server listens on (default: 8000)
- `env`: Environment variables to pass to the server
- `description`: Human-readable description

### SandboxMCPEnv

- `mcp_server`: SandboxMCPServerConfig instance
- `docker_image`: Docker image for sandboxes (default: "python:3.11-slim")
- `cpu_cores`, `memory_gb`, `disk_size_gb`: Resource limits
- `timeout_minutes`: Sandbox timeout
- `dns_wait_seconds`: Time to wait for DNS after port exposure
- `connection_retries`: Number of SSE connection attempts
- `max_turns`: Maximum conversation turns

## Writing MCP Servers for Sandboxes

MCP servers for sandbox environments should:

1. Use SSE transport (not stdio) for network connectivity
2. Listen on `0.0.0.0` to accept connections from outside the container
3. Use a configurable port (default: 8000)

Example server structure:

```python
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
import uvicorn

server = Server("my-server")

# Define tools with @server.list_tools() and @server.call_tool()

sse_transport = SseServerTransport("/messages/")

# Setup routes for /sse and /messages/
app = Starlette(routes=[...])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Requirements

- `verifiers>=0.1.4`
- `mcp>=1.14.1`
- `prime-sandboxes`

## Running

```bash
cd environments/mcp_envs/sandbox_mcp
uv run python -c "from sandbox_mcp import load_environment; env = load_environment(); print(env)"
```
