"""
Sandbox MCP Environment Example

This example demonstrates running MCP servers inside isolated sandboxes.
The sandbox provides a secure, isolated environment for running MCP servers
that can be connected to via SSE after port exposure.

This example uses a simple calculator MCP server that provides basic math tools.
"""

from datasets import Dataset

import verifiers as vf
from verifiers.envs.mcp.models import SandboxMCPServerConfig


# Simple MCP server code that will be written to the sandbox
CALCULATOR_SERVER_CODE = '''
"""Simple Calculator MCP Server"""
import asyncio
import json
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
import uvicorn

# Create server
server = Server("calculator")

@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="subtract",
            description="Subtract second number from first",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="multiply",
            description="Multiply two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="divide",
            description="Divide first number by second",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Dividend"},
                    "b": {"type": "number", "description": "Divisor"},
                },
                "required": ["a", "b"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    a = arguments.get("a", 0)
    b = arguments.get("b", 0)

    if name == "add":
        result = a + b
    elif name == "subtract":
        result = a - b
    elif name == "multiply":
        result = a * b
    elif name == "divide":
        if b == 0:
            return [TextContent(type="text", text="Error: Division by zero")]
        result = a / b
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return [TextContent(type="text", text=str(result))]

# SSE transport setup
sse_transport = SseServerTransport("/messages/")

async def handle_sse(request):
    """Handle SSE connections."""
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )
    return Response()

async def handle_messages(request):
    """Handle message posting."""
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )
    return Response()

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=handle_messages, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def load_environment(
    dataset: Dataset | None = None,
    max_turns: int = 10,
    **kwargs,
) -> vf.Environment:
    """Load a SandboxMCPEnv with a calculator MCP server.

    This creates an environment where:
    1. Each rollout gets a fresh sandbox
    2. The sandbox installs and runs a calculator MCP server
    3. The model can use add, subtract, multiply, divide tools
    4. Answers are evaluated based on correctness

    Args:
        dataset: Dataset with 'question' and 'answer' columns.
        max_turns: Maximum conversation turns.
        **kwargs: Additional arguments for SandboxMCPEnv.

    Returns:
        Configured SandboxMCPEnv instance.
    """
    # Default dataset with math questions
    if dataset is None:
        dataset = Dataset.from_dict({
            "question": [
                "What is 25 + 17?",
                "What is 100 - 37?",
                "What is 12 * 8?",
                "What is 144 / 12?",
                "What is (15 + 5) * 3?",
            ],
            "answer": ["42", "63", "96", "12", "60"],
        })

    # MCP server configuration
    mcp_server = SandboxMCPServerConfig(
        name="calculator",
        setup_commands=[
            # Install dependencies
            "pip install mcp starlette uvicorn",
            # Write the server code
            f"cat > /tmp/calculator_server.py << 'SERVEREOF'\n{CALCULATOR_SERVER_CODE}\nSERVEREOF",
        ],
        start_command="python /tmp/calculator_server.py",
        port=8000,
        description="Calculator MCP server with basic math operations",
    )

    # Create rubric for evaluation
    rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        """Judge whether the model got the correct answer."""
        judge_prompt = f"""
        The correct answer is: {answer}

        Did the model arrive at the correct numerical answer?
        Consider only the final numerical result, not the method.
        Answer 'yes' if correct, 'no' if incorrect.
        """
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    rubric.add_reward_func(judge_reward, weight=1.0)

    # System prompt
    system_prompt = """You are a helpful assistant with access to calculator tools.
Use the available tools to solve math problems step by step.
Always use the tools for calculations rather than doing mental math.
After getting the result, state the final answer clearly."""

    # Create the environment
    env = vf.SandboxMCPEnv(
        mcp_server=mcp_server,
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        max_turns=max_turns,
        # Sandbox configuration
        sandbox_name="calculator-mcp",
        docker_image="python:3.11-slim",
        cpu_cores=1,
        memory_gb=2,
        disk_size_gb=5,
        timeout_minutes=30,
        # Connection configuration
        dns_wait_seconds=10.0,
        connection_retries=15,
        connection_retry_delay=2.0,
        **kwargs,
    )

    return env


# Alternative example: File operations MCP server
FILE_SERVER_CODE = '''
"""File Operations MCP Server"""
import os
import asyncio
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
import uvicorn

server = Server("file-ops")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="write_file",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="read_file",
            description="Read content from a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="list_directory",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "write_file":
            path = arguments["path"]
            content = arguments["content"]
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return [TextContent(type="text", text=f"Written {len(content)} bytes to {path}")]
        elif name == "read_file":
            path = arguments["path"]
            with open(path, "r") as f:
                content = f.read()
            return [TextContent(type="text", text=content)]
        elif name == "list_directory":
            path = arguments["path"]
            files = os.listdir(path)
            return [TextContent(type="text", text="\\n".join(files))]
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

sse_transport = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )
    return Response()

async def handle_messages(request):
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )
    return Response()

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=handle_messages, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def load_file_ops_environment(
    dataset: Dataset | None = None,
    max_turns: int = 10,
    **kwargs,
) -> vf.Environment:
    """Load a SandboxMCPEnv with file operations MCP server.

    This provides tools for reading, writing, and listing files
    within the sandbox environment.
    """
    if dataset is None:
        dataset = Dataset.from_dict({
            "question": [
                "Create a file called /tmp/hello.txt with the content 'Hello, World!' and then read it back.",
                "List all files in /tmp directory",
            ],
            "answer": ["Hello, World!", "contains files"],
        })

    mcp_server = SandboxMCPServerConfig(
        name="file-ops",
        setup_commands=[
            "pip install mcp starlette uvicorn",
            f"cat > /tmp/file_server.py << 'SERVEREOF'\n{FILE_SERVER_CODE}\nSERVEREOF",
        ],
        start_command="python /tmp/file_server.py",
        port=8000,
        description="File operations MCP server",
    )

    rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    rubric.add_reward_func(judge_reward, weight=1.0)

    system_prompt = """You are a helpful assistant with file operation tools.
Use the tools to read, write, and manage files as requested.
Always report the results of your operations."""

    return vf.SandboxMCPEnv(
        mcp_server=mcp_server,
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        max_turns=max_turns,
        sandbox_name="file-ops-mcp",
        docker_image="python:3.11-slim",
        **kwargs,
    )
