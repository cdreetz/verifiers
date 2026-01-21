"""
Tool execution with clean async patterns.

Key design:
- Tools are just async functions
- Schema generation is separate from execution
- State injection via context, not magic args
- Parallel tool execution when possible
"""

from __future__ import annotations

import asyncio
import json
import inspect
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any, get_type_hints
from functools import wraps


@dataclass(frozen=True)
class ToolDef:
    """Definition of a tool."""

    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    parameters: dict[str, Any]  # JSON Schema
    inject_context: bool = False  # Whether to inject context as first arg


@dataclass
class ToolResult:
    """Result of tool execution."""

    tool_call_id: str
    name: str
    output: str
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0


class ToolExecutor:
    """
    Executes tools with proper async handling.

    Features:
    - Parallel execution of independent tools
    - Timeout handling per tool
    - Context injection for stateful tools
    - Clean error isolation
    """

    def __init__(
        self,
        tools: list[ToolDef],
        *,
        default_timeout_ms: float = 30000,
        max_parallel: int = 10,
    ):
        self.tools = {t.name: t for t in tools}
        self.default_timeout_ms = default_timeout_ms
        self.max_parallel = max_parallel
        self._semaphore = asyncio.Semaphore(max_parallel)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Generate OpenAI-compatible tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    async def execute(
        self,
        tool_calls: list[dict[str, Any]],
        context: Any = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls, in parallel where possible.

        Args:
            tool_calls: List of OpenAI-format tool calls
            context: Optional context to inject into tools that need it
        """
        tasks = [
            self._execute_one(tc, context)
            for tc in tool_calls
        ]
        return await asyncio.gather(*tasks)

    async def _execute_one(
        self,
        tool_call: dict[str, Any],
        context: Any,
    ) -> ToolResult:
        """Execute a single tool call."""
        import time

        start = time.perf_counter()
        call_id = tool_call.get("id", "")
        func_info = tool_call.get("function", {})
        name = func_info.get("name", "")
        args_str = func_info.get("arguments", "{}")

        # Check tool exists
        if name not in self.tools:
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output="",
                success=False,
                error=f"Unknown tool: {name}",
            )

        tool = self.tools[name]

        # Parse arguments
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output="",
                success=False,
                error=f"Invalid JSON arguments: {e}",
            )

        # Execute with timeout and semaphore
        async with self._semaphore:
            try:
                # Inject context if needed
                if tool.inject_context:
                    result = await asyncio.wait_for(
                        tool.func(context, **args),
                        timeout=self.default_timeout_ms / 1000,
                    )
                else:
                    result = await asyncio.wait_for(
                        tool.func(**args),
                        timeout=self.default_timeout_ms / 1000,
                    )

                # Convert result to string
                if isinstance(result, str):
                    output = result
                else:
                    output = json.dumps(result, default=str)

                return ToolResult(
                    tool_call_id=call_id,
                    name=name,
                    output=output,
                    success=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            except asyncio.TimeoutError:
                return ToolResult(
                    tool_call_id=call_id,
                    name=name,
                    output="",
                    success=False,
                    error=f"Tool execution timed out after {self.default_timeout_ms}ms",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                return ToolResult(
                    tool_call_id=call_id,
                    name=name,
                    output="",
                    success=False,
                    error=str(e),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )


# --- Decorators for defining tools ---


def tool(
    name: str | None = None,
    description: str | None = None,
    inject_context: bool = False,
):
    """
    Decorator to define a tool from an async function.

    Usage:
        @tool(description="Search the web")
        async def search(query: str) -> str:
            return await do_search(query)

        @tool(inject_context=True)
        async def run_code(ctx: Context, code: str) -> str:
            sandbox = ctx.resources["sandbox"]
            return await sandbox.execute(code)
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> ToolDef:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # Generate parameter schema from type hints
        params = _generate_schema(func, inject_context)

        return ToolDef(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters=params,
            inject_context=inject_context,
        )

    return decorator


def _generate_schema(
    func: Callable[..., Awaitable[Any]],
    skip_first: bool = False,
) -> dict[str, Any]:
    """Generate JSON Schema from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    params = list(sig.parameters.items())
    if skip_first and params:
        params = params[1:]  # Skip context parameter

    for param_name, param in params:
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, Any)
        schema = _type_to_schema(param_type)
        properties[param_name] = schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_schema(t: type) -> dict[str, Any]:
    """Convert Python type to JSON Schema."""
    # Handle basic types
    if t is str:
        return {"type": "string"}
    elif t is int:
        return {"type": "integer"}
    elif t is float:
        return {"type": "number"}
    elif t is bool:
        return {"type": "boolean"}
    elif t is list or (hasattr(t, "__origin__") and t.__origin__ is list):
        return {"type": "array"}
    elif t is dict or (hasattr(t, "__origin__") and t.__origin__ is dict):
        return {"type": "object"}
    else:
        return {"type": "string"}  # Fallback


# --- Example tools ---


@tool(description="Execute Python code and return the output")
async def python_exec(code: str) -> str:
    """Execute Python code in a sandboxed environment."""
    # This would be implemented with actual sandbox
    return f"Executed: {code[:50]}..."


@tool(description="Read a file from the filesystem", inject_context=True)
async def read_file(ctx: Any, path: str) -> str:
    """Read file contents. Requires filesystem access in context."""
    # ctx.resources["filesystem"].read(path)
    return f"Contents of {path}"
