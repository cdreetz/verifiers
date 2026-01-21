"""
Example: Complex sandbox-based coding environment using the new architecture.

This demonstrates how the new primitives compose for complex environments:
- Resource pooling for sandboxes
- Tool execution with state injection
- Multi-turn conversation with proper lifecycle
- Async-native throughout
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..core.types import RolloutInput, RolloutResult, Message, Role
from ..core.protocols import Context
from ..env.base import Environment, EnvironmentConfig
from ..resources.pool import ResourcePool, PoolConfig
from ..tools.executor import ToolExecutor, ToolDef, tool
from ..scoring.rubric import Rubric, reward, RewardComponent
from ..async_engine import AsyncExecutor, ExecutorConfig


# --- Sandbox Resource ---


@dataclass
class Sandbox:
    """Mock sandbox for demonstration."""

    id: str
    _alive: bool = True

    @classmethod
    async def create(cls) -> "Sandbox":
        """Create a new sandbox."""
        import uuid

        await asyncio.sleep(0.1)  # Simulate creation time
        return cls(id=str(uuid.uuid4()))

    async def execute(self, code: str, timeout_ms: float = 30000) -> str:
        """Execute code in the sandbox."""
        if not self._alive:
            raise RuntimeError("Sandbox is dead")
        await asyncio.sleep(0.05)  # Simulate execution
        return f"Output of: {code[:50]}..."

    async def reset(self) -> None:
        """Reset sandbox state."""
        await asyncio.sleep(0.01)

    async def terminate(self) -> None:
        """Terminate the sandbox."""
        self._alive = False

    async def is_alive(self) -> bool:
        return self._alive


# --- Tools that use sandbox ---


@tool(description="Execute Python code and return the result", inject_context=True)
async def run_python(ctx: Context, code: str) -> str:
    """Execute Python code in the sandbox."""
    sandbox: Sandbox = ctx.resources["sandbox"]
    return await sandbox.execute(code)


@tool(description="Read a file from the sandbox filesystem", inject_context=True)
async def read_file(ctx: Context, path: str) -> str:
    """Read file contents."""
    sandbox: Sandbox = ctx.resources["sandbox"]
    return await sandbox.execute(f"cat {path}")


@tool(description="Write content to a file", inject_context=True)
async def write_file(ctx: Context, path: str, content: str) -> str:
    """Write to a file."""
    sandbox: Sandbox = ctx.resources["sandbox"]
    return await sandbox.execute(f"echo '{content}' > {path}")


# --- Reward functions ---


@reward(weight=1.0)
async def tests_pass(completion: str, answer: str, result: RolloutResult) -> float:
    """Check if the solution passes tests."""
    # In real impl, would run tests in sandbox
    # Here just check if "pass" or "success" in output
    last_output = ""
    for step in result.trajectory:
        for msg in step.messages:
            if msg.role == Role.TOOL:
                last_output = msg.content

    if "pass" in last_output.lower() or "success" in last_output.lower():
        return 1.0
    return 0.0


@reward(weight=0.2, metric_only=True)
async def code_length(completion: str) -> float:
    """Penalize overly long solutions."""
    lines = completion.count("\n") + 1
    return max(0.0, 1.0 - (lines / 100))


@reward(weight=0.1, metric_only=True)
async def tool_efficiency(result: RolloutResult) -> float:
    """Reward efficient tool use."""
    tool_calls = sum(
        1 for step in result.trajectory for msg in step.messages if msg.tool_calls
    )
    # Ideal is 1-3 tool calls
    if 1 <= tool_calls <= 3:
        return 1.0
    elif tool_calls == 0:
        return 0.0
    else:
        return max(0.0, 1.0 - (tool_calls - 3) * 0.1)


# --- The Environment ---


class SandboxCodingEnv:
    """
    Coding environment with sandbox execution.

    This shows the full pattern:
    1. Resource pool manages sandboxes
    2. Tools execute in sandbox context
    3. Multi-turn loop handles tool calls
    4. Scoring runs after completion
    """

    def __init__(
        self,
        *,
        max_sandboxes: int = 10,
        max_turns: int = 10,
    ):
        self.env_id = "sandbox-coding"

        # Resource pool for sandboxes
        self.sandbox_pool = ResourcePool(
            create_func=Sandbox.create,
            destroy_func=lambda s: s.terminate(),
            health_func=lambda s: s.is_alive(),
            config=PoolConfig(
                max_size=max_sandboxes,
                acquire_timeout_ms=60000,
            ),
        )

        # Tool executor
        self.tool_executor = ToolExecutor(
            tools=[run_python, read_file, write_file],
            default_timeout_ms=30000,
        )

        # Scoring rubric
        self.rubric = Rubric(
            components=[tests_pass, code_length, tool_efficiency],
        )

        self.max_turns = max_turns

    async def __aenter__(self) -> "SandboxCodingEnv":
        await self.sandbox_pool.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.sandbox_pool.close()

    async def rollout(self, input: RolloutInput, ctx: Context) -> RolloutResult:
        """Execute a coding task with sandbox access."""
        import time

        result = RolloutResult(input=input, start_time=time.perf_counter())

        # Acquire sandbox for this rollout
        async with self.sandbox_pool.acquire() as sandbox:
            # Inject sandbox into context
            ctx.resources["sandbox"] = sandbox

            # Build messages
            messages: list[Message] = [
                Message(
                    role=Role.SYSTEM,
                    content="You are a coding assistant. Use the provided tools to write and test code.",
                )
            ]
            messages.extend(input.prompt)

            turn = 0
            while turn < self.max_turns:
                # Get LLM response
                response = await ctx.client.complete(
                    messages=tuple(messages),
                    model=ctx.model,
                    tools=tuple(self.tool_executor.get_openai_tools()),
                    **ctx.sampling_args,
                )

                # Add to trajectory
                from ..core.types import TrajectoryStep

                result.trajectory.append(
                    TrajectoryStep(
                        messages=(response,),
                        token_count=0,
                        step_type="assistant",
                    )
                )
                messages.append(response)

                # Check for tool calls
                if not response.tool_calls:
                    # No tool calls = done
                    result.is_completed = True
                    break

                # Execute tools
                tool_results = await self.tool_executor.execute(
                    [tc.to_openai() for tc in response.tool_calls],
                    context=ctx,
                )

                # Add tool results to conversation
                tool_messages = [
                    Message(
                        role=Role.TOOL,
                        content=tr.output if tr.success else f"Error: {tr.error}",
                        tool_call_id=tr.tool_call_id,
                    )
                    for tr in tool_results
                ]

                result.trajectory.append(
                    TrajectoryStep(
                        messages=tuple(tool_messages),
                        token_count=0,
                        step_type="tool",
                    )
                )
                messages.extend(tool_messages)

                turn += 1

            if turn >= self.max_turns:
                result.is_truncated = True

        # Build completion
        result.completion = tuple(
            m
            for step in result.trajectory
            for m in step.messages
            if m.role == Role.ASSISTANT
        )
        result.end_time = time.perf_counter()

        return result

    async def generate(
        self,
        inputs: list[RolloutInput],
        ctx: Context,
        *,
        max_concurrent: int = 64,
    ):
        """Generate rollouts with parallel execution."""

        async def process_one(input: RolloutInput) -> RolloutResult:
            result = await self.rollout(input, ctx)
            result = await self.rubric.score(result, ctx)
            return result

        # Limit concurrency to available sandboxes
        effective_concurrent = min(
            max_concurrent, self.sandbox_pool.config.max_size
        )

        executor = AsyncExecutor(
            process_one, ExecutorConfig(max_concurrent=effective_concurrent)
        )

        async for result in executor.map(inputs):
            yield result


# --- Example usage ---


async def example_usage():
    """Demonstrate the environment."""
    from datasets import Dataset

    # Create mock dataset
    dataset = Dataset.from_dict(
        {
            "example_id": [0, 1, 2],
            "prompt": [
                "Write a function to calculate fibonacci numbers",
                "Write a function to check if a string is a palindrome",
                "Write a function to sort a list",
            ],
            "answer": ["", "", ""],  # Tests would be here
            "task": ["coding", "coding", "coding"],
        }
    )

    # Create mock client
    class MockClient:
        async def complete(self, messages, model, **kwargs):
            return Message(
                role=Role.ASSISTANT,
                content="Here's the solution...",
                tool_calls=None,  # Would have tool calls in real usage
            )

    # Run environment
    async with SandboxCodingEnv(max_sandboxes=5) as env:
        ctx = Context(
            client=MockClient(),
            model="gpt-4",
            sampling_args={"temperature": 0.7},
        )

        inputs = [
            RolloutInput(
                example_id=ex["example_id"],
                prompt=(Message(role=Role.USER, content=ex["prompt"]),),
                answer=ex["answer"],
                task=ex["task"],
            )
            for ex in dataset
        ]

        results = []
        async for result in env.generate(inputs, ctx, max_concurrent=3):
            results.append(result)
            print(f"Completed {result.input.example_id}: reward={result.reward:.2f}")

    return results


if __name__ == "__main__":
    asyncio.run(example_usage())
