"""
Verifiers Core - Clean abstractions for environment development.

This module provides improved abstractions for building environments:

- ExecutionContext: Encapsulates all resources needed for rollout execution,
  replacing the pattern of threading 5+ separate parameters through functions.

- LifecycleRegistry: Explicit lifecycle hook registration without reflection,
  making code more predictable and easier to trace.

- ResourcePool: Centralized concurrency management with usage tracking.

Example usage:

    from verifiers.core import ExecutionContext, LifecycleRegistry

    # Create execution context
    ctx = ExecutionContext(
        client=client,
        model="gpt-4",
        sampling_args={"temperature": 0.7},
        gen_concurrency=32,
        score_concurrency=16,
    )

    # Use context managers for concurrency control
    async with ctx.generation_limit():
        response = await ctx.client.chat.completions.create(...)

    # Explicit lifecycle registration
    class MyEnv(MultiTurnEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.lifecycle = LifecycleRegistry()
            self.lifecycle.register_stop(self._game_won, priority=10)
            self.lifecycle.register_cleanup(self._save_logs)
"""

from verifiers.core.context import ExecutionContext
from verifiers.core.registry import (
    CleanupHook,
    LifecycleRegistry,
    StopCondition,
    TeardownHook,
    create_default_registry,
)

__all__ = [
    "ExecutionContext",
    "LifecycleRegistry",
    "StopCondition",
    "CleanupHook",
    "TeardownHook",
    "create_default_registry",
]
