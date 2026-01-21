"""
Verifiers v2 - Clean async-first architecture with backward compatibility.

Public API preserves v1 interface while using new internals.

Usage (unchanged from v1):

    import verifiers as vf

    # Define reward function
    @vf.reward
    async def correct(completion: str, answer: str) -> float:
        return 1.0 if completion.strip() == answer.strip() else 0.0

    # Create environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=vf.Rubric(funcs=[correct]),
        system_prompt="Answer concisely.",
    )

    # Generate
    outputs = await env.generate(
        client=client,
        model="gpt-4",
        max_concurrent=64,
    )

New v2 API (for advanced users):

    from verifiers.v2 import (
        Environment,
        Context,
        Rubric,
        ResourcePool,
        AsyncExecutor,
    )

    # Compose environment from primitives
    env = Environment(
        env_id="my-env",
        turn_handler=my_turn_handler,
        scorer=my_rubric,
        stop_conditions=[max_turns, task_complete],
    )

    # Stream results with backpressure
    async for result in env.generate(inputs, ctx):
        process(result)
"""

# --- v1 Compatibility API (default) ---

# Core types (preserved)
from .core.types import (
    Message,
    RolloutInput,
    RolloutResult,
    TrajectoryStep,
)

# Backward-compatible environment classes
from .compat.v1_env import (
    Environment,
    SingleTurnEnv,
    MultiTurnEnv,
)

# Decorators
from .core.protocols import StopCondition

def stop(priority: int = 50):
    """Decorator to mark a method as a stop condition."""
    def decorator(func):
        func._is_stop_condition = True
        func._priority = priority
        return func
    return decorator

def cleanup(priority: int = 50):
    """Decorator to mark a method for per-rollout cleanup."""
    def decorator(func):
        func._is_cleanup = True
        func._priority = priority
        return func
    return decorator

def teardown(priority: int = 50):
    """Decorator for environment shutdown."""
    def decorator(func):
        func._is_teardown = True
        func._priority = priority
        return func
    return decorator

# Scoring
from .scoring.rubric import (
    Rubric,
    RewardComponent,
    reward,
    group_reward,
)

# Parsing
# from .parsing import Parser, XMLParser, ThinkParser

# Error types
from .core.errors import (
    VerifierError,
    RolloutError,
    ScoringError,
    ResourceError,
)

# --- v2 API (explicit import) ---

# New primitives available via:
# from verifiers.v2 import ...
# from verifiers.async_engine import ...
# from verifiers.resources import ...

__version__ = "2.0.0"

__all__ = [
    # Types
    "Message",
    "RolloutInput",
    "RolloutResult",
    "TrajectoryStep",
    # Environments (v1 compat)
    "Environment",
    "SingleTurnEnv",
    "MultiTurnEnv",
    # Decorators
    "stop",
    "cleanup",
    "teardown",
    # Scoring
    "Rubric",
    "RewardComponent",
    "reward",
    "group_reward",
    # Errors
    "VerifierError",
    "RolloutError",
    "ScoringError",
    "ResourceError",
]
