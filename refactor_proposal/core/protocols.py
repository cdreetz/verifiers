"""Protocols (interfaces) for the system - enables composition without inheritance."""

from __future__ import annotations

from typing import Protocol, AsyncIterator, Any, runtime_checkable

from .types import RolloutInput, RolloutResult, Message


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients - any client implementing this works."""

    async def complete(
        self,
        messages: tuple[Message, ...],
        *,
        model: str,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: tuple[dict[str, Any], ...] | None = None,
        stop: tuple[str, ...] | None = None,
    ) -> Message:
        """Generate a single completion."""
        ...

    async def complete_batch(
        self,
        message_batches: list[tuple[Message, ...]],
        *,
        model: str,
        **kwargs: Any,
    ) -> list[Message]:
        """Generate completions for a batch (for efficiency)."""
        ...


@runtime_checkable
class Environment(Protocol):
    """Protocol for environments - implement this, don't inherit."""

    @property
    def env_id(self) -> str:
        """Unique identifier for this environment."""
        ...

    async def rollout(self, input: RolloutInput, ctx: "Context") -> RolloutResult:
        """Execute a single rollout."""
        ...


@runtime_checkable
class Scorer(Protocol):
    """Protocol for scoring rollout results."""

    async def score(self, result: RolloutResult, ctx: "Context") -> RolloutResult:
        """Score a rollout, filling in reward and metrics."""
        ...

    async def score_batch(
        self, results: list[RolloutResult], ctx: "Context"
    ) -> list[RolloutResult]:
        """Score a batch (for group-based rewards)."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Protocol for parsing completions."""

    def parse(self, content: str) -> dict[str, str]:
        """Extract structured fields from completion."""
        ...


@runtime_checkable
class Resource(Protocol):
    """Protocol for managed resources (sandboxes, connections, etc.)."""

    async def initialize(self) -> None:
        """Set up the resource."""
        ...

    async def cleanup(self) -> None:
        """Clean up the resource."""
        ...

    async def health_check(self) -> bool:
        """Check if resource is healthy."""
        ...


@runtime_checkable
class StopCondition(Protocol):
    """Protocol for stop conditions in multi-turn environments."""

    @property
    def name(self) -> str:
        """Name of this stop condition."""
        ...

    @property
    def priority(self) -> int:
        """Lower = checked first."""
        ...

    async def check(self, result: RolloutResult, ctx: "Context") -> bool:
        """Return True if rollout should stop."""
        ...


class Context(Protocol):
    """Execution context passed through the system."""

    @property
    def client(self) -> LLMClient:
        ...

    @property
    def model(self) -> str:
        ...

    @property
    def cancel_requested(self) -> bool:
        ...

    def request_cancel(self) -> None:
        ...
