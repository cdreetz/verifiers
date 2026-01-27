"""
ExecutionContext - Encapsulates all resources needed for rollout execution.

This replaces the pattern of threading 5+ separate parameters (client, model,
sampling_args, gen_sem, score_sem) through multiple function layers.

Usage:
    # Create context once at the top level
    ctx = ExecutionContext(
        client=client,
        model=model,
        sampling_args=sampling_args,
        gen_concurrency=32,
        score_concurrency=16,
    )

    # Pass context instead of individual parameters
    state = await env.rollout_with_context(input, ctx)

    # Use context managers for concurrency control
    async with ctx.generation_limit():
        response = await client.chat.completions.create(...)

    async with ctx.scoring_limit():
        score = await rubric.score(...)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncContextManager, Optional

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from verifiers.types import SamplingArgs


class _NullAsyncContext:
    """No-op async context manager for unlimited concurrency."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


_NULL_CONTEXT = _NullAsyncContext()


@dataclass
class ExecutionContext:
    """
    Encapsulates all resources needed for rollout execution.

    This class consolidates the various parameters that are typically threaded
    through multiple function layers during rollout execution:
    - OpenAI client
    - Model name
    - Sampling arguments
    - Concurrency limits (generation and scoring semaphores)
    - Feature flags (interleaved rollouts, scoring, max sequence length)

    The context provides async context managers for concurrency control,
    eliminating the need to manually manage semaphores.

    Attributes:
        client: AsyncOpenAI client for API calls
        model: Model name/identifier
        sampling_args: Generation sampling parameters
        gen_concurrency: Max concurrent generation requests (None = unlimited)
        score_concurrency: Max concurrent scoring operations (None = unlimited)
        interleaved_rollouts: Whether to use interleaved tokenization (Prime-RL feature)
        score_rollouts: Whether to score rollouts after generation
        max_seq_len: Maximum sequence length for truncation

    Example:
        ctx = ExecutionContext(
            client=AsyncOpenAI(),
            model="gpt-4",
            sampling_args={"temperature": 0.7},
            gen_concurrency=32,
            score_concurrency=16,
        )

        async with ctx.generation_limit():
            response = await ctx.client.chat.completions.create(...)
    """

    client: AsyncOpenAI
    model: str
    sampling_args: SamplingArgs = field(default_factory=dict)

    # Concurrency limits (None or 0 = unlimited)
    gen_concurrency: Optional[int] = None
    score_concurrency: Optional[int] = None

    # Feature flags
    interleaved_rollouts: bool = False
    score_rollouts: bool = True
    max_seq_len: Optional[int] = None

    # Internal semaphores (created lazily)
    _gen_sem: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _score_sem: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Initialize semaphores based on concurrency limits."""
        self._initialize_semaphores()

    def _initialize_semaphores(self):
        """Create semaphores if concurrency limits are set."""
        if self._initialized:
            return

        if self.gen_concurrency and self.gen_concurrency > 0:
            self._gen_sem = asyncio.Semaphore(self.gen_concurrency)

        if self.score_concurrency and self.score_concurrency > 0:
            self._score_sem = asyncio.Semaphore(self.score_concurrency)

        self._initialized = True

    @property
    def gen_semaphore(self) -> AsyncContextManager:
        """Get generation semaphore (or null context if unlimited)."""
        return self._gen_sem if self._gen_sem else _NULL_CONTEXT

    @property
    def score_semaphore(self) -> AsyncContextManager:
        """Get scoring semaphore (or null context if unlimited)."""
        return self._score_sem if self._score_sem else _NULL_CONTEXT

    @asynccontextmanager
    async def generation_limit(self):
        """
        Async context manager for generation concurrency control.

        Usage:
            async with ctx.generation_limit():
                response = await client.chat.completions.create(...)
        """
        if self._gen_sem:
            async with self._gen_sem:
                yield
        else:
            yield

    @asynccontextmanager
    async def scoring_limit(self):
        """
        Async context manager for scoring concurrency control.

        Usage:
            async with ctx.scoring_limit():
                score = await rubric.score(...)
        """
        if self._score_sem:
            async with self._score_sem:
                yield
        else:
            yield

    def with_sampling_args(self, **kwargs) -> "ExecutionContext":
        """
        Return a new context with updated sampling args.

        This creates a shallow copy with merged sampling args,
        useful for per-request overrides.
        """
        new_args = {**self.sampling_args, **kwargs}
        return ExecutionContext(
            client=self.client,
            model=self.model,
            sampling_args=new_args,
            gen_concurrency=self.gen_concurrency,
            score_concurrency=self.score_concurrency,
            interleaved_rollouts=self.interleaved_rollouts,
            score_rollouts=self.score_rollouts,
            max_seq_len=self.max_seq_len,
            _gen_sem=self._gen_sem,  # Share semaphores
            _score_sem=self._score_sem,
            _initialized=True,
        )

    def with_model(self, model: str) -> "ExecutionContext":
        """Return a new context with a different model."""
        return ExecutionContext(
            client=self.client,
            model=model,
            sampling_args=self.sampling_args,
            gen_concurrency=self.gen_concurrency,
            score_concurrency=self.score_concurrency,
            interleaved_rollouts=self.interleaved_rollouts,
            score_rollouts=self.score_rollouts,
            max_seq_len=self.max_seq_len,
            _gen_sem=self._gen_sem,
            _score_sem=self._score_sem,
            _initialized=True,
        )

    @classmethod
    def from_legacy_params(
        cls,
        client: "AsyncOpenAI",
        model: str,
        sampling_args: SamplingArgs,
        gen_sem: Optional[AsyncContextManager] = None,
        score_sem: Optional[AsyncContextManager] = None,
        interleaved_rollouts: bool = False,
        score_rollouts: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> "ExecutionContext":
        """
        Create ExecutionContext from legacy parameters.

        This is a compatibility bridge for code that still passes
        semaphores directly. The semaphores are stored but the
        context will use them via the legacy properties.
        """
        ctx = cls(
            client=client,
            model=model,
            sampling_args=sampling_args,
            interleaved_rollouts=interleaved_rollouts,
            score_rollouts=score_rollouts,
            max_seq_len=max_seq_len,
        )
        # Store legacy semaphores directly
        if gen_sem is not None and not isinstance(gen_sem, _NullAsyncContext):
            ctx._gen_sem = gen_sem  # type: ignore
        if score_sem is not None and not isinstance(score_sem, _NullAsyncContext):
            ctx._score_sem = score_sem  # type: ignore
        ctx._initialized = True
        return ctx

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to dict (excluding client and semaphores)."""
        return {
            "model": self.model,
            "sampling_args": self.sampling_args,
            "gen_concurrency": self.gen_concurrency,
            "score_concurrency": self.score_concurrency,
            "interleaved_rollouts": self.interleaved_rollouts,
            "score_rollouts": self.score_rollouts,
            "max_seq_len": self.max_seq_len,
        }
