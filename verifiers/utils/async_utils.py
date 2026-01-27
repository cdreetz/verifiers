import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, AsyncContextManager, Callable, Optional

logger = logging.getLogger(__name__)


async def maybe_await(func: Callable, *args, **kwargs):
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class NullAsyncContext:
    """No-op async context manager for unlimited concurrency."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


# Singleton instance for reuse
_NULL_CONTEXT = NullAsyncContext()


async def maybe_semaphore(
    limit: Optional[int] = None,
) -> AsyncContextManager:
    """
    Return either a real semaphore (if limit is set),
    or a no-op context manager (if limit is None or <= 0).

    Usage:
    maybe_sem = await maybe_semaphore(10)
    async with maybe_sem:
        await do_something()
    """
    if limit and limit > 0:
        return asyncio.Semaphore(limit)
    else:
        return _NULL_CONTEXT


def create_semaphore(limit: Optional[int] = None) -> AsyncContextManager:
    """
    Synchronous version of maybe_semaphore.

    Returns a real semaphore if limit > 0, otherwise returns a no-op context.
    Unlike maybe_semaphore, this is not async.

    Usage:
        sem = create_semaphore(10)
        async with sem:
            await do_something()
    """
    if limit and limit > 0:
        return asyncio.Semaphore(limit)
    return _NULL_CONTEXT


@dataclass
class ResourcePool:
    """
    Manages concurrency limits and other shared resources.

    This provides a centralized place to manage semaphores and other
    resources that need to be shared across async tasks.

    Usage:
        pool = ResourcePool(gen_limit=32, score_limit=16)

        async with pool.generation():
            response = await client.chat.completions.create(...)

        async with pool.scoring():
            score = await rubric.score(...)

    The pool can also track metrics about resource usage:
        print(pool.stats)  # {"gen_acquired": 100, "gen_released": 100, ...}
    """

    gen_limit: Optional[int] = None
    score_limit: Optional[int] = None

    # Internal state
    _gen_sem: Optional[asyncio.Semaphore] = field(default=None, init=False, repr=False)
    _score_sem: Optional[asyncio.Semaphore] = field(
        default=None, init=False, repr=False
    )
    _stats: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self._initialize()

    def _initialize(self):
        """Initialize semaphores based on limits."""
        if self._initialized:
            return

        if self.gen_limit and self.gen_limit > 0:
            self._gen_sem = asyncio.Semaphore(self.gen_limit)

        if self.score_limit and self.score_limit > 0:
            self._score_sem = asyncio.Semaphore(self.score_limit)

        self._stats = {
            "gen_acquired": 0,
            "gen_released": 0,
            "score_acquired": 0,
            "score_released": 0,
        }
        self._initialized = True

    @asynccontextmanager
    async def generation(self):
        """Acquire generation concurrency slot."""
        if self._gen_sem:
            async with self._gen_sem:
                self._stats["gen_acquired"] += 1
                try:
                    yield
                finally:
                    self._stats["gen_released"] += 1
        else:
            yield

    @asynccontextmanager
    async def scoring(self):
        """Acquire scoring concurrency slot."""
        if self._score_sem:
            async with self._score_sem:
                self._stats["score_acquired"] += 1
                try:
                    yield
                finally:
                    self._stats["score_released"] += 1
        else:
            yield

    @property
    def gen_semaphore(self) -> AsyncContextManager:
        """Get generation semaphore (or null context if unlimited)."""
        return self._gen_sem if self._gen_sem else _NULL_CONTEXT

    @property
    def score_semaphore(self) -> AsyncContextManager:
        """Get scoring semaphore (or null context if unlimited)."""
        return self._score_sem if self._score_sem else _NULL_CONTEXT

    @property
    def stats(self) -> dict[str, int]:
        """Get resource usage statistics."""
        return dict(self._stats)

    def reset_stats(self):
        """Reset usage statistics."""
        for key in self._stats:
            self._stats[key] = 0

    @property
    def gen_available(self) -> int:
        """Number of available generation slots (or -1 if unlimited)."""
        if self._gen_sem:
            return self._gen_sem._value
        return -1

    @property
    def score_available(self) -> int:
        """Number of available scoring slots (or -1 if unlimited)."""
        if self._score_sem:
            return self._score_sem._value
        return -1


async def gather_with_concurrency(
    limit: int,
    *coros,
    return_exceptions: bool = False,
):
    """
    Like asyncio.gather but with a concurrency limit.

    Args:
        limit: Maximum number of concurrent tasks
        *coros: Coroutines to run
        return_exceptions: If True, exceptions are returned instead of raised

    Returns:
        List of results in the same order as input coroutines
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_coro(coro, index):
        async with semaphore:
            return await coro

    tasks = [limited_coro(coro, i) for i, coro in enumerate(coros)]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


class EventLoopLagMonitor:
    """A class to monitor how busy the main event loop is."""

    def __init__(
        self,
        measure_interval: float = 0.1,
        max_measurements: int = int(1e5),
        logger: Any | None = None,
    ):
        assert measure_interval > 0 and max_measurements > 0
        self.measure_interval = measure_interval
        self.max_measurements = max_measurements
        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.lags = []
        self.logger.info(
            f"Event loop lag monitor initialized with measure_interval={self.measure_interval} and max_measurements={self.max_measurements}"
        )

    async def measure_lag(self):
        """Measures event loop lag by asynchronously sleeping for interval seconds"""
        next_time = perf_counter() + self.measure_interval
        await asyncio.sleep(self.measure_interval)
        now = perf_counter()
        lag = now - next_time
        return lag

    def get_lags(self) -> list[float]:
        """Get the list of measured event loop lags."""
        return self.lags

    def reset_lags(self):
        """Reset the list of measured event loop lags."""
        self.lags = []

    async def run(self):
        """Loop to measure event loop lag. Should be started as background task."""
        while True:
            lag = await self.measure_lag()
            self.lags.append(lag)
            if len(self.lags) > self.max_measurements:
                self.lags.pop(0)

    def run_in_background(self):
        """Run the event loop lag monitor as a background task."""
        return asyncio.create_task(self.run())
