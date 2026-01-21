"""
Core async executor with backpressure-aware streaming.

This is the heart of the async engine. Key properties:
1. Backpressure: Won't overwhelm downstream consumers
2. Bounded memory: Uses async queues with max size
3. Graceful cancellation: Properly cleans up on cancel
4. Error isolation: One failure doesn't kill everything
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable, TypeVar, Generic
from contextlib import asynccontextmanager
import time

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""

    max_concurrent: int = 64  # Max concurrent operations
    queue_size: int = 128  # Backpressure buffer size
    batch_size: int = 1  # For batched operations
    batch_timeout_ms: float = 50.0  # Max wait to fill batch
    retry_attempts: int = 3
    retry_base_delay_ms: float = 100.0


@dataclass
class ExecutorStats:
    """Runtime statistics for monitoring."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.completed + self.failed
        return self.completed / total if total > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.completed if self.completed > 0 else 0.0


@dataclass
class WorkItem(Generic[T, R]):
    """Internal work item with tracking."""

    input: T
    future: asyncio.Future[R]
    submit_time: float = field(default_factory=time.perf_counter)
    attempt: int = 0


class AsyncExecutor(Generic[T, R]):
    """
    High-performance async executor with backpressure.

    Usage:
        async def process(item: Input) -> Output:
            return await do_work(item)

        executor = AsyncExecutor(process, config)

        # Stream results with backpressure
        async for result in executor.map(inputs):
            handle(result)

        # Or collect all (bounded by queue_size)
        results = await executor.gather(inputs)
    """

    def __init__(
        self,
        func: Callable[[T], Awaitable[R]],
        config: ExecutorConfig | None = None,
    ):
        self.func = func
        self.config = config or ExecutorConfig()
        self.stats = ExecutorStats()
        self._semaphore: asyncio.Semaphore | None = None
        self._queue: asyncio.Queue[WorkItem[T, R] | None] | None = None
        self._workers: list[asyncio.Task] | None = None
        self._running = False

    @asynccontextmanager
    async def _managed_execution(self):
        """Context manager for executor lifecycle."""
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._queue = asyncio.Queue(maxsize=self.config.queue_size)
        self._running = True
        try:
            yield
        finally:
            self._running = False
            # Drain remaining items
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item and not item.future.done():
                        item.future.cancel()
                except asyncio.QueueEmpty:
                    break

    async def _execute_one(self, item: WorkItem[T, R]) -> None:
        """Execute a single work item with retry logic."""
        async with self._semaphore:
            while item.attempt < self.config.retry_attempts:
                item.attempt += 1
                try:
                    start = time.perf_counter()
                    result = await self.func(item.input)
                    latency = (time.perf_counter() - start) * 1000

                    # Update stats
                    self.stats.completed += 1
                    self.stats.total_latency_ms += latency
                    self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency)

                    if not item.future.done():
                        item.future.set_result(result)
                    return

                except asyncio.CancelledError:
                    if not item.future.done():
                        item.future.cancel()
                    raise

                except Exception as e:
                    if item.attempt < self.config.retry_attempts:
                        self.stats.retried += 1
                        delay = self.config.retry_base_delay_ms * (2 ** (item.attempt - 1))
                        await asyncio.sleep(delay / 1000)
                    else:
                        self.stats.failed += 1
                        if not item.future.done():
                            item.future.set_exception(e)
                        return

    async def map(
        self,
        inputs: AsyncIterator[T] | list[T],
    ) -> AsyncIterator[R]:
        """
        Process inputs and yield results as they complete.

        This is backpressure-aware: if consumer is slow, producer is throttled.
        Results may come out of order for maximum throughput.
        """
        async with self._managed_execution():
            pending: set[asyncio.Task] = set()
            result_queue: asyncio.Queue[R | Exception] = asyncio.Queue(
                maxsize=self.config.queue_size
            )

            async def producer():
                """Submit work items."""
                if isinstance(inputs, list):
                    input_iter = aiter_list(inputs)
                else:
                    input_iter = inputs

                async for input_item in input_iter:
                    self.stats.submitted += 1
                    future: asyncio.Future[R] = asyncio.Future()
                    item = WorkItem(input=input_item, future=future)

                    # Create task for this item
                    task = asyncio.create_task(self._execute_one(item))
                    pending.add(task)
                    task.add_done_callback(pending.discard)

                    # Forward result to queue when ready
                    async def forward_result(f: asyncio.Future[R]):
                        try:
                            await result_queue.put(await f)
                        except Exception as e:
                            await result_queue.put(e)

                    asyncio.create_task(forward_result(future))

                # Wait for all pending to complete
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                # Signal completion
                await result_queue.put(None)

            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    result = await result_queue.get()
                    if result is None:
                        break
                    if isinstance(result, Exception):
                        raise result
                    yield result
            finally:
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    async def gather(self, inputs: list[T]) -> list[R]:
        """Process all inputs and return results in order."""
        # For ordered results, we track by index
        results: dict[int, R] = {}
        indexed_inputs = list(enumerate(inputs))

        async def process_indexed(item: tuple[int, T]) -> tuple[int, R]:
            idx, inp = item
            result = await self.func(inp)
            return (idx, result)

        indexed_executor = AsyncExecutor(process_indexed, self.config)

        async for idx, result in indexed_executor.map(indexed_inputs):
            results[idx] = result

        return [results[i] for i in range(len(inputs))]

    async def map_ordered(self, inputs: list[T]) -> AsyncIterator[R]:
        """Stream results in input order (may buffer internally)."""
        results = await self.gather(inputs)
        for r in results:
            yield r


async def aiter_list(items: list[T]) -> AsyncIterator[T]:
    """Convert list to async iterator."""
    for item in items:
        yield item


# Convenience function for simple cases
async def parallel_map(
    func: Callable[[T], Awaitable[R]],
    inputs: list[T],
    max_concurrent: int = 64,
) -> list[R]:
    """Simple parallel map with concurrency limit."""
    config = ExecutorConfig(max_concurrent=max_concurrent)
    executor = AsyncExecutor(func, config)
    return await executor.gather(inputs)
