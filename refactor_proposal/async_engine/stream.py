"""
Async stream utilities for backpressure-aware data flow.

Key patterns:
- Bounded buffers prevent memory explosion
- Cancellation propagates cleanly
- Multiple consumers can tap into streams
"""

from __future__ import annotations

import asyncio
from typing import TypeVar, AsyncIterator, Callable, Awaitable, Generic
from dataclasses import dataclass
from contextlib import asynccontextmanager

T = TypeVar("T")
R = TypeVar("R")


class AsyncStream(Generic[T]):
    """
    Async stream with backpressure support.

    Producers block when buffer is full (backpressure).
    Consumers block when buffer is empty.
    """

    def __init__(self, max_size: int = 100):
        self._queue: asyncio.Queue[T | None] = asyncio.Queue(maxsize=max_size)
        self._closed = False
        self._error: Exception | None = None

    async def put(self, item: T) -> None:
        """Add item to stream (blocks if full)."""
        if self._closed:
            raise RuntimeError("Stream is closed")
        await self._queue.put(item)

    def put_nowait(self, item: T) -> None:
        """Add item without blocking (raises if full)."""
        if self._closed:
            raise RuntimeError("Stream is closed")
        self._queue.put_nowait(item)

    async def close(self, error: Exception | None = None) -> None:
        """Close the stream."""
        self._closed = True
        self._error = error
        await self._queue.put(None)  # Sentinel

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        item = await self._queue.get()
        if item is None:
            if self._error:
                raise self._error
            raise StopAsyncIteration
        return item


async def map_stream(
    source: AsyncIterator[T],
    func: Callable[[T], Awaitable[R]],
    max_concurrent: int = 10,
) -> AsyncIterator[R]:
    """
    Map a function over a stream with bounded concurrency.

    Results come out as they complete (may be out of order).
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    result_queue: asyncio.Queue[R | Exception | None] = asyncio.Queue()
    pending_count = 0
    source_done = False

    async def process_one(item: T) -> None:
        nonlocal pending_count
        async with semaphore:
            try:
                result = await func(item)
                await result_queue.put(result)
            except Exception as e:
                await result_queue.put(e)
            finally:
                pending_count -= 1
                if source_done and pending_count == 0:
                    await result_queue.put(None)

    async def producer() -> None:
        nonlocal pending_count, source_done
        async for item in source:
            pending_count += 1
            asyncio.create_task(process_one(item))
        source_done = True
        if pending_count == 0:
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


async def buffer_stream(
    source: AsyncIterator[T],
    size: int = 10,
) -> AsyncIterator[T]:
    """
    Buffer a stream to smooth out production/consumption rate differences.
    """
    buffer: asyncio.Queue[T | None] = asyncio.Queue(maxsize=size)

    async def fill_buffer() -> None:
        try:
            async for item in source:
                await buffer.put(item)
        finally:
            await buffer.put(None)

    filler = asyncio.create_task(fill_buffer())

    try:
        while True:
            item = await buffer.get()
            if item is None:
                break
            yield item
    finally:
        filler.cancel()


async def batch_stream(
    source: AsyncIterator[T],
    batch_size: int,
    timeout_ms: float = 100,
) -> AsyncIterator[list[T]]:
    """
    Batch items from a stream.

    Yields when batch is full OR timeout reached (whichever first).
    """
    batch: list[T] = []

    async def get_with_timeout() -> T | None:
        try:
            return await asyncio.wait_for(
                source.__anext__(),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            return None
        except StopAsyncIteration:
            return None

    while True:
        item = await get_with_timeout()

        if item is not None:
            batch.append(item)

        if len(batch) >= batch_size or (item is None and batch):
            yield batch
            batch = []

        if item is None and not batch:
            # Source exhausted and batch empty
            try:
                # Try one more time to confirm source is done
                await source.__anext__()
            except StopAsyncIteration:
                break


async def merge_streams(
    *sources: AsyncIterator[T],
) -> AsyncIterator[T]:
    """
    Merge multiple streams into one, yielding items as they arrive.
    """
    result_queue: asyncio.Queue[T | None] = asyncio.Queue()
    active_count = len(sources)

    async def forward(source: AsyncIterator[T]) -> None:
        nonlocal active_count
        try:
            async for item in source:
                await result_queue.put(item)
        finally:
            active_count -= 1
            if active_count == 0:
                await result_queue.put(None)

    tasks = [asyncio.create_task(forward(s)) for s in sources]

    try:
        while True:
            item = await result_queue.get()
            if item is None:
                break
            yield item
    finally:
        for task in tasks:
            task.cancel()


@dataclass
class Progress(Generic[T]):
    """Progress update for stream processing."""

    item: T
    completed: int
    total: int | None

    @property
    def percent(self) -> float | None:
        if self.total is None:
            return None
        return (self.completed / self.total) * 100


async def with_progress(
    source: AsyncIterator[T],
    total: int | None = None,
) -> AsyncIterator[Progress[T]]:
    """
    Wrap a stream with progress tracking.
    """
    completed = 0
    async for item in source:
        completed += 1
        yield Progress(item=item, completed=completed, total=total)


@asynccontextmanager
async def cancellable_stream(
    source: AsyncIterator[T],
) -> AsyncIterator[AsyncIterator[T]]:
    """
    Wrap a stream with cancellation support.

    Usage:
        async with cancellable_stream(source) as stream:
            async for item in stream:
                if should_stop:
                    break  # Cleanly cancels
    """
    cancel_event = asyncio.Event()

    async def wrapped() -> AsyncIterator[T]:
        async for item in source:
            if cancel_event.is_set():
                break
            yield item

    try:
        yield wrapped()
    finally:
        cancel_event.set()
