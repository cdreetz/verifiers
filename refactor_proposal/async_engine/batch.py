"""
Automatic batching for LLM calls and other batch-friendly operations.

Instead of N individual calls, collects items into batches for efficiency.
Transparent to callers - they still call with single items.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable, TypeVar, Generic
from collections import deque
import time

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for auto-batching."""

    max_batch_size: int = 32  # Max items per batch
    max_wait_ms: float = 20.0  # Max time to wait for batch to fill
    max_concurrent_batches: int = 8  # Max batches in flight


@dataclass
class PendingItem(Generic[T, R]):
    """Item waiting to be batched."""

    input: T
    future: asyncio.Future[R]
    submit_time: float = field(default_factory=time.perf_counter)


class AutoBatcher(Generic[T, R]):
    """
    Automatic batching layer for batch-friendly operations.

    Callers submit individual items, batcher collects them into batches
    and calls the batch function. Results are distributed back to callers.

    Usage:
        async def batch_embed(texts: list[str]) -> list[Embedding]:
            return await embedding_api.embed_batch(texts)

        batcher = AutoBatcher(batch_embed, BatchConfig(max_batch_size=32))

        # These get automatically batched together
        embed1 = await batcher.submit("hello")
        embed2 = await batcher.submit("world")
    """

    def __init__(
        self,
        batch_func: Callable[[list[T]], Awaitable[list[R]]],
        config: BatchConfig | None = None,
    ):
        self.batch_func = batch_func
        self.config = config or BatchConfig()
        self._pending: deque[PendingItem[T, R]] = deque()
        self._lock = asyncio.Lock()
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self._flush_task: asyncio.Task | None = None
        self._running = True

    async def submit(self, item: T) -> R:
        """Submit a single item, returns when processed."""
        future: asyncio.Future[R] = asyncio.Future()
        pending = PendingItem(input=item, future=future)

        async with self._lock:
            self._pending.append(pending)

            # Start flush timer if this is the first item
            if len(self._pending) == 1:
                self._schedule_flush()

            # Flush immediately if batch is full
            if len(self._pending) >= self.config.max_batch_size:
                await self._flush_batch()

        return await future

    def _schedule_flush(self) -> None:
        """Schedule a flush after the wait timeout."""
        if self._flush_task is not None:
            return

        async def delayed_flush():
            await asyncio.sleep(self.config.max_wait_ms / 1000)
            async with self._lock:
                if self._pending:
                    await self._flush_batch()
                self._flush_task = None

        self._flush_task = asyncio.create_task(delayed_flush())

    async def _flush_batch(self) -> None:
        """Process a batch of pending items."""
        if not self._pending:
            return

        # Collect batch
        batch_items: list[PendingItem[T, R]] = []
        while self._pending and len(batch_items) < self.config.max_batch_size:
            batch_items.append(self._pending.popleft())

        # Cancel flush timer if pending is empty
        if not self._pending and self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None

        # Process batch (release lock during processing)
        asyncio.create_task(self._process_batch(batch_items))

    async def _process_batch(self, items: list[PendingItem[T, R]]) -> None:
        """Execute the batch function and distribute results."""
        async with self._batch_semaphore:
            inputs = [item.input for item in items]

            try:
                results = await self.batch_func(inputs)

                if len(results) != len(items):
                    raise ValueError(
                        f"Batch function returned {len(results)} results "
                        f"for {len(items)} inputs"
                    )

                for item, result in zip(items, results):
                    if not item.future.done():
                        item.future.set_result(result)

            except Exception as e:
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(e)

    async def flush(self) -> None:
        """Force flush any pending items."""
        async with self._lock:
            while self._pending:
                await self._flush_batch()

    async def close(self) -> None:
        """Shut down the batcher, processing remaining items."""
        self._running = False
        await self.flush()
        if self._flush_task:
            self._flush_task.cancel()


class GroupBatcher(Generic[T, R]):
    """
    Batches items by group key for group-aware processing.

    Useful for scoring where items with the same example_id need
    to be processed together.
    """

    def __init__(
        self,
        group_func: Callable[[str, list[T]], Awaitable[list[R]]],
        key_func: Callable[[T], str],
        config: BatchConfig | None = None,
    ):
        self.group_func = group_func
        self.key_func = key_func
        self.config = config or BatchConfig()
        self._groups: dict[str, deque[PendingItem[T, R]]] = {}
        self._lock = asyncio.Lock()
        self._flush_tasks: dict[str, asyncio.Task] = {}

    async def submit(self, item: T) -> R:
        """Submit an item to its group."""
        key = self.key_func(item)
        future: asyncio.Future[R] = asyncio.Future()
        pending = PendingItem(input=item, future=future)

        async with self._lock:
            if key not in self._groups:
                self._groups[key] = deque()

            self._groups[key].append(pending)

            # Schedule flush for this group
            if key not in self._flush_tasks:
                self._schedule_group_flush(key)

        return await future

    def _schedule_group_flush(self, key: str) -> None:
        """Schedule flush for a specific group."""

        async def delayed_flush():
            await asyncio.sleep(self.config.max_wait_ms / 1000)
            async with self._lock:
                if key in self._groups and self._groups[key]:
                    await self._flush_group(key)
                self._flush_tasks.pop(key, None)

        self._flush_tasks[key] = asyncio.create_task(delayed_flush())

    async def _flush_group(self, key: str) -> None:
        """Process all items in a group."""
        if key not in self._groups:
            return

        items = list(self._groups[key])
        self._groups[key].clear()

        if not items:
            return

        # Process outside lock
        asyncio.create_task(self._process_group(key, items))

    async def _process_group(
        self, key: str, items: list[PendingItem[T, R]]
    ) -> None:
        """Execute group function and distribute results."""
        inputs = [item.input for item in items]

        try:
            results = await self.group_func(key, inputs)

            for item, result in zip(items, results):
                if not item.future.done():
                    item.future.set_result(result)

        except Exception as e:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)
