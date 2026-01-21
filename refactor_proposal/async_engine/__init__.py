"""
Async engine - the heart of the verifiers runtime.

This module provides the core async primitives for high-performance
parallel execution with proper backpressure, batching, and resource management.

Example usage:

    from verifiers.async_engine import (
        AsyncExecutor,
        ExecutorConfig,
        parallel_map,
        AutoBatcher,
        AsyncStream,
    )

    # Simple parallel map
    results = await parallel_map(process_item, items, max_concurrent=64)

    # Streaming with backpressure
    executor = AsyncExecutor(process_item, ExecutorConfig(max_concurrent=128))
    async for result in executor.map(items):
        handle(result)

    # Automatic batching for APIs
    batcher = AutoBatcher(batch_api_call, BatchConfig(max_batch_size=32))
    result = await batcher.submit(single_item)  # Automatically batched
"""

from .executor import (
    AsyncExecutor,
    ExecutorConfig,
    ExecutorStats,
    parallel_map,
)

from .batch import (
    AutoBatcher,
    GroupBatcher,
    BatchConfig,
)

from .stream import (
    AsyncStream,
    map_stream,
    buffer_stream,
    batch_stream,
    merge_streams,
    with_progress,
    cancellable_stream,
    Progress,
)

__all__ = [
    # Executor
    "AsyncExecutor",
    "ExecutorConfig",
    "ExecutorStats",
    "parallel_map",
    # Batching
    "AutoBatcher",
    "GroupBatcher",
    "BatchConfig",
    # Streams
    "AsyncStream",
    "map_stream",
    "buffer_stream",
    "batch_stream",
    "merge_streams",
    "with_progress",
    "cancellable_stream",
    "Progress",
]
