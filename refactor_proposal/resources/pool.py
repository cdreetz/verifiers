"""
Async resource pool for managing sandboxes, connections, and other resources.

Key features:
- Bounded pool size with backpressure
- Health checking and automatic eviction
- Graceful shutdown with cleanup
- Per-resource timeout handling
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Awaitable, AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

R = TypeVar("R")  # Resource type


class ResourceState(Enum):
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    IN_USE = "in_use"
    UNHEALTHY = "unhealthy"
    CLOSING = "closing"


@dataclass
class PooledResource(Generic[R]):
    """Wrapper around a resource with pool metadata."""

    resource: R
    state: ResourceState = ResourceState.INITIALIZING
    created_at: float = field(default_factory=time.perf_counter)
    last_used: float = field(default_factory=time.perf_counter)
    use_count: int = 0
    consecutive_failures: int = 0


@dataclass
class PoolConfig:
    """Configuration for resource pool."""

    min_size: int = 0  # Minimum pool size (pre-warmed)
    max_size: int = 10  # Maximum pool size
    acquire_timeout_ms: float = 30000  # Timeout waiting for resource
    max_idle_ms: float = 300000  # Evict after idle (5 min)
    max_lifetime_ms: float = 3600000  # Max resource lifetime (1 hour)
    health_check_interval_ms: float = 30000  # Health check frequency
    max_consecutive_failures: int = 3  # Failures before eviction


@dataclass
class PoolStats:
    """Pool statistics for monitoring."""

    current_size: int = 0
    available: int = 0
    in_use: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    total_evictions: int = 0
    failed_acquisitions: int = 0
    health_check_failures: int = 0


class ResourcePool(Generic[R]):
    """
    Async resource pool with lifecycle management.

    Usage:
        async def create_sandbox() -> Sandbox:
            return await Sandbox.create()

        async def destroy_sandbox(s: Sandbox) -> None:
            await s.terminate()

        pool = ResourcePool(
            create_func=create_sandbox,
            destroy_func=destroy_sandbox,
            health_func=lambda s: s.is_alive(),
            config=PoolConfig(max_size=10)
        )

        async with pool:
            async with pool.acquire() as sandbox:
                result = await sandbox.execute(code)
    """

    def __init__(
        self,
        create_func: Callable[[], Awaitable[R]],
        destroy_func: Callable[[R], Awaitable[None]] | None = None,
        health_func: Callable[[R], Awaitable[bool]] | None = None,
        config: PoolConfig | None = None,
    ):
        self.create_func = create_func
        self.destroy_func = destroy_func or (lambda r: asyncio.sleep(0))
        self.health_func = health_func or (lambda r: asyncio.coroutine(lambda: True)())
        self.config = config or PoolConfig()

        self._resources: list[PooledResource[R]] = []
        self._available: asyncio.Queue[PooledResource[R]] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._stats = PoolStats()
        self._running = False
        self._health_task: asyncio.Task | None = None
        self._closed = asyncio.Event()

    @property
    def stats(self) -> PoolStats:
        return self._stats

    async def __aenter__(self) -> "ResourcePool[R]":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def start(self) -> None:
        """Start the pool and pre-warm resources."""
        self._running = True
        self._closed.clear()

        # Pre-warm minimum resources
        warm_tasks = [self._create_resource() for _ in range(self.config.min_size)]
        await asyncio.gather(*warm_tasks, return_exceptions=True)

        # Start health check loop
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def close(self) -> None:
        """Shut down the pool, cleaning up all resources."""
        self._running = False

        # Stop health checks
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Destroy all resources
        async with self._lock:
            destroy_tasks = []
            for pr in self._resources:
                pr.state = ResourceState.CLOSING
                destroy_tasks.append(self._destroy_resource(pr))
            await asyncio.gather(*destroy_tasks, return_exceptions=True)
            self._resources.clear()

        self._closed.set()

    async def _create_resource(self) -> PooledResource[R] | None:
        """Create and add a new resource to the pool."""
        async with self._lock:
            if len(self._resources) >= self.config.max_size:
                return None

            pr: PooledResource[R] = PooledResource(
                resource=None,  # type: ignore
                state=ResourceState.INITIALIZING,
            )
            self._resources.append(pr)
            self._stats.current_size += 1

        try:
            resource = await self.create_func()
            pr.resource = resource
            pr.state = ResourceState.AVAILABLE
            self._stats.available += 1
            await self._available.put(pr)
            return pr

        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            async with self._lock:
                self._resources.remove(pr)
                self._stats.current_size -= 1
            return None

    async def _destroy_resource(self, pr: PooledResource[R]) -> None:
        """Destroy a resource."""
        try:
            if pr.resource is not None:
                await self.destroy_func(pr.resource)
        except Exception as e:
            logger.error(f"Error destroying resource: {e}")
        finally:
            self._stats.total_evictions += 1

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[R]:
        """
        Acquire a resource from the pool.

        Usage:
            async with pool.acquire() as resource:
                await resource.do_something()
        """
        pr = await self._acquire()
        try:
            yield pr.resource
        except Exception as e:
            # Track failure for health checking
            pr.consecutive_failures += 1
            raise
        finally:
            await self._release(pr)

    async def _acquire(self) -> PooledResource[R]:
        """Internal acquire logic."""
        timeout = self.config.acquire_timeout_ms / 1000
        deadline = time.perf_counter() + timeout

        while True:
            # Try to get from available queue
            try:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self._stats.failed_acquisitions += 1
                    raise TimeoutError("Timed out waiting for resource")

                pr = await asyncio.wait_for(
                    self._available.get(),
                    timeout=min(remaining, 0.1),  # Check periodically
                )

                # Validate resource is still good
                if await self._validate_resource(pr):
                    pr.state = ResourceState.IN_USE
                    pr.last_used = time.perf_counter()
                    pr.use_count += 1
                    self._stats.available -= 1
                    self._stats.in_use += 1
                    self._stats.total_acquisitions += 1
                    return pr
                else:
                    # Resource is bad, destroy and try again
                    await self._evict(pr)

            except asyncio.TimeoutError:
                # Try to create a new resource
                async with self._lock:
                    if len(self._resources) < self.config.max_size:
                        pr = await self._create_resource()
                        if pr:
                            # Get it from the queue (we just put it there)
                            pr = await self._available.get()
                            pr.state = ResourceState.IN_USE
                            pr.last_used = time.perf_counter()
                            pr.use_count += 1
                            self._stats.available -= 1
                            self._stats.in_use += 1
                            self._stats.total_acquisitions += 1
                            return pr

    async def _release(self, pr: PooledResource[R]) -> None:
        """Return a resource to the pool."""
        self._stats.in_use -= 1
        self._stats.total_releases += 1

        # Check if resource should be evicted
        now = time.perf_counter()
        age_ms = (now - pr.created_at) * 1000
        failures = pr.consecutive_failures

        if (
            age_ms > self.config.max_lifetime_ms
            or failures >= self.config.max_consecutive_failures
        ):
            await self._evict(pr)
            return

        # Return to available pool
        pr.state = ResourceState.AVAILABLE
        pr.last_used = now
        pr.consecutive_failures = 0  # Reset on successful use
        self._stats.available += 1
        await self._available.put(pr)

    async def _validate_resource(self, pr: PooledResource[R]) -> bool:
        """Check if a resource is valid for use."""
        # Check idle timeout
        now = time.perf_counter()
        idle_ms = (now - pr.last_used) * 1000
        if idle_ms > self.config.max_idle_ms:
            return False

        # Check lifetime
        age_ms = (now - pr.created_at) * 1000
        if age_ms > self.config.max_lifetime_ms:
            return False

        return True

    async def _evict(self, pr: PooledResource[R]) -> None:
        """Remove and destroy a resource."""
        async with self._lock:
            if pr in self._resources:
                self._resources.remove(pr)
                self._stats.current_size -= 1

        pr.state = ResourceState.CLOSING
        await self._destroy_resource(pr)

    async def _health_check_loop(self) -> None:
        """Periodic health check of resources."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_ms / 1000)

                # Get snapshot of resources to check
                async with self._lock:
                    to_check = [
                        pr
                        for pr in self._resources
                        if pr.state == ResourceState.AVAILABLE
                    ]

                for pr in to_check:
                    if not self._running:
                        break

                    try:
                        healthy = await asyncio.wait_for(
                            self.health_func(pr.resource),
                            timeout=5.0,
                        )
                        if not healthy:
                            self._stats.health_check_failures += 1
                            pr.state = ResourceState.UNHEALTHY
                            await self._evict(pr)
                    except Exception as e:
                        logger.warning(f"Health check failed: {e}")
                        self._stats.health_check_failures += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
