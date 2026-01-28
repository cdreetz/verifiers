"""
Resource Manager Protocol and Base Classes

Provides a standard interface for managing resources (sandboxes, containers, connections, etc.)
with support for different allocation modes and lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


class AllocationMode(Enum):
    """Resource allocation modes."""

    ONE_TO_ONE = "one_to_one"  # Each rollout gets dedicated resource
    POOL = "pool"  # Resources pooled and reused across rollouts
    SHARED = "shared"  # Single resource shared across all rollouts


@dataclass
class ResourceMetrics:
    """Metrics for a single resource instance."""

    resource_id: str
    created_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    released_at: float | None = None
    acquisition_count: int = 0
    total_usage_time: float = 0.0
    last_used_at: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ready_time(self) -> float:
        """Time from creation to ready."""
        if self.ready_at is None:
            return 0.0
        return self.ready_at - self.created_at

    @property
    def is_ready(self) -> bool:
        return self.ready_at is not None


@dataclass
class ManagerMetrics:
    """Aggregate metrics for a resource manager."""

    total_created: int = 0
    total_destroyed: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    total_errors: int = 0
    creation_times: list[float] = field(default_factory=list)
    acquisition_times: list[float] = field(default_factory=list)
    usage_times: list[float] = field(default_factory=list)

    @property
    def avg_creation_time(self) -> float:
        return sum(self.creation_times) / len(self.creation_times) if self.creation_times else 0.0

    @property
    def avg_acquisition_time(self) -> float:
        return sum(self.acquisition_times) / len(self.acquisition_times) if self.acquisition_times else 0.0

    @property
    def avg_usage_time(self) -> float:
        return sum(self.usage_times) / len(self.usage_times) if self.usage_times else 0.0

    @property
    def active_count(self) -> int:
        return self.total_created - self.total_destroyed


# Type variable for resource type
R = TypeVar("R")


@dataclass
class ResourceHandle(Generic[R]):
    """Handle to an acquired resource with metadata."""

    resource: R
    resource_id: str
    metrics: ResourceMetrics
    acquired_at: float = field(default_factory=time.time)
    _release_callback: Callable[[], Any] | None = field(default=None, repr=False)

    async def release(self) -> None:
        """Release this resource back to the manager."""
        if self._release_callback:
            await self._release_callback()


@runtime_checkable
class ResourceManager(Protocol[R]):
    """
    Protocol for resource managers.

    Implementations must provide methods for:
    - Acquiring resources for rollouts
    - Releasing resources after use
    - Lifecycle management (startup, teardown)
    - Metrics collection
    """

    @property
    def mode(self) -> AllocationMode:
        """The allocation mode for this manager."""
        ...

    @property
    def metrics(self) -> ManagerMetrics:
        """Aggregate metrics for the manager."""
        ...

    async def startup(self) -> None:
        """Initialize the manager and any pre-allocated resources."""
        ...

    async def acquire(self, state: dict[str, Any]) -> ResourceHandle[R]:
        """
        Acquire a resource for a rollout.

        Args:
            state: The rollout state dict (can be used for resource customization)

        Returns:
            ResourceHandle containing the resource and metadata
        """
        ...

    async def release(self, handle: ResourceHandle[R]) -> None:
        """
        Release a resource after use.

        Args:
            handle: The resource handle to release
        """
        ...

    async def teardown(self) -> None:
        """Cleanup all resources and shutdown the manager."""
        ...


class BaseResourceManager(ABC, Generic[R]):
    """
    Abstract base class for resource managers.

    Provides common functionality for:
    - Metrics tracking
    - Lifecycle management
    - Logging
    """

    def __init__(
        self,
        mode: AllocationMode = AllocationMode.ONE_TO_ONE,
        max_resources: int | None = None,
    ):
        self._mode = mode
        self._max_resources = max_resources
        self._metrics = ManagerMetrics()
        self._resource_metrics: dict[str, ResourceMetrics] = {}
        self._lock = asyncio.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._is_started = False
        self._is_shutdown = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def mode(self) -> AllocationMode:
        return self._mode

    @property
    def metrics(self) -> ManagerMetrics:
        return self._metrics

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def is_shutdown(self) -> bool:
        return self._is_shutdown

    async def startup(self) -> None:
        """Initialize the manager."""
        if self._is_started:
            return

        async with self._lock:
            if self._is_started:  # Double-check after acquiring lock
                return

            if self._max_resources:
                self._semaphore = asyncio.Semaphore(self._max_resources)

            await self._do_startup()
            self._is_started = True
            self.logger.debug(f"Started {self.__class__.__name__} in {self._mode.value} mode")

    @abstractmethod
    async def _do_startup(self) -> None:
        """Implementation-specific startup logic."""
        ...

    async def acquire(self, state: dict[str, Any]) -> ResourceHandle[R]:
        """Acquire a resource with metrics tracking."""
        if not self._is_started:
            await self.startup()

        start_time = time.time()

        if self._semaphore:
            await self._semaphore.acquire()

        try:
            handle = await self._do_acquire(state)
            acquisition_time = time.time() - start_time

            # Update metrics
            self._metrics.total_acquisitions += 1
            self._metrics.acquisition_times.append(acquisition_time)
            handle.metrics.acquisition_count += 1
            handle.metrics.last_used_at = time.time()

            # Set up release callback
            original_release = handle._release_callback

            async def release_with_tracking():
                if original_release:
                    await original_release()
                await self.release(handle)

            handle._release_callback = release_with_tracking

            self.logger.debug(
                f"Acquired resource {handle.resource_id} in {acquisition_time:.2f}s"
            )
            return handle

        except Exception as e:
            self._metrics.total_errors += 1
            if self._semaphore:
                self._semaphore.release()
            raise

    @abstractmethod
    async def _do_acquire(self, state: dict[str, Any]) -> ResourceHandle[R]:
        """Implementation-specific acquisition logic."""
        ...

    async def release(self, handle: ResourceHandle[R]) -> None:
        """Release a resource with metrics tracking."""
        usage_time = time.time() - handle.acquired_at
        handle.metrics.total_usage_time += usage_time
        self._metrics.total_releases += 1
        self._metrics.usage_times.append(usage_time)

        try:
            await self._do_release(handle)
            self.logger.debug(
                f"Released resource {handle.resource_id} after {usage_time:.2f}s"
            )
        finally:
            if self._semaphore:
                self._semaphore.release()

    @abstractmethod
    async def _do_release(self, handle: ResourceHandle[R]) -> None:
        """Implementation-specific release logic."""
        ...

    async def teardown(self) -> None:
        """Cleanup all resources."""
        if self._is_shutdown:
            return

        async with self._lock:
            if self._is_shutdown:
                return

            self.logger.info(
                f"Tearing down {self.__class__.__name__} "
                f"({self._metrics.active_count} active resources)"
            )
            await self._do_teardown()
            self._is_shutdown = True

    @abstractmethod
    async def _do_teardown(self) -> None:
        """Implementation-specific teardown logic."""
        ...

    @asynccontextmanager
    async def acquire_context(self, state: dict[str, Any]) -> AsyncIterator[ResourceHandle[R]]:
        """
        Context manager for acquiring and automatically releasing a resource.

        Usage:
            async with manager.acquire_context(state) as handle:
                # Use handle.resource
                pass
            # Resource automatically released
        """
        handle = await self.acquire(state)
        try:
            yield handle
        finally:
            await self.release(handle)

    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics | None:
        """Get metrics for a specific resource."""
        return self._resource_metrics.get(resource_id)

    def _create_metrics(self, resource_id: str) -> ResourceMetrics:
        """Create and track metrics for a new resource."""
        metrics = ResourceMetrics(resource_id=resource_id)
        self._resource_metrics[resource_id] = metrics
        self._metrics.total_created += 1
        return metrics

    def _mark_ready(self, resource_id: str) -> None:
        """Mark a resource as ready."""
        metrics = self._resource_metrics.get(resource_id)
        if metrics:
            metrics.ready_at = time.time()
            self._metrics.creation_times.append(metrics.ready_time)

    def _mark_destroyed(self, resource_id: str) -> None:
        """Mark a resource as destroyed."""
        metrics = self._resource_metrics.get(resource_id)
        if metrics:
            metrics.released_at = time.time()
            self._metrics.total_destroyed += 1
