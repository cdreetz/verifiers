"""Base resource management abstractions for lifecycle tracking and cleanup."""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ResourceState(Enum):
    CREATING = "creating"
    READY = "ready"
    ERROR = "error"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


@dataclass
class ResourceError:
    resource_id: str
    rollout_id: str | None
    error: Exception
    timestamp: float
    phase: str  # "create", "ready", "execute", "health_check", "destroy"


@dataclass
class ManagedResource:
    id: str
    state: ResourceState = ResourceState.CREATING
    created_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    destroyed_at: float | None = None
    error: Exception | None = None
    rollout_id: str | None = None

    def mark_ready(self) -> None:
        self.state = ResourceState.READY
        self.ready_at = time.time()

    def mark_error(self, error: Exception) -> None:
        self.state = ResourceState.ERROR
        self.error = error

    def mark_destroying(self) -> None:
        self.state = ResourceState.DESTROYING

    def mark_destroyed(self) -> None:
        self.state = ResourceState.DESTROYED
        self.destroyed_at = time.time()

    @property
    def is_active(self) -> bool:
        return self.state in (ResourceState.CREATING, ResourceState.READY)


class ResourceManager(ABC):
    """Base class for resource lifecycle management."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        health_check_interval: float = 30.0,
        enable_health_monitoring: bool = False,
    ):
        self.resources: dict[str, ManagedResource] = {}
        self.errors: list[ResourceError] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        self.enable_health_monitoring = enable_health_monitoring
        self.health_monitor_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def acquire(self, rollout_id: str | None = None, **kwargs: Any) -> ManagedResource:
        """Create and track a new resource."""
        resource = self.create_resource_object(rollout_id)
        original_id = resource.id

        async with self.lock:
            self.resources[resource.id] = resource

        try:
            await self.create_with_retry(resource, **kwargs)

            if resource.id != original_id:
                async with self.lock:
                    del self.resources[original_id]
                    self.resources[resource.id] = resource

            resource.mark_ready()

            if self.enable_health_monitoring and self.health_monitor_task is None:
                self.health_monitor_task = asyncio.create_task(self.health_monitor_loop())

            return resource

        except Exception as e:
            resource.mark_error(e)
            self.record_error(resource.id, rollout_id, e, "create")
            raise

    async def release(self, resource_id: str) -> None:
        """Release a resource."""
        async with self.lock:
            resource = self.resources.get(resource_id)
            if resource is None:
                return

        if resource.state == ResourceState.DESTROYED:
            return

        resource.mark_destroying()

        try:
            await self.destroy_with_retry(resource_id)
            resource.mark_destroyed()
        except Exception as e:
            self.record_error(resource_id, resource.rollout_id, e, "destroy")
            resource.mark_destroyed()

    async def release_all(self) -> None:
        """Release all active resources."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            self.health_monitor_task = None

        async with self.lock:
            active = [r for r in self.resources.values() if r.is_active]

        if active:
            self.logger.info(f"Releasing {len(active)} resources")
            await asyncio.gather(*[self.release(r.id) for r in active], return_exceptions=True)

    def get_resource(self, resource_id: str) -> ManagedResource | None:
        return self.resources.get(resource_id)

    def get_active_resources(self) -> list[ManagedResource]:
        return [r for r in self.resources.values() if r.is_active]

    def record_error(self, resource_id: str, rollout_id: str | None, error: Exception, phase: str) -> None:
        self.errors.append(ResourceError(resource_id, rollout_id, error, time.time(), phase))

    def get_errors_for_rollout(self, rollout_id: str) -> list[ResourceError]:
        return [e for e in self.errors if e.rollout_id == rollout_id]

    def get_all_errors(self) -> list[ResourceError]:
        return list(self.errors)

    async def health_monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                for resource in list(self.resources.values()):
                    if resource.state == ResourceState.READY:
                        try:
                            if not await self.check_health(resource.id):
                                error = RuntimeError(f"Health check failed for {resource.id}")
                                resource.mark_error(error)
                                self.record_error(resource.id, resource.rollout_id, error, "health_check")
                        except Exception as e:
                            resource.mark_error(e)
                            self.record_error(resource.id, resource.rollout_id, e, "health_check")
            except asyncio.CancelledError:
                break

    async def create_with_retry(self, resource: ManagedResource, **kwargs: Any) -> None:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self.create_resource(resource, **kwargs)
                return
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        raise last_error or RuntimeError("Resource creation failed")

    async def destroy_with_retry(self, resource_id: str) -> None:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self.destroy_resource(resource_id)
                return
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        raise last_error or RuntimeError("Resource destruction failed")

    def create_resource_object(self, rollout_id: str | None) -> ManagedResource:
        return ManagedResource(id=f"resource-{uuid.uuid4().hex[:8]}", rollout_id=rollout_id)

    @abstractmethod
    async def create_resource(self, resource: ManagedResource, **kwargs: Any) -> None:
        """Create the actual resource. Update resource.id if needed."""
        pass

    @abstractmethod
    async def destroy_resource(self, resource_id: str) -> None:
        """Destroy the resource."""
        pass

    @abstractmethod
    async def check_health(self, resource_id: str) -> bool:
        """Check if resource is healthy."""
        pass
