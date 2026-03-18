"""Tests for the ResourceManager base class."""

import asyncio
import pytest

from verifiers.envs.experimental.managers.resource_manager import (
    ManagedResource,
    ResourceError,
    ResourceManager,
    ResourceState,
)


class MockResourceManager(ResourceManager):
    """Mock implementation of ResourceManager for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.created_resources: list[str] = []
        self.destroyed_resources: list[str] = []
        self.health_check_results: dict[str, bool] = {}
        self.should_fail_create = False
        self.should_fail_destroy = False
        self.should_fail_health_check = False

    async def _create_resource(self, resource: ManagedResource, **kwargs) -> None:
        if self.should_fail_create:
            raise RuntimeError("Mock creation failure")
        self.created_resources.append(resource.id)

    async def _destroy_resource(self, resource_id: str) -> None:
        if self.should_fail_destroy:
            raise RuntimeError("Mock destruction failure")
        self.destroyed_resources.append(resource_id)

    async def _check_health(self, resource_id: str) -> bool:
        if self.should_fail_health_check:
            raise RuntimeError("Mock health check failure")
        return self.health_check_results.get(resource_id, True)


class TestManagedResource:
    """Tests for the ManagedResource dataclass."""

    def test_initial_state(self):
        """Test that new resources start in CREATING state."""
        resource = ManagedResource(id="test-1")
        assert resource.state == ResourceState.CREATING
        assert resource.error is None
        assert resource.ready_at is None
        assert resource.destroyed_at is None

    def test_mark_ready(self):
        """Test marking a resource as ready."""
        resource = ManagedResource(id="test-1")
        resource.mark_ready()

        assert resource.state == ResourceState.READY
        assert resource.ready_at is not None

    def test_mark_error(self):
        """Test marking a resource with an error."""
        resource = ManagedResource(id="test-1")
        error = RuntimeError("test error")
        resource.mark_error(error)

        assert resource.state == ResourceState.ERROR
        assert resource.error is error

    def test_mark_destroying(self):
        """Test marking a resource as destroying."""
        resource = ManagedResource(id="test-1")
        resource.mark_destroying()
        assert resource.state == ResourceState.DESTROYING

    def test_mark_destroyed(self):
        """Test marking a resource as destroyed."""
        resource = ManagedResource(id="test-1")
        resource.mark_destroyed()

        assert resource.state == ResourceState.DESTROYED
        assert resource.destroyed_at is not None

    def test_is_active_creating(self):
        """Test is_active returns True for CREATING state."""
        resource = ManagedResource(id="test-1")
        assert resource.is_active is True

    def test_is_active_ready(self):
        """Test is_active returns True for READY state."""
        resource = ManagedResource(id="test-1")
        resource.mark_ready()
        assert resource.is_active is True

    def test_is_active_error(self):
        """Test is_active returns False for ERROR state."""
        resource = ManagedResource(id="test-1")
        resource.mark_error(RuntimeError("test"))
        assert resource.is_active is False

    def test_is_active_destroyed(self):
        """Test is_active returns False for DESTROYED state."""
        resource = ManagedResource(id="test-1")
        resource.mark_destroyed()
        assert resource.is_active is False

    def test_ready_wait_time_before_ready(self):
        """Test ready_wait_time returns time since creation when not ready."""
        resource = ManagedResource(id="test-1")
        wait_time = resource.ready_wait_time
        assert wait_time >= 0

    def test_ready_wait_time_after_ready(self):
        """Test ready_wait_time returns time between creation and ready."""
        resource = ManagedResource(id="test-1")
        resource.mark_ready()
        wait_time = resource.ready_wait_time
        assert wait_time >= 0


class TestResourceManager:
    """Tests for the ResourceManager base class."""

    @pytest.fixture
    def manager(self):
        """Create a mock resource manager for testing."""
        return MockResourceManager(
            health_check_interval=0.1,
            enable_health_monitoring=False,
            max_retries=2,
            retry_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_acquire_success(self, manager: MockResourceManager):
        """Test successful resource acquisition."""
        resource = await manager.acquire(rollout_id="rollout-1")

        assert resource.state == ResourceState.READY
        assert resource.rollout_id == "rollout-1"
        assert len(manager.created_resources) == 1

    @pytest.mark.asyncio
    async def test_acquire_failure(self, manager: MockResourceManager):
        """Test resource acquisition failure."""
        manager.should_fail_create = True

        with pytest.raises(RuntimeError, match="Mock creation failure"):
            await manager.acquire(rollout_id="rollout-1")

        # Resource should be tracked with ERROR state
        resources = list(manager._resources.values())
        assert len(resources) == 1
        assert resources[0].state == ResourceState.ERROR

    @pytest.mark.asyncio
    async def test_release_success(self, manager: MockResourceManager):
        """Test successful resource release."""
        resource = await manager.acquire(rollout_id="rollout-1")
        await manager.release(resource.id)

        assert resource.state == ResourceState.DESTROYED
        assert resource.id in manager.destroyed_resources

    @pytest.mark.asyncio
    async def test_release_unknown_resource(self, manager: MockResourceManager):
        """Test releasing an unknown resource (should not raise)."""
        await manager.release("unknown-id")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_release_already_destroyed(self, manager: MockResourceManager):
        """Test releasing an already destroyed resource."""
        resource = await manager.acquire(rollout_id="rollout-1")
        await manager.release(resource.id)
        await manager.release(resource.id)  # Second release should be no-op

        # Should only have one destroy call
        assert manager.destroyed_resources.count(resource.id) == 1

    @pytest.mark.asyncio
    async def test_release_failure(self, manager: MockResourceManager):
        """Test resource release failure (should mark destroyed anyway)."""
        resource = await manager.acquire(rollout_id="rollout-1")
        manager.should_fail_destroy = True

        await manager.release(resource.id)

        # Should still be marked as destroyed
        assert resource.state == ResourceState.DESTROYED

    @pytest.mark.asyncio
    async def test_release_all(self, manager: MockResourceManager):
        """Test releasing all resources."""
        r1 = await manager.acquire(rollout_id="rollout-1")
        r2 = await manager.acquire(rollout_id="rollout-2")
        r3 = await manager.acquire(rollout_id="rollout-3")

        await manager.release_all()

        assert r1.state == ResourceState.DESTROYED
        assert r2.state == ResourceState.DESTROYED
        assert r3.state == ResourceState.DESTROYED

    @pytest.mark.asyncio
    async def test_get_active_resources(self, manager: MockResourceManager):
        """Test getting active resources."""
        r1 = await manager.acquire(rollout_id="rollout-1")
        r2 = await manager.acquire(rollout_id="rollout-2")

        active = manager.get_active_resources()
        assert len(active) == 2

        await manager.release(r1.id)

        active = manager.get_active_resources()
        assert len(active) == 1
        assert active[0].id == r2.id

    @pytest.mark.asyncio
    async def test_get_resource(self, manager: MockResourceManager):
        """Test getting a specific resource."""
        resource = await manager.acquire(rollout_id="rollout-1")

        found = manager.get_resource(resource.id)
        assert found is resource

        not_found = manager.get_resource("unknown-id")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_health_check_success(self, manager: MockResourceManager):
        """Test successful health check."""
        resource = await manager.acquire(rollout_id="rollout-1")
        manager.health_check_results[resource.id] = True

        is_healthy = await manager.health_check(resource.id)
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, manager: MockResourceManager):
        """Test health check returning False."""
        resource = await manager.acquire(rollout_id="rollout-1")
        manager.health_check_results[resource.id] = False

        is_healthy = await manager.health_check(resource.id)
        assert is_healthy is False
        assert resource.state == ResourceState.ERROR

    @pytest.mark.asyncio
    async def test_health_check_exception(self, manager: MockResourceManager):
        """Test health check raising an exception."""
        resource = await manager.acquire(rollout_id="rollout-1")
        manager.should_fail_health_check = True

        is_healthy = await manager.health_check(resource.id)
        assert is_healthy is False
        assert resource.state == ResourceState.ERROR

    @pytest.mark.asyncio
    async def test_error_tracking(self, manager: MockResourceManager):
        """Test error tracking for rollouts."""
        manager.should_fail_create = True

        with pytest.raises(RuntimeError):
            await manager.acquire(rollout_id="rollout-1")

        errors = manager.get_errors_for_rollout("rollout-1")
        assert len(errors) == 1
        assert errors[0].phase == "create"
        assert errors[0].rollout_id == "rollout-1"

    @pytest.mark.asyncio
    async def test_get_all_errors(self, manager: MockResourceManager):
        """Test getting all errors."""
        manager.should_fail_create = True

        with pytest.raises(RuntimeError):
            await manager.acquire(rollout_id="rollout-1")
        with pytest.raises(RuntimeError):
            await manager.acquire(rollout_id="rollout-2")

        errors = manager.get_all_errors()
        assert len(errors) == 2

    @pytest.mark.asyncio
    async def test_clear_errors(self, manager: MockResourceManager):
        """Test clearing errors."""
        manager.should_fail_create = True

        with pytest.raises(RuntimeError):
            await manager.acquire(rollout_id="rollout-1")

        assert len(manager.get_all_errors()) == 1

        manager.clear_errors()
        assert len(manager.get_all_errors()) == 0


class TestHealthMonitoring:
    """Tests for periodic health monitoring."""

    @pytest.fixture
    def manager(self):
        """Create a mock resource manager with health monitoring enabled."""
        return MockResourceManager(
            health_check_interval=0.05,
            enable_health_monitoring=True,
            max_retries=1,
            retry_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_health_monitoring_starts_on_acquire(self, manager: MockResourceManager):
        """Test that health monitoring starts when first resource is acquired."""
        assert manager._health_monitor_task is None

        await manager.acquire(rollout_id="rollout-1")

        assert manager._health_monitor_task is not None

        await manager.stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_health_monitoring_detects_unhealthy(self, manager: MockResourceManager):
        """Test that health monitoring detects unhealthy resources."""
        resource = await manager.acquire(rollout_id="rollout-1")
        manager.health_check_results[resource.id] = False

        # Wait for health check to run
        await asyncio.sleep(0.1)

        assert resource.state == ResourceState.ERROR

        await manager.stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_health_monitoring_stops_on_release_all(self, manager: MockResourceManager):
        """Test that health monitoring stops when all resources are released."""
        await manager.acquire(rollout_id="rollout-1")
        assert manager._health_monitor_task is not None

        await manager.release_all()

        assert manager._health_monitor_task is None


class TestResourceError:
    """Tests for the ResourceError dataclass."""

    def test_resource_error_repr(self):
        """Test ResourceError string representation."""
        error = ResourceError(
            resource_id="test-1",
            rollout_id="rollout-1",
            error=RuntimeError("test error"),
            timestamp=1234567890.0,
            phase="create",
        )

        repr_str = repr(error)
        assert "test-1" in repr_str
        assert "rollout-1" in repr_str
        assert "create" in repr_str
