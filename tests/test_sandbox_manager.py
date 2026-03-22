"""Tests for the SandboxManager class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from verifiers.envs.experimental.resource_managers.sandbox_manager import (
    ManagedSandbox,
    SandboxManager,
)
from verifiers.envs.experimental.resource_managers.base import ResourceState
from verifiers.envs.experimental.resource_managers.retry import RetryConfig
from verifiers.envs.experimental.resource_managers.errors import (
    SandboxCreationError,
    SandboxNotReadyError,
)


class TestManagedSandbox:
    """Tests for the ManagedSandbox dataclass."""

    def test_initial_state(self):
        """Test ManagedSandbox initialization."""
        sandbox = ManagedSandbox(id="sandbox-1")

        assert sandbox.id == "sandbox-1"
        assert sandbox.state == ResourceState.CREATING
        assert sandbox.command_times == []
        assert sandbox.ready_wait_time == 0.0

    def test_avg_command_time_empty(self):
        """Test average execution time with no commands."""
        sandbox = ManagedSandbox(id="sandbox-1")
        assert sandbox.avg_command_time == 0.0

    def test_avg_command_time(self):
        """Test average execution time calculation."""
        sandbox = ManagedSandbox(id="sandbox-1")
        sandbox.command_times = [1.0, 2.0, 3.0]

        assert sandbox.avg_command_time == 2.0


@pytest.fixture
def mock_sandbox_client():
    """Create a mock sandbox client."""
    with patch(
        "verifiers.envs.experimental.resource_managers.sandbox_manager.AsyncSandboxClient"
    ) as mock_class:
        mock_client = MagicMock()

        # Mock create method with incrementing IDs
        create_counter = [0]
        def create_side_effect(*args, **kwargs):
            create_counter[0] += 1
            mock_sandbox = MagicMock()
            mock_sandbox.id = f"sandbox-{create_counter[0]:03d}"
            return mock_sandbox
        mock_client.create = AsyncMock(side_effect=create_side_effect)

        # Mock delete method
        mock_client.delete = AsyncMock()

        # Mock bulk_delete method
        mock_client.bulk_delete = AsyncMock()

        # Mock wait_for_creation method
        mock_client.wait_for_creation = AsyncMock()

        # Mock execute_command method
        mock_result = MagicMock()
        mock_result.stdout = "ok"  # Return "ok" by default for health checks
        mock_result.stderr = ""
        mock_client.execute_command = AsyncMock(return_value=mock_result)

        # Mock close method (async)
        mock_client.close = AsyncMock()

        mock_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_create_sandbox_request():
    """Create a mock CreateSandboxRequest."""
    with patch(
        "verifiers.envs.experimental.resource_managers.sandbox_manager.CreateSandboxRequest"
    ) as mock_class:
        mock_request = MagicMock()
        mock_class.return_value = mock_request
        yield mock_request


class TestSandboxManager:
    """Tests for the SandboxManager class."""

    @pytest.fixture
    def manager(self, mock_sandbox_client, mock_create_sandbox_request):
        """Create a SandboxManager for testing."""
        manager = SandboxManager(
            default_request=mock_create_sandbox_request,
            timeout_per_command=30,
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
            enable_health_monitoring=False,
        )
        return manager

    @pytest.mark.asyncio
    async def test_acquire_creates_sandbox(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that acquire creates a sandbox via the API."""
        sandbox = await manager.acquire(rollout_id="rollout-1")

        assert sandbox.id == "sandbox-001"  # First sandbox gets ID 001
        assert sandbox.state == ResourceState.READY
        assert sandbox.rollout_id == "rollout-1"
        mock_sandbox_client.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_failure_raises_error(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that creation failures raise SandboxCreationError."""
        mock_sandbox_client.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with pytest.raises(SandboxCreationError):
            await manager.acquire(rollout_id="rollout-1")

    @pytest.mark.asyncio
    async def test_release_deletes_sandbox(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that release deletes the sandbox via the API."""
        sandbox = await manager.acquire(rollout_id="rollout-1")
        sandbox_id = sandbox.id

        assert sandbox.state == ResourceState.READY
        await manager.release(sandbox_id)

        # Verify sandbox was marked as destroyed
        assert sandbox.state == ResourceState.DESTROYED

    @pytest.mark.asyncio
    async def test_wait_for_ready(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test waiting for sandbox to be ready."""
        sandbox = await manager.acquire(rollout_id="rollout-1")
        await manager.wait_for_ready(sandbox.id)

        mock_sandbox_client.wait_for_creation.assert_called()
        assert sandbox.ready_wait_time >= 0

    @pytest.mark.asyncio
    async def test_wait_for_ready_failure(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that wait_for_ready failure raises SandboxNotReadyError."""
        mock_sandbox_client.wait_for_creation = AsyncMock(
            side_effect=RuntimeError("Sandbox failed to start")
        )

        sandbox = await manager.acquire(rollout_id="rollout-1")

        with pytest.raises(SandboxNotReadyError):
            await manager.wait_for_ready(sandbox.id)

    @pytest.mark.asyncio
    async def test_execute_command(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test executing a command in a sandbox."""
        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Set up a specific output for this command
        mock_result = MagicMock()
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        output = await manager.execute_command(sandbox.id, "ls -la")

        mock_sandbox_client.execute_command.assert_called()
        assert output == "command output"
        assert len(sandbox.command_times) == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test command output includes stderr."""
        mock_result = MagicMock()
        mock_result.stdout = "stdout output"
        mock_result.stderr = "stderr output"
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        sandbox = await manager.acquire(rollout_id="rollout-1")
        output = await manager.execute_command(sandbox.id, "ls -la")

        assert "stdout output" in output
        assert "stderr" in output
        assert "stderr output" in output

    @pytest.mark.asyncio
    async def test_execute_command_timeout(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test command timeout handling."""
        from prime_sandboxes import CommandTimeoutError

        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Set up the timeout error after acquire
        mock_sandbox_client.execute_command = AsyncMock(
            side_effect=CommandTimeoutError(
                sandbox_id=sandbox.id,
                command="sleep 100",
                timeout=30
            )
        )

        output = await manager.execute_command(sandbox.id, "sleep 100")

        assert "timed out" in output.lower()
        assert len(sandbox.command_times) == 1

    @pytest.mark.asyncio
    async def test_execute_command_unknown_sandbox(
        self, manager: SandboxManager
    ):
        """Test executing command on unknown sandbox."""
        with pytest.raises(ValueError, match="Unknown sandbox"):
            await manager.execute_command("unknown-id", "ls")

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test successful health check."""
        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        sandbox = await manager.acquire(rollout_id="rollout-1")
        is_healthy = await manager.health_check(sandbox.id)

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test health check reports unhealthy after consecutive failures."""
        mock_result = MagicMock()
        mock_result.stdout = "error"
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Default consecutive failure threshold is 3
        # First two failures return True (not yet unhealthy)
        assert await manager.health_check(sandbox.id) is True
        assert await manager.health_check(sandbox.id) is True
        # Third consecutive failure triggers unhealthy
        assert await manager.health_check(sandbox.id) is False

    @pytest.mark.asyncio
    async def test_release_all_uses_sync_client(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that release_all uses sync client for shutdown safety."""
        with patch(
            "verifiers.envs.experimental.resource_managers.sandbox_manager.SandboxClient"
        ) as mock_sync_class:
            mock_sync_client = MagicMock()
            mock_sync_class.return_value = mock_sync_client

            await manager.acquire(rollout_id="rollout-1")
            await manager.acquire(rollout_id="rollout-2")

            await manager.release_all()

            # Should use sync client for bulk delete
            mock_sync_client.bulk_delete.assert_called()

    @pytest.mark.asyncio
    async def test_teardown(self, manager: SandboxManager, mock_sandbox_client):
        """Test teardown closes the client."""
        await manager.teardown()
        mock_sandbox_client.close.assert_called_once()


class TestSandboxManagerErrors:
    """Tests for error tracking in SandboxManager."""

    @pytest.fixture
    def manager(self, mock_sandbox_client, mock_create_sandbox_request):
        """Create a SandboxManager for testing."""
        return SandboxManager(
            default_request=mock_create_sandbox_request,
            retry_config=RetryConfig(max_attempts=1, initial_delay=0.01),
            enable_health_monitoring=False,
        )

    @pytest.mark.asyncio
    async def test_creation_error_tracked(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that creation errors are tracked."""
        mock_sandbox_client.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with pytest.raises(SandboxCreationError):
            await manager.acquire(rollout_id="rollout-1")

        errors = manager.get_errors_for_rollout("rollout-1")
        assert len(errors) == 1
        assert errors[0].phase == "create"

    @pytest.mark.asyncio
    async def test_ready_error_tracked(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that ready errors are tracked."""
        mock_sandbox_client.wait_for_creation = AsyncMock(
            side_effect=RuntimeError("Failed to start")
        )

        sandbox = await manager.acquire(rollout_id="rollout-1")

        with pytest.raises(SandboxNotReadyError):
            await manager.wait_for_ready(sandbox.id)

        errors = manager.get_errors_for_rollout("rollout-1")
        assert len(errors) == 1
        assert errors[0].phase == "ready"

    @pytest.mark.asyncio
    async def test_execute_error_tracked(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that execution errors are tracked."""
        mock_sandbox_client.execute_command = AsyncMock(
            side_effect=RuntimeError("Execution failed")
        )

        sandbox = await manager.acquire(rollout_id="rollout-1")

        with pytest.raises(Exception):
            await manager.execute_command(sandbox.id, "ls")

        errors = manager.get_errors_for_rollout("rollout-1")
        assert len(errors) == 1
        assert errors[0].phase == "execute"
