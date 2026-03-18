"""Tests for the NewSandboxEnv class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datasets import Dataset

from verifiers.envs.experimental.new_sandbox_env import NewSandboxEnv
from verifiers.envs.experimental.managers.resource_manager import ResourceState


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return Dataset.from_dict({"question": ["mock question"], "info": [{}]})


@pytest.fixture
def mock_sandbox_manager():
    """Create a mock SandboxManager."""
    with patch(
        "verifiers.envs.experimental.new_sandbox_env.SandboxManager"
    ) as mock_class:
        mock_manager = MagicMock()

        # Mock acquire
        mock_sandbox = MagicMock()
        mock_sandbox.id = "sandbox-123"
        mock_sandbox.state = ResourceState.READY
        mock_sandbox.rollout_id = None
        mock_sandbox.error = None
        mock_sandbox.command_execution_times = []
        mock_sandbox.sandbox_ready_wait_time = 0.5
        mock_manager.acquire = AsyncMock(return_value=mock_sandbox)

        # Mock release
        mock_manager.release = AsyncMock()

        # Mock release_all
        mock_manager.release_all = AsyncMock()

        # Mock wait_for_ready
        mock_manager.wait_for_ready = AsyncMock()

        # Mock execute_command
        mock_manager.execute_command = AsyncMock(return_value="command output")

        # Mock get_resource
        mock_manager.get_resource = MagicMock(return_value=mock_sandbox)

        # Mock get_active_resources
        mock_manager.get_active_resources = MagicMock(return_value=[mock_sandbox])

        # Mock get_errors_for_rollout
        mock_manager.get_errors_for_rollout = MagicMock(return_value=[])

        # Mock bulk_delete
        mock_manager.bulk_delete = AsyncMock()

        # Mock teardown
        mock_manager.teardown = MagicMock()

        # Mock _sandbox_client for compatibility property
        mock_manager._sandbox_client = MagicMock()

        # Mock _with_retry for compatibility property
        mock_manager._with_retry = MagicMock()

        mock_class.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def mock_create_sandbox_request():
    """Create a mock CreateSandboxRequest."""
    with patch(
        "verifiers.envs.experimental.new_sandbox_env.CreateSandboxRequest"
    ) as mock_class:
        mock_request = MagicMock()
        mock_request.model_copy = MagicMock(return_value=mock_request)
        mock_class.return_value = mock_request
        yield mock_request


class TestNewSandboxEnv:
    """Tests for the NewSandboxEnv class."""

    @pytest.fixture
    def env(self, mock_dataset, mock_sandbox_manager, mock_create_sandbox_request):
        """Create a NewSandboxEnv for testing."""
        env = NewSandboxEnv(
            dataset=mock_dataset,
            max_retries=1,
            base_delay=0.1,
            enable_health_monitoring=False,
        )
        env.logger = MagicMock()
        return env

    @pytest.mark.asyncio
    async def test_setup_state_acquires_sandbox(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that setup_state acquires a sandbox through the manager."""
        state = {"trajectory_id": "traj-1"}
        state = await env.setup_state(state)

        mock_sandbox_manager.acquire.assert_called_once()
        assert state["sandbox_id"] == "sandbox-123"
        assert state["managed_sandbox"] is not None
        assert "sandbox_state" in state

    @pytest.mark.asyncio
    async def test_destroy_sandbox_releases(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that destroy_sandbox releases through the manager."""
        state = {"trajectory_id": "traj-1"}
        state = await env.setup_state(state)

        await env.destroy_sandbox(state)

        mock_sandbox_manager.release.assert_called_with("sandbox-123")

    @pytest.mark.asyncio
    async def test_destroy_sandbox_without_id(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test destroy_sandbox with no sandbox_id is a no-op."""
        state = {}
        await env.destroy_sandbox(state)
        mock_sandbox_manager.release.assert_not_called()

    @pytest.mark.asyncio
    async def test_bash_executes_command(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that bash tool executes commands through the manager."""
        state = {"trajectory_id": "traj-1"}
        state = await env.setup_state(state)

        # Simulate sandbox being ready
        state["sandbox_state"]["ready"] = True

        output = await env.bash(
            command="ls -la",
            sandbox_id="sandbox-123",
            sandbox_state=state["sandbox_state"],
        )

        mock_sandbox_manager.execute_command.assert_called_with(
            "sandbox-123",
            "ls -la",
            working_dir=None,
            timeout=30,
        )
        assert output == "command output"

    @pytest.mark.asyncio
    async def test_bash_waits_for_ready(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that bash waits for sandbox to be ready."""
        state = {"trajectory_id": "traj-1"}
        state = await env.setup_state(state)

        # Sandbox not ready yet
        state["sandbox_state"]["ready"] = False

        await env.bash(
            command="ls -la",
            sandbox_id="sandbox-123",
            sandbox_state=state["sandbox_state"],
        )

        mock_sandbox_manager.wait_for_ready.assert_called_with("sandbox-123")
        assert state["sandbox_state"]["ready"] is True

    def test_update_tool_args_for_bash(self, env: NewSandboxEnv):
        """Test update_tool_args injects sandbox state for bash."""
        state = {
            "sandbox_id": "sandbox-123",
            "sandbox_state": {"ready": True},
            "working_dir": "/app",
        }

        updated = env.update_tool_args(
            tool_name="bash",
            tool_args={"command": "ls"},
            messages=[],
            state=state,
        )

        assert updated["sandbox_id"] == "sandbox-123"
        assert updated["sandbox_state"] == {"ready": True}
        assert updated["working_dir"] == "/app"

    def test_active_sandboxes_property(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that active_sandboxes returns IDs from manager."""
        mock_sandbox1 = MagicMock()
        mock_sandbox1.id = "sandbox-1"
        mock_sandbox2 = MagicMock()
        mock_sandbox2.id = "sandbox-2"
        mock_sandbox_manager.get_active_resources.return_value = [mock_sandbox1, mock_sandbox2]

        active = env.active_sandboxes

        assert active == {"sandbox-1", "sandbox-2"}

    def test_sandbox_client_property(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that sandbox_client returns the manager's client."""
        assert env.sandbox_client is mock_sandbox_manager._sandbox_client

    def test_with_retry_property(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that with_retry returns the manager's retry wrapper."""
        assert env.with_retry is mock_sandbox_manager._with_retry

    def test_get_sandbox_errors_for_rollout(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test getting errors for a rollout."""
        mock_error = MagicMock()
        mock_sandbox_manager.get_errors_for_rollout.return_value = [mock_error]

        errors = env.get_sandbox_errors_for_rollout("rollout-1")

        mock_sandbox_manager.get_errors_for_rollout.assert_called_with("rollout-1")
        assert len(errors) == 1

    def test_get_sandbox(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test getting a sandbox by ID."""
        sandbox = env.get_sandbox("sandbox-123")
        mock_sandbox_manager.get_resource.assert_called_with("sandbox-123")
        assert sandbox is not None

    def test_is_sandbox_healthy_when_ready(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test is_sandbox_healthy returns True for READY sandbox."""
        mock_sandbox = MagicMock()
        mock_sandbox.state = ResourceState.READY
        mock_sandbox_manager.get_resource.return_value = mock_sandbox

        assert env.is_sandbox_healthy("sandbox-123") is True

    def test_is_sandbox_healthy_when_error(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test is_sandbox_healthy returns False for ERROR sandbox."""
        mock_sandbox = MagicMock()
        mock_sandbox.state = ResourceState.ERROR
        mock_sandbox_manager.get_resource.return_value = mock_sandbox

        assert env.is_sandbox_healthy("sandbox-123") is False

    def test_is_sandbox_healthy_unknown(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test is_sandbox_healthy returns False for unknown sandbox."""
        mock_sandbox_manager.get_resource.return_value = None

        assert env.is_sandbox_healthy("unknown") is False


class TestNewSandboxEnvTeardown:
    """Tests for teardown behavior."""

    @pytest.fixture
    def env(self, mock_dataset, mock_sandbox_manager, mock_create_sandbox_request):
        """Create a NewSandboxEnv for testing."""
        env = NewSandboxEnv(
            dataset=mock_dataset,
            enable_health_monitoring=False,
        )
        return env

    @pytest.mark.asyncio
    async def test_teardown_sandboxes(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that teardown releases all sandboxes."""
        await env.teardown_sandboxes()
        mock_sandbox_manager.release_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_teardown_sandbox_client(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that teardown shuts down the client."""
        await env.teardown_sandbox_client()
        mock_sandbox_manager.teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_delete_sandboxes(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test the bulk_delete_sandboxes method."""
        env.logger = MagicMock()
        global_ids_to_delete = ["sandbox1", "sandbox3"]
        await env.bulk_delete_sandboxes(global_ids_to_delete)

        mock_sandbox_manager.bulk_delete.assert_called_once_with(global_ids_to_delete)


class TestNewSandboxEnvMetrics:
    """Tests for the monitor rubric metrics."""

    @pytest.fixture
    def env(self, mock_dataset, mock_sandbox_manager, mock_create_sandbox_request):
        """Create a NewSandboxEnv for testing."""
        return NewSandboxEnv(
            dataset=mock_dataset,
            enable_health_monitoring=False,
        )

    @pytest.mark.asyncio
    async def test_metrics_with_sandbox_state(self, env: NewSandboxEnv):
        """Test metrics extraction from sandbox state."""
        from verifiers.envs.experimental.new_sandbox_env import NewSandboxMonitorRubric

        rubric = NewSandboxMonitorRubric()

        mock_sandbox = MagicMock()
        mock_sandbox.state = ResourceState.READY
        mock_sandbox.error = None

        state = {
            "sandbox_state": {
                "ready": True,
                "ready_wait_time": 1.5,
                "command_execution_times": [0.5, 1.0, 1.5],
            },
            "managed_sandbox": mock_sandbox,
        }

        ready_time = await rubric.sandbox_ready_wait_time(state)
        assert ready_time == 1.5

        exec_time = await rubric.sandbox_command_execution_time(state)
        assert exec_time == 1.0  # average of [0.5, 1.0, 1.5]

        sandbox_state = await rubric.sandbox_state_metric(state)
        assert sandbox_state == "ready"

        error_count = await rubric.sandbox_error_count(state)
        assert error_count == 0

    @pytest.mark.asyncio
    async def test_metrics_without_sandbox_state(self, env: NewSandboxEnv):
        """Test metrics return defaults without sandbox state."""
        from verifiers.envs.experimental.new_sandbox_env import NewSandboxMonitorRubric

        rubric = NewSandboxMonitorRubric()
        state = {}

        ready_time = await rubric.sandbox_ready_wait_time(state)
        assert ready_time == 0.0

        exec_time = await rubric.sandbox_command_execution_time(state)
        assert exec_time == 0.0

        sandbox_state = await rubric.sandbox_state_metric(state)
        assert sandbox_state == "none"

        error_count = await rubric.sandbox_error_count(state)
        assert error_count == 0

    @pytest.mark.asyncio
    async def test_metrics_with_error(self, env: NewSandboxEnv):
        """Test error count metric with error."""
        from verifiers.envs.experimental.new_sandbox_env import NewSandboxMonitorRubric

        rubric = NewSandboxMonitorRubric()

        mock_sandbox = MagicMock()
        mock_sandbox.state = ResourceState.ERROR
        mock_sandbox.error = RuntimeError("test error")

        state = {
            "sandbox_state": {"ready": False, "ready_wait_time": 0, "command_execution_times": []},
            "managed_sandbox": mock_sandbox,
        }

        error_count = await rubric.sandbox_error_count(state)
        assert error_count == 1

        sandbox_state = await rubric.sandbox_state_metric(state)
        assert sandbox_state == "error"
