"""Tests for the NewSandboxEnv class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datasets import Dataset

from verifiers.envs.experimental.new_sandbox_env import NewSandboxEnv
from verifiers.envs.experimental.resource_managers.base import ResourceState
from verifiers.envs.experimental.resource_managers.sandbox_manager import ManagedSandbox
from verifiers.envs.experimental.resource_managers.errors import SandboxFailureInfo


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
        mock_sandbox = MagicMock(spec=ManagedSandbox)
        mock_sandbox.id = "sandbox-123"
        mock_sandbox.state = ResourceState.READY
        mock_sandbox.rollout_id = None
        mock_sandbox.error = None
        mock_sandbox.failure_info = SandboxFailureInfo()
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

        # Mock get_failure_info
        mock_manager.get_failure_info = MagicMock(return_value=SandboxFailureInfo())

        # Mock teardown
        mock_manager.teardown = MagicMock()

        # Mock print_summary
        mock_manager.print_summary = MagicMock()

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
        mock_sandbox_manager.wait_for_ready.assert_called_once()
        assert state["sandbox_id"] == "sandbox-123"

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_releases(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that cleanup_sandbox releases through the manager."""
        state = {"trajectory_id": "traj-1"}
        state = await env.setup_state(state)

        await env.cleanup_sandbox(state)

        mock_sandbox_manager.release.assert_called_with("sandbox-123")

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_without_id(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test cleanup_sandbox with no sandbox_id is a no-op."""
        state = {}
        await env.cleanup_sandbox(state)
        mock_sandbox_manager.release.assert_not_called()

    @pytest.mark.asyncio
    async def test_bash_executes_command(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that bash tool executes commands through the manager."""
        output = await env.bash(
            command="ls -la",
            sandbox_id="sandbox-123",
        )

        mock_sandbox_manager.execute_command.assert_called_with(
            "sandbox-123",
            "ls -la",
            timeout=30,
        )
        assert output == "command output"

    def test_update_tool_args_for_bash(self, env: NewSandboxEnv):
        """Test update_tool_args injects sandbox_id for bash."""
        state = {
            "sandbox_id": "sandbox-123",
        }

        updated = env.update_tool_args(
            tool_name="bash",
            tool_args={"command": "ls"},
            messages=[],
            state=state,
        )

        assert updated["sandbox_id"] == "sandbox-123"

    def test_update_tool_args_non_bash(self, env: NewSandboxEnv):
        """Test update_tool_args does not inject sandbox_id for non-bash tools."""
        state = {
            "sandbox_id": "sandbox-123",
        }

        updated = env.update_tool_args(
            tool_name="other_tool",
            tool_args={"arg": "value"},
            messages=[],
            state=state,
        )

        assert "sandbox_id" not in updated

    def test_get_sandbox(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test getting a sandbox from state."""
        state = {"sandbox_id": "sandbox-123"}
        sandbox = env.get_sandbox(state)
        mock_sandbox_manager.get_resource.assert_called_with("sandbox-123")
        assert sandbox is not None

    def test_get_sandbox_no_id(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test getting a sandbox when state has no sandbox_id."""
        state = {}
        sandbox = env.get_sandbox(state)
        assert sandbox is None

    def test_get_failure_info(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test getting failure info for a sandbox."""
        state = {"sandbox_id": "sandbox-123"}
        failure_info = env.get_failure_info(state)
        mock_sandbox_manager.get_failure_info.assert_called_with("sandbox-123")
        assert failure_info is not None
        assert not failure_info.has_failure

    def test_get_failure_info_no_sandbox(self, env: NewSandboxEnv):
        """Test getting failure info when no sandbox exists."""
        state = {}
        failure_info = env.get_failure_info(state)
        assert failure_info is None

    def test_had_sandbox_failure_false(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test had_sandbox_failure returns False when no failure."""
        state = {"sandbox_id": "sandbox-123"}
        assert env.had_sandbox_failure(state) is False

    def test_had_sandbox_failure_true(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test had_sandbox_failure returns True when failure exists."""
        mock_sandbox_manager.get_failure_info.return_value = SandboxFailureInfo(oom=True)
        state = {"sandbox_id": "sandbox-123"}
        assert env.had_sandbox_failure(state) is True

    def test_had_oom(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test had_oom returns True when OOM occurred."""
        mock_sandbox_manager.get_failure_info.return_value = SandboxFailureInfo(oom=True)
        state = {"sandbox_id": "sandbox-123"}
        assert env.had_oom(state) is True

    def test_had_timeout(self, env: NewSandboxEnv, mock_sandbox_manager):
        """Test had_timeout returns True when timeout occurred."""
        mock_sandbox_manager.get_failure_info.return_value = SandboxFailureInfo(command_timeout=True)
        state = {"sandbox_id": "sandbox-123"}
        assert env.had_timeout(state) is True


class TestNewSandboxEnvTeardown:
    """Tests for teardown behavior."""

    @pytest.fixture
    def env(self, mock_dataset, mock_sandbox_manager, mock_create_sandbox_request):
        """Create a NewSandboxEnv for testing."""
        env = NewSandboxEnv(
            dataset=mock_dataset,
        )
        return env

    @pytest.mark.asyncio
    async def test_teardown(
        self, env: NewSandboxEnv, mock_sandbox_manager
    ):
        """Test that teardown releases all sandboxes and shuts down client."""
        await env.teardown()
        mock_sandbox_manager.print_summary.assert_called_once()
        mock_sandbox_manager.release_all.assert_called_once()
        mock_sandbox_manager.teardown.assert_called_once()
