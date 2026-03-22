"""Sandbox environment using centralized resource management."""

import logging
from typing import Any, Awaitable, Callable

from prime_sandboxes import AdvancedConfigs, CreateSandboxRequest

import verifiers as vf
from verifiers.envs.experimental.resource_managers.errors import SandboxFailureInfo
from verifiers.envs.experimental.resource_managers.retry import RetryConfig
from verifiers.envs.experimental.resource_managers.sandbox_manager import (
    BackgroundJob,
    ManagedSandbox,
    SandboxManager,
    SetupCallback,
)

logger = logging.getLogger(__name__)


class NewSandboxEnv(vf.StatefulToolEnv):
    """Sandbox environment using SandboxManager for lifecycle management.

    Simpler than SandboxEnv - delegates all sandbox operations to the manager.
    All failure tracking is handled by the manager, not environment state.
    """

    def __init__(
        self,
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        timeout_minutes: int = 60,
        timeout_per_command: int = 30,
        environment_vars: dict[str, str] | None = None,
        gpu_count: int = 0,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        labels: list[str] | None = None,
        # Manager configuration
        retry_config: RetryConfig | None = None,
        wait_for_creation_max_attempts: int = 120,
        sandbox_client_max_workers: int = 100,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.timeout_per_command = timeout_per_command

        # Default sandbox request
        self.sandbox_request = CreateSandboxRequest(
            name="sandbox",
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_minutes=timeout_minutes,
            environment_vars=environment_vars,
            team_id=team_id,
            advanced_configs=advanced_configs,
            labels=labels or [],
        )

        self.sandbox_manager = SandboxManager(
            default_request=self.sandbox_request,
            timeout_per_command=timeout_per_command,
            retry_config=retry_config,
            wait_for_creation_max_attempts=wait_for_creation_max_attempts,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
        )
        self.add_tool(self.bash, args_to_skip=["sandbox_id"])

    def get_sandbox_request(self, state: vf.State) -> CreateSandboxRequest:
        """Override to customize sandbox per rollout."""
        return self.sandbox_request.model_copy()

    def get_setup_callback(self, state: vf.State) -> SetupCallback | None:
        """Override to run setup commands after sandbox is ready.

        Example:
            async def get_setup_callback(self, state):
                async def setup(sandbox_id):
                    await self.sandbox_manager.execute_command(
                        sandbox_id, "pip install numpy pandas"
                    )
                return setup
        """
        return None

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Create sandbox for this rollout."""
        request = self.get_sandbox_request(state)
        rollout_id = state.get("trajectory_id")

        sandbox = await self.sandbox_manager.acquire(rollout_id=rollout_id, request=request)
        setup_callback = self.get_setup_callback(state)
        await self.sandbox_manager.wait_for_ready(sandbox.id, setup_callback=setup_callback)

        state["sandbox_id"] = sandbox.id
        return await super().setup_state(state, **kwargs)

    async def bash(self, command: str, sandbox_id: str) -> str:
        """Execute command in sandbox."""
        return await self.sandbox_manager.execute_command(
            sandbox_id, command, timeout=self.timeout_per_command
        )

    async def run_background_job(
        self,
        state: vf.State,
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ) -> BackgroundJob:
        """Run a long-running command as a background job.

        Use this for commands that may take longer than the command timeout.
        """
        sandbox_id = state["sandbox_id"]
        return await self.sandbox_manager.run_background_job(
            sandbox_id=sandbox_id,
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            poll_interval=poll_interval,
        )

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject sandbox_id into bash calls."""
        if tool_name == "bash":
            tool_args["sandbox_id"] = state["sandbox_id"]
        return tool_args

    # Failure info access methods

    def get_failure_info(self, state: vf.State) -> SandboxFailureInfo | None:
        """Get failure info for the sandbox in this rollout.

        Use this in scoring to check if the sandbox had issues.
        """
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            return self.sandbox_manager.get_failure_info(sandbox_id)
        return None

    def get_sandbox(self, state: vf.State) -> ManagedSandbox | None:
        """Get the ManagedSandbox for this rollout."""
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            resource = self.sandbox_manager.get_resource(sandbox_id)
            return resource if isinstance(resource, ManagedSandbox) else None
        return None

    def had_sandbox_failure(self, state: vf.State) -> bool:
        """Check if the sandbox had any failure."""
        failure_info = self.get_failure_info(state)
        return failure_info.has_failure if failure_info else False

    def had_oom(self, state: vf.State) -> bool:
        """Check if the sandbox ran out of memory."""
        failure_info = self.get_failure_info(state)
        return failure_info.oom if failure_info else False

    def had_timeout(self, state: vf.State) -> bool:
        """Check if the sandbox had a timeout (sandbox-level or command)."""
        failure_info = self.get_failure_info(state)
        if not failure_info:
            return False
        return failure_info.timeout or failure_info.command_timeout

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        """Release sandbox after rollout."""
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.sandbox_manager.release(sandbox_id)

    @vf.teardown
    async def teardown(self) -> None:
        """Release all sandboxes and print summary."""
        self.sandbox_manager.print_summary()
        await self.sandbox_manager.release_all()
        self.sandbox_manager.teardown()
