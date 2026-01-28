"""
Sandbox Environment

Provides tool-based interaction with Prime Sandboxes.
Supports both legacy mode (direct sandbox management) and new resource manager mode.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import tenacity as tc
from prime_sandboxes import CommandTimeoutError

import verifiers as vf
from verifiers.resources import (
    AllocationMode,
    ResourceHandle,
    SandboxConfig,
    SandboxResource,
    SandboxResourceManager,
)
from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)

try:
    from prime_sandboxes import (
        AdvancedConfigs,
        AsyncSandboxClient,
        CreateSandboxRequest,
        SandboxClient,
    )
    from prime_sandboxes.core import APIClient
except ImportError:
    raise ImportError(
        "prime-sandboxes is not installed. Please install it with `uv pip install prime-sandboxes`."
    )


logger = logging.getLogger(__name__)


class ThreadedAsyncSandboxClient:
    """
    Mirrors AsyncSandboxClient's interface but dispatches each method call to a
    ThreadPoolExecutor where each thread maintains its own client via
    thread-local storage.

    NOTE: Consider using SandboxResourceManager instead for new code.
    """

    def __init__(
        self,
        max_workers: int = 100,
        max_connections: int = 100,
        max_keepalive_connections: int = 50,
        **client_kwargs,
    ):
        """Initialize the threaded sandbox client."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sandbox-client-executor",
        )
        self.client_kwargs = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
            **client_kwargs,
        }

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Dynamically proxy attribute access to dispatch method calls to the thread pool."""

        @functools.wraps(getattr(AsyncSandboxClient, name, lambda: None))
        async def wrapper(*args, **kwargs):
            def run_in_thread():
                loop = get_or_create_thread_loop()
                sandbox_client = get_or_create_thread_attr(
                    "sandbox_client",
                    AsyncSandboxClient,
                    **self.client_kwargs,
                )
                method = getattr(sandbox_client, name)
                return loop.run_until_complete(method(*args, **kwargs))

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, run_in_thread)

        return wrapper

    def teardown(self, wait: bool = True) -> None:
        """Teardown the thread pool executor."""
        self.executor.shutdown(wait=wait)


class SandboxState(TypedDict):
    ready: bool
    ready_wait_time: float
    command_execution_times: list[float]


class SandboxCreationError(vf.SandboxError): ...


class SandboxNotReadyError(vf.SandboxError): ...


class SandboxMonitorRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.sandbox_ready_wait_time)
        self.add_metric(self.sandbox_command_execution_time)

    async def sandbox_ready_wait_time(self, state: vf.State) -> float:
        return state["sandbox_state"]["ready_wait_time"]

    async def sandbox_command_execution_time(self, state: vf.State) -> float:
        command_execution_times = state["sandbox_state"]["command_execution_times"]
        return (
            sum(command_execution_times) / len(command_execution_times)
            if len(command_execution_times) > 0
            else 0.0
        )


class SandboxEnv(vf.StatefulToolEnv):
    """
    Tool-based environment for interacting with Prime Sandboxes.

    Supports two modes of operation:
    1. Legacy mode (default): Direct sandbox management with one sandbox per rollout
    2. Resource manager mode: Use SandboxResourceManager for advanced allocation modes

    Example (legacy mode):
        class MyEnv(SandboxEnv):
            async def env_response(self, messages, state, **kwargs):
                # Use self.bash() tool
                pass

    Example (resource manager mode):
        manager = SandboxResourceManager(
            config=SandboxConfig(docker_image="python:3.11"),
            mode=AllocationMode.POOL,
            pool_size=10,
        )
        env = SandboxEnv(resource_manager=manager)
    """

    def __init__(
        self,
        # Resource manager mode
        resource_manager: SandboxResourceManager | None = None,
        # Legacy mode parameters
        sandbox_name: str = "sandbox-env",
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        timeout_per_command_seconds: int = 30,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        # Retry configuration
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        stop_errors: list[type[Exception]] | None = None,
        # Client configuration (legacy mode)
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        **kwargs,
    ):
        super().__init__(
            stop_errors=stop_errors if stop_errors is not None else [vf.SandboxError],
            **kwargs,
        )
        self.add_rubric(SandboxMonitorRubric())
        self.timeout_per_command_seconds = timeout_per_command_seconds

        # Determine mode based on resource_manager
        self._resource_manager = resource_manager
        self._use_resource_manager = resource_manager is not None

        if self._use_resource_manager:
            # Resource manager mode
            self.logger.info(
                f"Using resource manager mode: {resource_manager.mode.value}"
            )
        else:
            # Legacy mode
            self.sandbox_client = ThreadedAsyncSandboxClient(
                max_workers=sandbox_client_max_workers,
                max_connections=sandbox_client_max_connections,
                max_keepalive_connections=sandbox_client_max_keepalive_connections,
            )
            self.sandbox_request = CreateSandboxRequest(
                name=sandbox_name,
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
            )
            self.active_sandboxes: set[str] = set()
            self.with_retry = tc.AsyncRetrying(
                stop=tc.stop_after_attempt(max_retries),
                wait=tc.wait_exponential_jitter(
                    initial=base_delay,
                    exp_base=backoff_factor,
                    max=max_backoff_seconds,
                    jitter=jitter,
                ),
                before_sleep=tc.before_sleep_log(self.logger, logging.WARNING),
                reraise=True,
            ).wraps

        self.add_tool(
            self.bash, args_to_skip=["sandbox_id", "working_dir", "sandbox_state"]
        )

    @property
    def resource_manager(self) -> SandboxResourceManager | None:
        """Get the resource manager if in resource manager mode."""
        return self._resource_manager

    async def _wait_for_sandbox_ready(
        self, sandbox_state: SandboxState, sandbox_id: str
    ):
        """Wait for sandbox to be created (legacy mode)."""
        s = time.time()
        self.logger.debug(f"Waiting for sandbox {sandbox_id} to be ready")
        try:
            await self.sandbox_client.wait_for_creation(sandbox_id)
            sandbox_state["ready"] = True
        except Exception as e:
            raise SandboxNotReadyError(e)
        ready_wait_time = time.time() - s
        sandbox_state["ready_wait_time"] = ready_wait_time
        self.logger.debug(f"Waited {ready_wait_time:.1f}s for sandbox to be ready")

    async def bash(
        self,
        command: str,
        sandbox_id: str,
        sandbox_state: SandboxState,
        working_dir: str | None = None,
    ) -> str:
        """Execute `command` inside persistent sandbox container."""
        if self._use_resource_manager:
            # Resource manager mode - command execution is handled by the manager
            # This path is used when tool is called directly
            return await self._execute_command_via_manager(
                command, sandbox_id, sandbox_state, working_dir
            )

        # Legacy mode
        if not sandbox_state["ready"]:
            await self._wait_for_sandbox_ready(sandbox_state, sandbox_id)

        s = time.time()
        self.logger.debug(f"Executing command {command} in sandbox {sandbox_id}")
        try:
            results = await self.sandbox_client.execute_command(
                sandbox_id,
                command,
                working_dir=working_dir,
                timeout=self.timeout_per_command_seconds,
            )
        except CommandTimeoutError:
            timeout_msg = f"Command timed out after {self.timeout_per_command_seconds}s"
            self.logger.warning(f"{timeout_msg} in sandbox {sandbox_id}")
            sandbox_state["command_execution_times"].append(
                self.timeout_per_command_seconds
            )
            return f"Error: {timeout_msg}"
        except Exception as e:
            raise vf.SandboxError from e
        command_execution_time = time.time() - s
        sandbox_state["command_execution_times"].append(command_execution_time)
        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        combined = stdout
        if stderr:
            if combined:
                combined = f"{combined}\nstderr:\n{stderr}"
            else:
                combined = f"stderr:\n{stderr}"
        output = combined or "(no output)"
        self.logger.debug(
            f"Executed command in {command_execution_time:.1f}s. Got output: {output}"
        )
        return output

    async def _execute_command_via_manager(
        self,
        command: str,
        sandbox_id: str,
        sandbox_state: SandboxState,
        working_dir: str | None = None,
    ) -> str:
        """Execute command using resource manager."""
        if not self._resource_manager:
            raise RuntimeError("Resource manager not configured")

        # Wait for ready if needed
        if not sandbox_state["ready"]:
            s = time.time()
            await self._resource_manager.client.wait_for_creation(sandbox_id)
            sandbox_state["ready"] = True
            sandbox_state["ready_wait_time"] = time.time() - s

        s = time.time()
        self.logger.debug(f"Executing command {command} in sandbox {sandbox_id}")

        try:
            results = await self._resource_manager.client.execute_command(
                sandbox_id,
                command,
                working_dir=working_dir,
                timeout=self.timeout_per_command_seconds,
            )
        except CommandTimeoutError:
            timeout_msg = f"Command timed out after {self.timeout_per_command_seconds}s"
            self.logger.warning(f"{timeout_msg} in sandbox {sandbox_id}")
            sandbox_state["command_execution_times"].append(
                float(self.timeout_per_command_seconds)
            )
            return f"Error: {timeout_msg}"
        except Exception as e:
            raise vf.SandboxError from e

        command_execution_time = time.time() - s
        sandbox_state["command_execution_times"].append(command_execution_time)

        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        combined = stdout
        if stderr:
            combined = f"{combined}\nstderr:\n{stderr}" if combined else f"stderr:\n{stderr}"

        output = combined or "(no output)"
        self.logger.debug(
            f"Executed command in {command_execution_time:.1f}s. Got output: {output}"
        )
        return output

    async def post_rollout(self, state: vf.State):
        """
        Override for custom post-rollout logic. For example, if sandbox state is needed for reward functions,
        run computation here and cache the result in state before sandbox is destroyed.
        """
        pass

    @vf.cleanup
    async def destroy_sandbox(self, state: vf.State):
        """Cleanup sandbox after rollout."""
        await self.post_rollout(state)

        if self._use_resource_manager:
            # Resource manager mode - release handle
            handle: ResourceHandle[SandboxResource] | None = state.get("_resource_handle")
            if handle:
                await self._resource_manager.release(handle)
                state["_resource_handle"] = None
            return

        # Legacy mode
        sandbox_id = state.get("sandbox_id")
        if sandbox_id is None:
            return

        async def _delete_sandbox(sandbox_id: str):
            await self.sandbox_client.delete(sandbox_id)
            self.active_sandboxes.discard(sandbox_id)
            self.logger.debug(f"Deleted sandbox {sandbox_id}")

        try:
            await self.with_retry(_delete_sandbox)(sandbox_id)
        except Exception as e:
            # only warn, not raise an error on deletion
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    def get_sandbox_request(self, state: vf.State) -> CreateSandboxRequest:
        """Return sandbox request for this rollout. Override to customize per-state."""
        return self.sandbox_request.model_copy()

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create per-rollout sandbox."""
        if self._use_resource_manager:
            # Resource manager mode
            await self._resource_manager.startup()
            handle = await self._resource_manager.acquire(state)
            state["_resource_handle"] = handle
            state["sandbox_id"] = handle.resource_id
            state["sandbox_state"] = handle.resource.sandbox_state
            return await super().setup_state(state, **kwargs)

        # Legacy mode
        request = self.get_sandbox_request(state)
        try:
            sandbox = await self.with_retry(self.sandbox_client.create)(request)
        except Exception as e:
            raise SandboxCreationError(e)
        self.active_sandboxes.add(sandbox.id)
        self.logger.debug(f"Created sandbox {sandbox.id}")
        state["sandbox_id"] = sandbox.id
        state["sandbox_state"] = {
            "ready": False,
            "ready_wait_time": 0.0,
            "command_execution_times": [],
        }
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        updated_args = dict(tool_args)
        if tool_name == "bash":
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["sandbox_state"] = state["sandbox_state"]
            updated_args["working_dir"] = state.get("working_dir")
        return updated_args

    async def bulk_delete_sandboxes(self, global_ids: list[str]) -> None:
        """Delete multiple sandboxes by their global IDs (legacy mode only)."""
        if self._use_resource_manager:
            self.logger.warning("bulk_delete_sandboxes not supported in resource manager mode")
            return

        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(global_ids)
            self.logger.debug(f"Bulk deleted sandboxes: {global_ids}")
            self.active_sandboxes.difference_update(global_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {global_ids}: {e}")

    @vf.teardown
    async def teardown_sandboxes(self):
        """Delete all active sandboxes."""
        if self._use_resource_manager:
            # Resource manager handles its own teardown
            await self._resource_manager.teardown()
            return

        # Legacy mode
        if len(self.active_sandboxes) == 0:
            return
        self.logger.info(f"Deleting {len(self.active_sandboxes)} remaining sandboxes")

        # Use sync client for teardown - avoids event loop issues during shutdown
        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)

        # Delete in batches of 100
        batch_size = 100
        for i in range(0, len(sandbox_ids), batch_size):
            batch = sandbox_ids[i : i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sandbox_id in batch:
                    self.active_sandboxes.discard(sandbox_id)
                self.logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                self.logger.warning(f"Bulk delete failed for batch: {e}")

    @vf.teardown
    async def teardown_sandbox_client(self):
        """Teardown the sandbox client."""
        if self._use_resource_manager:
            # Resource manager handles client teardown
            return

        self.sandbox_client.teardown()


# Convenience functions for using sandboxes without subclassing


async def with_sandbox(
    state: vf.State,
    manager: SandboxResourceManager,
    setup_fn: Callable[[ResourceHandle[SandboxResource]], Any] | None = None,
) -> ResourceHandle[SandboxResource]:
    """
    Acquire a sandbox for a state dict.

    Usage in custom environments:
        manager = SandboxResourceManager(...)
        await manager.startup()

        async def setup_state(self, state):
            handle = await with_sandbox(state, manager)
            state["_sandbox_handle"] = handle
            return state

        async def cleanup(self, state):
            handle = state.get("_sandbox_handle")
            if handle:
                await manager.release(handle)
    """
    await manager.startup()
    handle = await manager.acquire(state)

    # Store in state for later access
    state["sandbox_id"] = handle.resource_id
    state["sandbox_state"] = handle.resource.sandbox_state
    state["_resource_handle"] = handle

    if setup_fn:
        await setup_fn(handle)

    return handle


async def execute_in_sandbox(
    state: vf.State,
    command: str,
    manager: SandboxResourceManager | None = None,
    working_dir: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Execute a command in the sandbox associated with a state.

    Usage in custom environments:
        result = await execute_in_sandbox(state, "echo hello", manager)
    """
    handle: ResourceHandle[SandboxResource] | None = state.get("_resource_handle")

    if handle is None and manager is None:
        raise ValueError("No sandbox handle in state and no manager provided")

    if handle is not None:
        # Use handle's manager reference (if we had one) or provided manager
        if manager is None:
            raise ValueError("Manager required when handle doesn't have manager reference")
        return await manager.execute_command(handle, command, working_dir, timeout)

    # No handle - need to acquire first
    if manager is None:
        raise ValueError("Manager required to acquire sandbox")

    handle = await with_sandbox(state, manager)
    return await manager.execute_command(handle, command, working_dir, timeout)
