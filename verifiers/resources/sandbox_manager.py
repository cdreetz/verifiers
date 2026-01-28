"""
Sandbox Resource Manager

Manages Prime Sandbox resources with support for different allocation modes:
- ONE_TO_ONE: Each rollout gets a dedicated sandbox (destroyed after use)
- POOL: Sandboxes are pooled and reused across rollouts
- SHARED: Single sandbox shared across all rollouts
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import tenacity as tc

from verifiers.resources.resource_manager import (
    AllocationMode,
    BaseResourceManager,
    ResourceHandle,
    ResourceMetrics,
)
from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

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


class SandboxState(TypedDict):
    """Per-sandbox state tracking."""

    ready: bool
    ready_wait_time: float
    command_execution_times: list[float]


@dataclass
class SandboxConfig:
    """Configuration for sandbox creation."""

    name: str = "sandbox"
    docker_image: str = "python:3.11-slim"
    start_command: str = "tail -f /dev/null"
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 5
    gpu_count: int = 0
    timeout_minutes: int = 60
    environment_vars: dict[str, str] | None = None
    team_id: str | None = None
    advanced_configs: AdvancedConfigs | None = None

    def to_request(self, suffix: str = "") -> CreateSandboxRequest:
        """Convert to a CreateSandboxRequest."""
        name = f"{self.name}-{suffix}" if suffix else self.name
        return CreateSandboxRequest(
            name=name,
            docker_image=self.docker_image,
            start_command=self.start_command,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            disk_size_gb=self.disk_size_gb,
            gpu_count=self.gpu_count,
            timeout_minutes=self.timeout_minutes,
            environment_vars=self.environment_vars,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
        )


@dataclass
class SandboxResource:
    """A sandbox resource with its state."""

    sandbox_id: str
    config: SandboxConfig
    sandbox_state: SandboxState = field(
        default_factory=lambda: SandboxState(
            ready=False,
            ready_wait_time=0.0,
            command_execution_times=[],
        )
    )
    working_dir: str | None = None
    in_use: bool = False


class ThreadedAsyncSandboxClient:
    """
    Thread-safe wrapper for AsyncSandboxClient.

    Dispatches each method call to a ThreadPoolExecutor where each thread
    maintains its own client via thread-local storage. This avoids event
    loop issues when calling async code from sync contexts.
    """

    def __init__(
        self,
        max_workers: int = 100,
        max_connections: int = 100,
        max_keepalive_connections: int = 50,
        **client_kwargs,
    ):
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
        """Proxy method calls to thread pool."""

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

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=wait)


class SandboxResourceManager(BaseResourceManager[SandboxResource]):
    """
    Manages Prime Sandbox resources with different allocation modes.

    Modes:
        ONE_TO_ONE: Each rollout gets a dedicated sandbox (default)
        POOL: Sandboxes are pooled and reused
        SHARED: Single sandbox for all rollouts

    Usage:
        # Create manager
        manager = SandboxResourceManager(
            config=SandboxConfig(docker_image="python:3.11"),
            mode=AllocationMode.POOL,
            pool_size=10,
        )

        # Start manager
        await manager.startup()

        # Acquire sandbox for rollout
        async with manager.acquire_context(state) as handle:
            sandbox = handle.resource
            result = await manager.execute_command(
                handle, "echo hello"
            )

        # Cleanup
        await manager.teardown()
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        mode: AllocationMode = AllocationMode.ONE_TO_ONE,
        pool_size: int = 10,
        max_concurrent: int | None = None,
        client_max_workers: int = 100,
        client_max_connections: int = 100,
        client_max_keepalive_connections: int = 50,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        command_timeout_seconds: int = 30,
    ):
        super().__init__(mode=mode, max_resources=max_concurrent)

        self.config = config or SandboxConfig()
        self.pool_size = pool_size
        self.command_timeout_seconds = command_timeout_seconds

        # Client configuration
        self._client: ThreadedAsyncSandboxClient | None = None
        self._client_max_workers = client_max_workers
        self._client_max_connections = client_max_connections
        self._client_max_keepalive_connections = client_max_keepalive_connections

        # Retry configuration
        self._retry_config = {
            "max_retries": max_retries,
            "base_delay": base_delay,
            "backoff_factor": backoff_factor,
            "max_backoff_seconds": max_backoff_seconds,
            "jitter": jitter,
        }

        # Resource tracking
        self._active_sandboxes: set[str] = set()
        self._pool: asyncio.Queue[SandboxResource] | None = None
        self._shared_resource: SandboxResource | None = None
        self._pool_lock = asyncio.Lock()

    @property
    def client(self) -> ThreadedAsyncSandboxClient:
        """Get the sandbox client, initializing if needed."""
        if self._client is None:
            self._client = ThreadedAsyncSandboxClient(
                max_workers=self._client_max_workers,
                max_connections=self._client_max_connections,
                max_keepalive_connections=self._client_max_keepalive_connections,
            )
        return self._client

    @property
    def active_sandbox_ids(self) -> set[str]:
        """Get set of active sandbox IDs."""
        return self._active_sandboxes.copy()

    def _create_retry_wrapper(self):
        """Create tenacity retry wrapper."""
        return tc.AsyncRetrying(
            stop=tc.stop_after_attempt(self._retry_config["max_retries"]),
            wait=tc.wait_exponential_jitter(
                initial=self._retry_config["base_delay"],
                exp_base=self._retry_config["backoff_factor"],
                max=self._retry_config["max_backoff_seconds"],
                jitter=self._retry_config["jitter"],
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        ).wraps

    async def _do_startup(self) -> None:
        """Initialize the manager based on mode."""
        if self._mode == AllocationMode.POOL:
            self._pool = asyncio.Queue()
            # Pre-create pool sandboxes
            tasks = [
                self._create_sandbox(f"pool-{i}")
                for i in range(self.pool_size)
            ]
            sandboxes = await asyncio.gather(*tasks, return_exceptions=True)

            for sandbox in sandboxes:
                if isinstance(sandbox, SandboxResource):
                    await self._pool.put(sandbox)
                else:
                    self.logger.warning(f"Failed to create pool sandbox: {sandbox}")

            self.logger.info(
                f"Initialized sandbox pool with {self._pool.qsize()}/{self.pool_size} sandboxes"
            )

        elif self._mode == AllocationMode.SHARED:
            self._shared_resource = await self._create_sandbox("shared")
            self.logger.info(f"Initialized shared sandbox: {self._shared_resource.sandbox_id}")

    async def _create_sandbox(
        self,
        suffix: str = "",
        config: SandboxConfig | None = None,
    ) -> SandboxResource:
        """Create a new sandbox."""
        config = config or self.config
        request = config.to_request(suffix)
        retry = self._create_retry_wrapper()

        sandbox = await retry(self.client.create)(request)
        sandbox_id = sandbox.id

        self._active_sandboxes.add(sandbox_id)
        metrics = self._create_metrics(sandbox_id)

        self.logger.debug(f"Created sandbox {sandbox_id}")

        return SandboxResource(
            sandbox_id=sandbox_id,
            config=config,
        )

    async def _wait_for_ready(self, resource: SandboxResource) -> None:
        """Wait for a sandbox to be ready."""
        if resource.sandbox_state["ready"]:
            return

        start = time.time()
        self.logger.debug(f"Waiting for sandbox {resource.sandbox_id} to be ready")

        await self.client.wait_for_creation(resource.sandbox_id)

        ready_wait_time = time.time() - start
        resource.sandbox_state["ready"] = True
        resource.sandbox_state["ready_wait_time"] = ready_wait_time
        self._mark_ready(resource.sandbox_id)

        self.logger.debug(
            f"Sandbox {resource.sandbox_id} ready in {ready_wait_time:.1f}s"
        )

    async def _do_acquire(self, state: dict[str, Any]) -> ResourceHandle[SandboxResource]:
        """Acquire a sandbox based on allocation mode."""
        if self._mode == AllocationMode.ONE_TO_ONE:
            # Create new sandbox for this rollout
            suffix = state.get("trajectory_id", str(time.time()))[:8]
            resource = await self._create_sandbox(suffix)

        elif self._mode == AllocationMode.POOL:
            # Get from pool
            if self._pool is None:
                raise RuntimeError("Pool not initialized")
            resource = await self._pool.get()

        elif self._mode == AllocationMode.SHARED:
            # Return shared resource
            if self._shared_resource is None:
                raise RuntimeError("Shared sandbox not initialized")
            resource = self._shared_resource

        else:
            raise ValueError(f"Unknown allocation mode: {self._mode}")

        resource.in_use = True
        metrics = self._resource_metrics.get(resource.sandbox_id)
        if metrics is None:
            metrics = self._create_metrics(resource.sandbox_id)

        return ResourceHandle(
            resource=resource,
            resource_id=resource.sandbox_id,
            metrics=metrics,
        )

    async def _do_release(self, handle: ResourceHandle[SandboxResource]) -> None:
        """Release a sandbox based on allocation mode."""
        resource = handle.resource
        resource.in_use = False

        if self._mode == AllocationMode.ONE_TO_ONE:
            # Destroy sandbox
            await self._destroy_sandbox(resource.sandbox_id)

        elif self._mode == AllocationMode.POOL:
            # Return to pool (optionally reset state)
            if self._pool is not None:
                resource.sandbox_state["command_execution_times"] = []
                await self._pool.put(resource)

        elif self._mode == AllocationMode.SHARED:
            # Nothing to do - sandbox stays alive
            pass

    async def _destroy_sandbox(self, sandbox_id: str) -> None:
        """Destroy a single sandbox."""
        retry = self._create_retry_wrapper()
        try:
            await retry(self.client.delete)(sandbox_id)
            self._active_sandboxes.discard(sandbox_id)
            self._mark_destroyed(sandbox_id)
            self.logger.debug(f"Destroyed sandbox {sandbox_id}")
        except Exception as e:
            self.logger.warning(f"Failed to destroy sandbox {sandbox_id}: {e}")

    async def _do_teardown(self) -> None:
        """Cleanup all sandboxes."""
        sandbox_ids = list(self._active_sandboxes)

        if not sandbox_ids:
            if self._client:
                self._client.shutdown()
            return

        self.logger.info(f"Destroying {len(sandbox_ids)} sandboxes")

        # Use sync client for teardown to avoid event loop issues
        sync_client = SandboxClient(APIClient())

        # Delete in batches
        batch_size = 100
        for i in range(0, len(sandbox_ids), batch_size):
            batch = sandbox_ids[i : i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sid in batch:
                    self._active_sandboxes.discard(sid)
                self.logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                self.logger.warning(f"Bulk delete failed: {e}")

        if self._client:
            self._client.shutdown()

    async def execute_command(
        self,
        handle: ResourceHandle[SandboxResource],
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """
        Execute a command in a sandbox.

        Args:
            handle: Resource handle from acquire()
            command: Command to execute
            working_dir: Optional working directory
            timeout: Optional timeout in seconds

        Returns:
            Command output (stdout + stderr)
        """
        resource = handle.resource
        timeout = timeout or self.command_timeout_seconds

        # Wait for ready if needed
        if not resource.sandbox_state["ready"]:
            await self._wait_for_ready(resource)

        start = time.time()
        self.logger.debug(f"Executing command in {resource.sandbox_id}: {command[:50]}...")

        try:
            from prime_sandboxes import CommandTimeoutError

            results = await self.client.execute_command(
                resource.sandbox_id,
                command,
                working_dir=working_dir or resource.working_dir,
                timeout=timeout,
            )
        except CommandTimeoutError:
            timeout_msg = f"Command timed out after {timeout}s"
            self.logger.warning(f"{timeout_msg} in sandbox {resource.sandbox_id}")
            resource.sandbox_state["command_execution_times"].append(float(timeout))
            return f"Error: {timeout_msg}"
        except Exception as e:
            self.logger.error(f"Command failed in {resource.sandbox_id}: {e}")
            raise

        execution_time = time.time() - start
        resource.sandbox_state["command_execution_times"].append(execution_time)

        # Combine stdout and stderr
        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        output = stdout
        if stderr:
            output = f"{output}\nstderr:\n{stderr}" if output else f"stderr:\n{stderr}"

        self.logger.debug(
            f"Command completed in {execution_time:.1f}s: {output[:100]}..."
        )
        return output or "(no output)"

    async def upload_file(
        self,
        handle: ResourceHandle[SandboxResource],
        local_path: str,
        remote_path: str,
    ) -> None:
        """Upload a file to the sandbox."""
        resource = handle.resource

        if not resource.sandbox_state["ready"]:
            await self._wait_for_ready(resource)

        await self.client.upload_file(resource.sandbox_id, local_path, remote_path)
        self.logger.debug(f"Uploaded {local_path} to {remote_path}")

    async def download_file(
        self,
        handle: ResourceHandle[SandboxResource],
        remote_path: str,
        local_path: str,
    ) -> None:
        """Download a file from the sandbox."""
        resource = handle.resource

        if not resource.sandbox_state["ready"]:
            await self._wait_for_ready(resource)

        await self.client.download_file(resource.sandbox_id, remote_path, local_path)
        self.logger.debug(f"Downloaded {remote_path} to {local_path}")

    def get_config_for_state(self, state: dict[str, Any]) -> SandboxConfig:
        """
        Get sandbox config for a specific state.

        Override this method to customize sandbox config per-rollout.

        Args:
            state: The rollout state

        Returns:
            SandboxConfig to use for this rollout
        """
        return self.config


def create_sandbox_manager(
    mode: str | AllocationMode = AllocationMode.ONE_TO_ONE,
    docker_image: str = "python:3.11-slim",
    start_command: str = "tail -f /dev/null",
    cpu_cores: int = 1,
    memory_gb: int = 2,
    disk_size_gb: int = 5,
    gpu_count: int = 0,
    timeout_minutes: int = 60,
    pool_size: int = 10,
    **kwargs,
) -> SandboxResourceManager:
    """
    Factory function to create a SandboxResourceManager.

    Args:
        mode: Allocation mode ("one_to_one", "pool", "shared", or AllocationMode enum)
        docker_image: Docker image for sandboxes
        start_command: Start command for containers
        cpu_cores: CPU cores per sandbox
        memory_gb: Memory per sandbox
        disk_size_gb: Disk size per sandbox
        gpu_count: GPUs per sandbox
        timeout_minutes: Sandbox timeout
        pool_size: Number of sandboxes in pool (for POOL mode)
        **kwargs: Additional arguments passed to SandboxResourceManager

    Returns:
        Configured SandboxResourceManager
    """
    if isinstance(mode, str):
        mode = AllocationMode(mode)

    config = SandboxConfig(
        docker_image=docker_image,
        start_command=start_command,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        gpu_count=gpu_count,
        timeout_minutes=timeout_minutes,
    )

    return SandboxResourceManager(
        config=config,
        mode=mode,
        pool_size=pool_size,
        **kwargs,
    )
