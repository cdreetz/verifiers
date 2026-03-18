import asyncio
import functools
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from prime_sandboxes import AsyncSandboxClient, CommandTimeoutError, CreateSandboxRequest, SandboxClient
from prime_sandboxes.core import APIClient

from verifiers.envs.experimental.managers.resource_manager import ManagedResource, ResourceManager
from verifiers.utils.thread_utils import get_or_create_thread_attr, get_or_create_thread_loop
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class ThreadedAsyncSandboxClient:
    """Wraps AsyncSandboxClient to run in thread pool for better concurrency."""

    def __init__(self, max_workers: int = 100, max_connections: int = 100):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sandbox-")
        self.client_kwargs = {"max_connections": max_connections}

    def __getattr__(self, name: str) -> Any:
        @functools.wraps(getattr(AsyncSandboxClient, name, lambda: None))
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            def run():
                loop = get_or_create_thread_loop()
                client = get_or_create_thread_attr("sandbox_client", AsyncSandboxClient, **self.client_kwargs)
                return loop.run_until_complete(getattr(client, name)(*args, **kwargs))
            return await asyncio.get_event_loop().run_in_executor(self.executor, run)
        return wrapper

    def teardown(self) -> None:
        self.executor.shutdown(wait=True)


@dataclass
class ManagedSandbox(ManagedResource):
    """Sandbox with execution timing metrics."""
    command_times: list[float] = field(default_factory=list)
    ready_wait_time: float = 0.0


class SandboxManager(ResourceManager):
    """Manages sandbox lifecycle with proper tracking and cleanup."""

    def __init__(
        self,
        default_request: CreateSandboxRequest | None = None,
        timeout_per_command: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay, **kwargs)
        self.default_request = default_request
        self.timeout_per_command = timeout_per_command
        self.client = ThreadedAsyncSandboxClient()

    def create_resource_object(self, rollout_id: str | None) -> ManagedSandbox:
        return ManagedSandbox(id=f"sandbox-pending-{uuid.uuid4().hex[:8]}", rollout_id=rollout_id)

    async def create_resource(self, resource: ManagedResource, **kwargs: Any) -> None:
        request = kwargs.get("request", self.default_request)
        if request is None:
            raise ValueError("No sandbox request provided")
        result = await self.client.create(request)
        resource.id = result.id

    async def destroy_resource(self, resource_id: str) -> None:
        if getattr(sys, "is_finalizing", lambda: False)():
            return
        if getattr(self.client.executor, "_shutdown", False):
            return
        try:
            await self.client.delete(resource_id)
        except RuntimeError as e:
            if "interpreter shutdown" in str(e) or "Event loop is closed" in str(e):
                return
            raise

    async def check_health(self, resource_id: str) -> bool:
        try:
            result = await self.client.execute_command(resource_id, "echo ok", timeout=10)
            return result.stdout.strip() == "ok"
        except Exception:
            return False

    async def wait_for_ready(self, sandbox_id: str) -> None:
        """Wait for sandbox to be ready."""
        sandbox = cast(ManagedSandbox, self.get_resource(sandbox_id))
        if sandbox is None:
            raise ValueError(f"Unknown sandbox: {sandbox_id}")
        start = time.time()
        await self.client.wait_for_creation(sandbox_id)
        sandbox.ready_wait_time = time.time() - start

    async def execute_command(self, sandbox_id: str, command: str, working_dir: str | None = None, timeout: int | None = None) -> str:
        """Execute command in sandbox."""
        sandbox = cast(ManagedSandbox, self.get_resource(sandbox_id))
        timeout = timeout or self.timeout_per_command
        start = time.time()

        try:
            result = await self.client.execute_command(sandbox_id, command, working_dir=working_dir, timeout=timeout)
        except CommandTimeoutError:
            if sandbox:
                sandbox.command_times.append(timeout)
            return f"Error: Command timed out after {timeout}s"

        if sandbox:
            sandbox.command_times.append(time.time() - start)

        stdout = result.stdout.strip()
        stderr = (result.stderr or "").strip()
        if stderr:
            return f"{stdout}\nstderr:\n{stderr}" if stdout else f"stderr:\n{stderr}"
        return stdout or "(no output)"

    async def release_all(self) -> None:
        """Release all sandboxes using bulk delete."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            self.health_monitor_task = None

        active_ids = [r.id for r in self.resources.values() if r.is_active]
        if not active_ids:
            return

        self.logger.info(f"Releasing {len(active_ids)} sandboxes")
        try:
            sync_client = SandboxClient(APIClient())
            for i in range(0, len(active_ids), 100):
                batch = active_ids[i:i+100]
                try:
                    sync_client.bulk_delete(sandbox_ids=batch)
                    for sid in batch:
                        if s := self.resources.get(sid):
                            s.mark_destroyed()
                except Exception as e:
                    self.logger.warning(f"Bulk delete failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to release sandboxes: {e}")

    def teardown(self) -> None:
        """Shutdown the client thread pool."""
        self.client.teardown()

    def get_summary(self) -> dict[str, Any]:
        """Get sandbox lifecycle metrics."""
        state_counts: dict[str, int] = {}
        for r in self.resources.values():
            state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1

        error_phases: dict[str, int] = {}
        for e in self.errors:
            error_phases[e.phase] = error_phases.get(e.phase, 0) + 1

        error_rollouts = {e.rollout_id for e in self.errors if e.rollout_id}

        return {
            "total_sandboxes": len(self.resources),
            "state_counts": state_counts,
            "total_errors": len(self.errors),
            "errors_by_phase": error_phases,
            "rollouts_with_errors": len(error_rollouts),
        }

    def print_summary(self, title: str = "SANDBOX SUMMARY") -> None:
        """Print sandbox lifecycle summary."""
        summary = self.get_summary()
        if summary["total_sandboxes"] == 0:
            return

        print(f"\n{'=' * 60}")
        print(title)
        print('=' * 60)
        print(f"Total sandboxes: {summary['total_sandboxes']}")

        if summary["state_counts"]:
            print("\nStates:")
            for state, count in sorted(summary["state_counts"].items()):
                print(f"  {state}: {count}")

        if summary["total_errors"] > 0:
            print(f"\nErrors: {summary['total_errors']}")
            for phase, count in sorted(summary["errors_by_phase"].items()):
                print(f"  {phase}: {count}")
            if summary["rollouts_with_errors"]:
                print(f"Affected rollouts: {summary['rollouts_with_errors']}")
        else:
            print("\nNo errors")

        print('=' * 60 + "\n")
