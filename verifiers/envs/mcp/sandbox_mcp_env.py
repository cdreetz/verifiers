import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import tenacity as tc

import verifiers as vf
from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.types import Message

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
        "prime-sandboxes is not installed. "
        "Please install it with `uv pip install prime-sandboxes`."
    )

from .models import SandboxMCPServerConfig
from .mcp_tool_wrapper import MCPToolWrapper
from .transports.sandbox_sse import SandboxSSETransport


class SandboxMCPEnv(StatefulToolEnv):
    """Environment that runs MCP servers inside isolated sandboxes.

    This environment combines the sandbox infrastructure from SandboxEnv
    with MCP server connectivity. Each rollout gets its own sandbox with
    an MCP server running inside, connected via SSE.

    The workflow is:
    1. Create a sandbox per rollout
    2. Run setup commands (install dependencies, etc.)
    3. Start the MCP server
    4. Expose the server port
    5. Connect via SSE transport
    6. Execute tool calls
    7. Cleanup sandbox after rollout
    """

    def __init__(
        self,
        mcp_server: SandboxMCPServerConfig,
        sandbox_name: str = "mcp-sandbox-env",
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        environment_vars: Optional[Dict[str, str]] = None,
        team_id: Optional[str] = None,
        advanced_configs: Optional[AdvancedConfigs] = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        dns_wait_seconds: float = 10.0,
        connection_retries: int = 10,
        connection_retry_delay: float = 3.0,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs,
    ):
        """Initialize the sandbox MCP environment.

        Args:
            mcp_server: Configuration for the MCP server to run in sandboxes.
            sandbox_name: Base name for created sandboxes.
            docker_image: Docker image to use for sandboxes.
            start_command: Command to run when sandbox starts.
            cpu_cores: Number of CPU cores for each sandbox.
            memory_gb: Memory allocation in GB.
            disk_size_gb: Disk size in GB.
            gpu_count: Number of GPUs (0 for none).
            timeout_minutes: Sandbox timeout in minutes.
            environment_vars: Environment variables for the sandbox.
            team_id: Optional team ID for sandbox billing.
            advanced_configs: Advanced sandbox configurations.
            max_retries: Max retries for sandbox operations.
            base_delay: Base delay for retry backoff.
            backoff_factor: Exponential backoff factor.
            max_backoff_seconds: Maximum backoff delay.
            jitter: Jitter for retry timing.
            dns_wait_seconds: Time to wait for DNS after port exposure.
            connection_retries: Number of SSE connection retry attempts.
            connection_retry_delay: Delay between connection retries.
            max_turns: Maximum conversation turns.
            error_formatter: Function to format tool errors.
            **kwargs: Additional arguments for StatefulToolEnv.
        """
        super().__init__(
            tools=[],
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs,
        )

        self.mcp_server_config = mcp_server
        self.sandbox_client = AsyncSandboxClient()
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
        self.dns_wait_seconds = dns_wait_seconds
        self.connection_retries = connection_retries
        self.connection_retry_delay = connection_retry_delay
        self.error_formatter = error_formatter

        # Per-rollout state
        self._transports: Dict[str, SandboxSSETransport] = {}

        # Retry wrapper for sandbox operations
        self.with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.ERROR),
            reraise=True,
        ).wraps

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create sandbox, setup MCP server, and connect.

        This method:
        1. Creates a new sandbox
        2. Waits for it to be ready
        3. Runs setup commands
        4. Starts the MCP server
        5. Exposes the port
        6. Waits for DNS propagation
        7. Connects via SSE transport
        8. Registers tools
        """
        # Create sandbox
        sandbox = await self.with_retry(self.sandbox_client.create)(
            self.sandbox_request
        )
        self.active_sandboxes.add(sandbox.id)
        state["sandbox_id"] = sandbox.id
        self.logger.debug(f"Created sandbox {sandbox.id}")

        # Wait for sandbox to be ready
        await self.sandbox_client.wait_for_creation(sandbox.id)
        self.logger.debug(f"Sandbox {sandbox.id} is ready")

        # Run setup commands
        for cmd in self.mcp_server_config.setup_commands:
            self.logger.debug(f"Running setup command: {cmd}")
            result = await self.sandbox_client.execute_command(sandbox.id, cmd)
            if result.stderr:
                self.logger.warning(f"Setup command stderr: {result.stderr}")

        # Build environment string for the server command
        env_str = ""
        if self.mcp_server_config.env:
            env_parts = [f"{k}={v}" for k, v in self.mcp_server_config.env.items()]
            env_str = " ".join(env_parts) + " "

        # Start MCP server in background
        start_cmd = (
            f"nohup {env_str}{self.mcp_server_config.start_command} "
            f"> /tmp/mcp_server.log 2>&1 &"
        )
        self.logger.debug(f"Starting MCP server: {start_cmd}")
        await self.sandbox_client.execute_command(sandbox.id, start_cmd)

        # Give server a moment to start
        await asyncio.sleep(2)

        # Expose the port
        port = self.mcp_server_config.port
        exposed = await self.sandbox_client.expose(sandbox.id, port)
        state["exposed_url"] = exposed.url
        state["mcp_port"] = port
        self.logger.info(f"Exposed port {port} at {exposed.url}")

        # Wait for DNS propagation
        self.logger.debug(f"Waiting {self.dns_wait_seconds}s for DNS propagation")
        await asyncio.sleep(self.dns_wait_seconds)

        # Connect via SSE transport
        transport = SandboxSSETransport(
            name=self.mcp_server_config.name,
            url=exposed.url,
            sandbox_id=sandbox.id,
            connection_retries=self.connection_retries,
            retry_delay=self.connection_retry_delay,
            logger=self.logger,
        )

        tools = await transport.connect()
        self._transports[sandbox.id] = transport
        state["mcp_transport"] = transport

        # Register tools for this rollout
        wrapper_tools = []
        for tool in tools.values():
            wrapper = MCPToolWrapper(
                self.mcp_server_config.name, tool, transport
            )
            wrapper_tools.append(wrapper)
            self.logger.info(f"Registered MCP tool: {wrapper.__name__}")

        # Update tool maps for this rollout
        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}

        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update tool arguments - no modifications needed for MCP tools."""
        return tool_args

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        """Call an MCP tool."""
        if tool_name in self.tool_map:
            tool_wrapper = self.tool_map[tool_name]
            try:
                result = await tool_wrapper(**tool_args)
                return {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call_id,
                }
            except Exception as e:
                return {
                    "role": "tool",
                    "content": self.error_formatter(e),
                    "tool_call_id": tool_call_id,
                }
        else:
            return {
                "role": "tool",
                "content": f"Error: Tool '{tool_name}' not found",
                "tool_call_id": tool_call_id,
            }

    async def post_rollout(self, state: vf.State):
        """Override for custom post-rollout logic.

        Called before sandbox destruction. Use this to extract
        any final state from the sandbox for reward computation.
        """
        pass

    @vf.cleanup
    async def destroy_sandbox(self, state: vf.State):
        """Clean up sandbox and disconnect transport after rollout."""
        await self.post_rollout(state)

        sandbox_id = state.get("sandbox_id")
        if sandbox_id is None:
            return

        # Disconnect transport
        transport = self._transports.pop(sandbox_id, None)
        if transport:
            try:
                await transport.disconnect()
            except Exception as e:
                self.logger.warning(f"Error disconnecting transport: {e}")

        # Delete sandbox
        async def _delete_sandbox(sid: str):
            await self.sandbox_client.delete(sid)
            self.active_sandboxes.discard(sid)
            self.logger.debug(f"Deleted sandbox {sid}")

        try:
            await self.with_retry(_delete_sandbox)(sandbox_id)
        except Exception as e:
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def bulk_delete_sandboxes(self, global_ids: List[str]) -> None:
        """Delete multiple sandboxes by their global IDs."""
        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(global_ids)
            self.logger.debug(f"Bulk deleted sandboxes: {global_ids}")
            self.active_sandboxes.difference_update(global_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {global_ids}: {e}")

    @vf.teardown
    async def teardown_sandboxes(self):
        """Delete all active sandboxes using sync client.

        Uses the synchronous SandboxClient for teardown to avoid event loop issues
        during signal handling and interpreter shutdown.
        """
        # Disconnect all transports
        for transport in self._transports.values():
            try:
                await transport.disconnect()
            except Exception:
                pass
        self._transports.clear()

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
