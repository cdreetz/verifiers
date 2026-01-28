"""
CLI Agent Environment

Runs full agent code inside sandboxes, intercepting agent API requests
via HTTP proxy server. Each agent request triggers one rollout step.

Supports both legacy mode (direct sandbox management) and new resource manager mode.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shlex
import shutil
import subprocess
import time
import uuid
from typing import Any, cast

from aiohttp import web
from openai import AsyncOpenAI
from prime_sandboxes import (
    AdvancedConfigs,
    AsyncSandboxClient,
    CreateSandboxRequest,
)

import verifiers as vf
from verifiers.resources import (
    AllocationMode,
    ResourceHandle,
    SandboxConfig,
    SandboxResource,
    SandboxResourceManager,
)
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    MessageType,
    ModelResponse,
    SamplingArgs,
    State,
)

logger = logging.getLogger(__name__)


class CliAgentEnv(vf.MultiTurnEnv):
    """
    Environment for running full agent code inside sandboxes.

    Extends MultiTurnEnv to reuse rollout loop, but intercepts agent's
    API requests via HTTP proxy server. Each agent request triggers one
    rollout step.

    Supports two modes:
    1. Legacy mode (default): Direct sandbox management with one sandbox per rollout
    2. Resource manager mode: Use SandboxResourceManager for advanced allocation modes

    Example (legacy mode):
        env = CliAgentEnv(
            run_command="python agent.py",
            docker_image="python:3.11",
        )

    Example (resource manager mode):
        manager = SandboxResourceManager(
            config=SandboxConfig(docker_image="python:3.11"),
            mode=AllocationMode.POOL,
            pool_size=10,
        )
        env = CliAgentEnv(
            run_command="python agent.py",
            resource_manager=manager,
        )
    """

    def __init__(
        self,
        run_command: str,
        # Resource manager mode
        resource_manager: SandboxResourceManager | None = None,
        # Interception configuration
        interception_port: int = 8765,
        interception_url: str | None = None,
        # Rollout configuration
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 2.0,
        # Legacy sandbox configuration (ignored if resource_manager is provided)
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.interception_port = interception_port
        self.interception_url = interception_url
        self.timeout_seconds = timeout_seconds

        # Resource manager mode
        self._resource_manager = resource_manager
        self._use_resource_manager = resource_manager is not None

        # Legacy sandbox configuration
        self.docker_image = docker_image
        self.start_command = start_command
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_minutes = timeout_minutes
        self.environment_vars = environment_vars
        self.team_id = team_id
        self.advanced_configs = advanced_configs

        # Tunnel management
        self.tunnels: list[dict[str, Any]] = []
        self.tunnel_lock = asyncio.Lock()
        self.tunnel_round_robin_index = 0

        # Interception state
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}
        self.interception_server: Any = None
        self.server_lock = asyncio.Lock()
        self.server_runner: Any = None
        self.server_site: Any = None

    @property
    def resource_manager(self) -> SandboxResourceManager | None:
        """Get the resource manager if in resource manager mode."""
        return self._resource_manager

    def ensure_cloudflared_installed(self) -> str:
        """Install cloudflared if not already installed. Returns path to cloudflared binary."""
        path = shutil.which("cloudflared")
        if path:
            return path

        logger.info("Installing cloudflared...")
        system = platform.system()

        if system == "Darwin":
            cmd = ["brew", "install", "cloudflare/cloudflare/cloudflared"]
        elif system == "Linux":
            script = (
                "curl -L --output cloudflared.deb "
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb "
                "&& sudo dpkg -i cloudflared.deb && rm cloudflared.deb"
            )
            cmd = ["bash", "-c", script]
        else:
            raise RuntimeError(
                f"Unsupported platform: {system}. "
                "Please install cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
            )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install cloudflared: {result.stderr}")

        path = shutil.which("cloudflared")
        if not path:
            raise RuntimeError("cloudflared installed but not found in PATH")
        return path

    def extract_tunnel_url_from_line(self, line: str) -> str | None:
        """Extract tunnel URL from a line of cloudflared output."""
        if ".trycloudflare.com" not in line:
            return None

        start_idx = line.find("https://")
        if start_idx == -1:
            return None

        url_start = start_idx
        url_end = url_start + 8
        while url_end < len(line) and not line[url_end].isspace():
            url_end += 1

        url = line[url_start:url_end].rstrip("/")
        if ".trycloudflare.com" in url:
            return url
        return None

    def start_cloudflared_tunnel(self) -> tuple[str, subprocess.Popen]:
        """Start cloudflared tunnel and return (URL, process)."""
        cloudflared_path = self.ensure_cloudflared_installed()

        tunnel_process = subprocess.Popen(
            [
                cloudflared_path,
                "tunnel",
                "--url",
                f"http://localhost:{self.interception_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stderr_lines = []
        max_wait_seconds = 30
        check_interval = 0.5
        max_iterations = int(max_wait_seconds / check_interval)

        for _ in range(max_iterations):
            if tunnel_process.poll() is not None:
                if tunnel_process.stderr:
                    remaining = tunnel_process.stderr.read()
                    stderr_lines.append(remaining)
                error_output = "".join(stderr_lines)
                raise RuntimeError(
                    f"cloudflared tunnel failed to start: {error_output}"
                )

            if tunnel_process.stderr:
                line = tunnel_process.stderr.readline()
                if line:
                    stderr_lines.append(line)
                    url = self.extract_tunnel_url_from_line(line)
                    if url:
                        logger.info(f"Cloudflare tunnel started: {url}")
                        return url, tunnel_process

            time.sleep(check_interval)

        all_output = "".join(stderr_lines)
        for line in stderr_lines:
            url = self.extract_tunnel_url_from_line(line)
            if url:
                logger.info(f"Cloudflare tunnel started: {url}")
                return url, tunnel_process

        raise RuntimeError(
            f"Failed to get tunnel URL from cloudflared after {max_wait_seconds} seconds. "
            f"Output: {all_output[:500]}"
        )

    async def get_tunnel_url(self) -> str:
        """Get tunnel URL from pool, creating new tunnels as needed (1 per 50 active rollouts)."""
        async with self.tunnel_lock:
            total_active_rollouts = len(self.active_rollouts)
            required_tunnels = max(1, (total_active_rollouts + 49) // 50)

            while len(self.tunnels) < required_tunnels:
                try:
                    url, process = self.start_cloudflared_tunnel()
                    self.tunnels.append(
                        {
                            "url": url,
                            "process": process,
                            "active_rollouts": 0,
                        }
                    )
                    logger.debug(
                        f"Created tunnel {len(self.tunnels)}/{required_tunnels}: {url}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create tunnel: {e}")
                    raise

            tunnel = self.tunnels[self.tunnel_round_robin_index % len(self.tunnels)]
            self.tunnel_round_robin_index += 1
            tunnel["active_rollouts"] += 1

            return tunnel["url"]

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout."""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        await self.ensure_interception_server()

        # Auto-start Cloudflare tunnel if not provided
        tunnel_url: str | None = None
        if self.interception_url is None:
            tunnel_url = await self.get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = (
                f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)

        if self._use_resource_manager:
            # Resource manager mode
            await self._resource_manager.startup()

            # Create per-rollout config with environment vars
            config = self._resource_manager.get_config_for_state(state)
            config.environment_vars = env_vars
            config.docker_image = docker_image

            # Override config for this rollout
            original_config = self._resource_manager.config
            self._resource_manager.config = config

            handle = await self._resource_manager.acquire(state)
            state["_resource_handle"] = handle
            state["sandbox_id"] = handle.resource_id

            # Restore original config
            self._resource_manager.config = original_config

            # Wait for ready and run post-setup
            await self._resource_manager._wait_for_ready(handle.resource)
            await self.post_sandbox_setup(state, self._resource_manager.client)
        else:
            # Legacy mode
            sandbox_client = AsyncSandboxClient()
            sandbox_request = CreateSandboxRequest(
                name=rollout_id,
                docker_image=docker_image,
                start_command=self.start_command,
                cpu_cores=self.cpu_cores,
                memory_gb=self.memory_gb,
                disk_size_gb=self.disk_size_gb,
                gpu_count=self.gpu_count,
                timeout_minutes=self.timeout_minutes,
                environment_vars=env_vars,
                team_id=self.team_id,
                advanced_configs=self.advanced_configs,
            )
            logger.debug(
                f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
                f"docker_image={docker_image}"
            )
            sandbox = await sandbox_client.create(sandbox_request)
            state["sandbox_id"] = sandbox.id
            logger.debug(f"Created sandbox {sandbox.id}")
            await sandbox_client.wait_for_creation(sandbox.id)

            await self.post_sandbox_setup(state, sandbox_client)

        request_id_queue: asyncio.Queue = asyncio.Queue()
        state["request_id_queue"] = request_id_queue
        state["tunnel_url"] = tunnel_url if self.interception_url is None else None
        state["agent_completed"] = False
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_id_queue,
        }

        # Start the agent
        if self._use_resource_manager:
            await self.start_agent(state, self._resource_manager.client)
        else:
            await self.start_agent(state, AsyncSandboxClient())

        return state

    async def get_docker_image(self, state: State) -> str:
        """Get the Docker image for the sandbox. Override for per-task images."""
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_sandbox_setup(
        self, state: State, sandbox_client: Any
    ) -> None:
        """Hook for post-sandbox setup. Override to upload files, run commands, etc."""
        pass

    async def start_agent(
        self, state: State, sandbox_client: Any
    ) -> None:
        """Start the agent command with automatic completion detection."""
        sandbox_id = state["sandbox_id"]

        wrapped_command = f"""
{self.run_command}
EXIT_CODE=$?
echo $EXIT_CODE > /tmp/vf_exit_code
touch /tmp/vf_complete
"""

        await sandbox_client.execute_command(
            sandbox_id,
            f"nohup bash -c {shlex.quote(wrapped_command)} "
            f"> /tmp/agent_stdout.log 2> /tmp/agent_stderr.log &",
        )

        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(state)
        )

    async def wait_for_completion(self, state: State) -> None:
        """Block until agent completion marker appears, then set state flag."""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        try:
            if self._use_resource_manager:
                sandbox_client = self._resource_manager.client
            else:
                sandbox_client = AsyncSandboxClient()

            await sandbox_client.execute_command(
                sandbox_id,
                "while ! test -f /tmp/vf_complete; do sleep 0.1; done",
                timeout=int(self.timeout_seconds),
            )
            state["agent_completed"] = True
        except Exception as e:
            logger.debug(f"Completion wait ended: {e}")
            state["agent_completed"] = True

    async def check_agent_completed(self, state: State) -> bool:
        """Check if agent process has completed."""
        return state.get("agent_completed", False)

    async def get_prompt_messages(self, state: State) -> Messages:
        """Wait for agent to make an API request OR agent completion, whichever comes first."""
        request_id_queue = state["request_id_queue"]

        while True:
            try:
                request_id = await asyncio.wait_for(
                    request_id_queue.get(),
                    timeout=self.poll_interval,
                )
                state["current_request_id"] = request_id
                intercept = self.intercepts[request_id]
                return intercept["messages"]

            except asyncio.TimeoutError:
                if await self.check_agent_completed(state):
                    state["agent_completed"] = True
                    return []
                if time.time() - state["timing"]["start_time"] > self.timeout_seconds:
                    return []

    async def get_model_response(
        self,
        state: State,
        prompt: Messages,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> ModelResponse:
        """Get model response and unblock the waiting HTTP handler."""
        if not prompt:
            from openai.types.chat import ChatCompletion, ChatCompletionMessage
            from openai.types.chat.chat_completion import Choice

            return ChatCompletion(
                id="agent-completed",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content=""),
                    )
                ],
                created=int(time.time()),
                model=model or state["model"],
                object="chat.completion",
            )

        request_id = state.get("current_request_id")
        intercept = self.intercepts.get(request_id) if request_id else None

        if intercept:
            model = intercept.get("model") or model
            oai_tools = intercept.get("tools") or oai_tools

        response = await super().get_model_response(
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            oai_tools=oai_tools,
            sampling_args=sampling_args,
            message_type=message_type,
        )

        if intercept:
            intercept["response_future"].set_result(response)
            state["current_request_id"] = None

        return response

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
        """Add model response and update top-level prompt on first turn."""
        if not prompt_messages:
            return
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await super().add_model_response(state, prompt_messages, response)

    def truncate(self, s: str, limit: int = 200) -> str:
        return (s[:limit] + "...") if len(s) > limit else s

    def log_request(self, rollout_id: str, body: dict) -> None:
        logger.debug(f"[{rollout_id}] <- INTERCEPTED REQUEST")
        for msg in body.get("messages", []):
            logger.debug(
                f"  [{msg.get('role', '?')}] {self.truncate(msg.get('content', ''))}"
            )
        if body.get("tools"):
            logger.debug(f"  [tools] {len(body['tools'])} tool(s)")

    def log_response(self, rollout_id: str, response: dict) -> None:
        logger.debug(f"[{rollout_id}] -> RESPONSE")
        msg = response.get("choices", [{}])[0].get("message", {})
        if msg.get("content"):
            logger.debug(f"  [assistant] {self.truncate(msg['content'])}")
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            logger.debug(
                f"  [tool_call] {func.get('name')}({self.truncate(func.get('arguments', ''), 100)})"
            )

    async def ensure_interception_server(self):
        """Start shared HTTP server if needed."""
        async with self.server_lock:
            if self.interception_server is not None:
                return

            app = web.Application()  # type: ignore
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self.handle_intercepted_request,
            )

            runner = web.AppRunner(app)  # type: ignore
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)  # type: ignore
            await site.start()

            self.interception_server = app
            self.server_runner = runner
            self.server_site = site

            logger.debug(
                f"Started interception server on port {self.interception_port}"
            )

    async def handle_intercepted_request(self, request: Any) -> Any:
        """HTTP handler: queue request, wait for response, return."""
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response(  # type: ignore
                {"error": "Rollout not found"}, status=404
            )

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response(  # type: ignore
                {"error": f"Invalid JSON: {e}"}, status=400
            )

        self.log_request(rollout_id, request_body)

        request_id = f"req_{uuid.uuid4().hex[:8]}"
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "response_future": asyncio.Future(),
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        try:
            response_future = cast(asyncio.Future[Any], intercept["response_future"])
            response = await response_future
        except asyncio.CancelledError:
            return web.json_response(  # type: ignore
                {"error": "Rollout cancelled"}, status=499
            )
        except Exception as e:
            logger.error(f"Error processing intercepted request: {e}")
            return web.json_response(  # type: ignore
                {"error": str(e)}, status=500
            )

        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )

        self.log_response(rollout_id, response_dict)
        return web.json_response(response_dict)  # type: ignore

    @vf.teardown
    async def teardown_tunnel(self):
        """Stop cloudflared tunnels and HTTP interception server."""
        async with self.tunnel_lock:
            for tunnel in self.tunnels:
                process = tunnel.get("process")
                if process:
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping cloudflared tunnel: {e}")
                        try:
                            process.kill()
                        except Exception:
                            pass
            self.tunnels.clear()

        async with self.server_lock:
            if self.server_runner is not None:
                await self.server_runner.cleanup()
                self.server_runner = None
                self.server_site = None
                self.interception_server = None
                logger.debug("Stopped HTTP interception server")

    @vf.teardown
    async def teardown_resource_manager(self):
        """Teardown the resource manager if in resource manager mode."""
        if self._use_resource_manager and self._resource_manager:
            await self._resource_manager.teardown()

    @vf.cleanup
    async def cleanup_interception_context(self, state: State):
        """Cleanup interception context for rollout."""
        task = state.get("completion_wait_task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        rollout_id = state.get("rollout_id")
        if rollout_id:
            for request_id in list(self.intercepts.keys()):
                intercept = self.intercepts.get(request_id)
                if intercept and intercept.get("rollout_id") == rollout_id:
                    future = intercept.get("response_future")
                    if future and not future.done():
                        future.cancel()
                    del self.intercepts[request_id]

            if rollout_id in self.active_rollouts:
                del self.active_rollouts[rollout_id]

        tunnel_url = state.get("tunnel_url")
        if tunnel_url:
            async with self.tunnel_lock:
                for tunnel in self.tunnels:
                    if tunnel["url"] == tunnel_url:
                        tunnel["active_rollouts"] = max(
                            0, tunnel["active_rollouts"] - 1
                        )
                        break

    @vf.stop
    async def agent_completed(self, state: State) -> bool:
        """Check if agent has completed."""
        return state.get("agent_completed", False)

    @vf.stop
    async def timeout_reached(self, state: State) -> bool:
        """Check rollout timeout."""
        elapsed = time.time() - state["timing"]["start_time"]
        return elapsed > self.timeout_seconds

    async def post_rollout(self, state: State):
        """
        Override for custom post-rollout logic. For example, if sandbox state is needed for reward functions,
        run computation here and cache the result in state before sandbox is destroyed.
        """
        pass

    @vf.cleanup
    async def destroy_sandbox(self, state: State):
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
        if sandbox_id:
            try:
                sandbox_client = AsyncSandboxClient()
                await sandbox_client.delete(sandbox_id)
                logger.debug(f"Deleted sandbox {sandbox_id}")
            except Exception as e:
                logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Generate a response from the environment.
        For CliAgentEnv, there is no environment response - the agent
        controls the conversation flow via its requests.
        """
        return []


def create_cli_agent_env(
    run_command: str,
    mode: str | AllocationMode = AllocationMode.ONE_TO_ONE,
    docker_image: str = "python:3.11-slim",
    start_command: str = "tail -f /dev/null",
    cpu_cores: int = 1,
    memory_gb: int = 2,
    pool_size: int = 10,
    **kwargs,
) -> CliAgentEnv:
    """
    Factory function to create a CliAgentEnv with resource manager.

    Args:
        run_command: Command to run the agent
        mode: Allocation mode ("one_to_one", "pool", "shared")
        docker_image: Docker image for sandboxes
        start_command: Start command for containers
        cpu_cores: CPU cores per sandbox
        memory_gb: Memory per sandbox
        pool_size: Number of sandboxes in pool (for POOL mode)
        **kwargs: Additional arguments passed to CliAgentEnv

    Returns:
        Configured CliAgentEnv
    """
    if isinstance(mode, str):
        mode = AllocationMode(mode)

    config = SandboxConfig(
        docker_image=docker_image,
        start_command=start_command,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
    )

    manager = SandboxResourceManager(
        config=config,
        mode=mode,
        pool_size=pool_size,
    )

    return CliAgentEnv(
        run_command=run_command,
        resource_manager=manager,
        **kwargs,
    )
