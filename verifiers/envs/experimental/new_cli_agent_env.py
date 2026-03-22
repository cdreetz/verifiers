"""CliAgentEnv using centralized resource management."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from prime_sandboxes import (
    AdvancedConfigs,
    CreateSandboxRequest,
)
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.envs.experimental.resource_managers.sandbox_manager import BackgroundJob, ManagedSandbox, SandboxManager
from verifiers.envs.experimental.resource_managers.base import ResourceState
from verifiers.envs.experimental.resource_managers.retry import RetryConfig
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    MessageType,
    ModelResponse,
    SamplingArgs,
    State,
)

logger = logging.getLogger(__name__)


class NewCliAgentEnv(vf.MultiTurnEnv):
    """
    Environment for running full agent code inside sandboxes.

    This is an improved version of CliAgentEnv that uses SandboxManager for:
    - Atomic resource tracking (no sandbox leakage)
    - Better error tracking (errors associated with rollouts)
    - Lifecycle observability

    The API is designed to be compatible with CliAgentEnv for easy migration.
    """

    def __init__(
        self,
        run_command: str,
        interception_port: int = 8765,
        interception_url: str | None = None,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 2.0,
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
        labels: list[str] | None = None,
        # Resource manager options
        max_retries: int = 5,
        retry_delay: float = 0.5,
        enable_health_monitoring: bool = False,
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.interception_port = interception_port
        self.interception_url = interception_url
        self.tunnel: Tunnel | None = None
        self.tunnel_lock = asyncio.Lock()
        self.timeout_seconds = timeout_seconds
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
        self.labels = labels
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}
        self.interception_server: Any = None
        self.server_lock = asyncio.Lock()
        self.server_runner: Any = None
        self.server_site: Any = None

        # Create default sandbox request
        self._default_request = CreateSandboxRequest(
            name="cli-agent",
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
            labels=labels if labels else [],
        )

        # Create the sandbox manager with proper RetryConfig
        retry_config = RetryConfig(
            max_attempts=max_retries,
            initial_delay=retry_delay,
        )
        self.sandbox_manager = SandboxManager(
            default_request=self._default_request,
            timeout_per_command=300,  # Longer timeout for agent commands
            retry_config=retry_config,
            enable_health_monitoring=enable_health_monitoring,
        )

    # -------------------------------------------------------------------------
    # Compatibility properties
    # -------------------------------------------------------------------------

    @property
    def active_sandboxes(self) -> set[str]:
        """Set of active sandbox IDs (for compatibility)."""
        return {r.id for r in self.sandbox_manager.get_active_resources()}

    @property
    def sandbox_client(self):
        """Access to the underlying sandbox client (for compatibility)."""
        return self.sandbox_manager.client

    def get_sandbox_request(self, state: State) -> CreateSandboxRequest:
        """Return sandbox request for this rollout. Override to customize per-state."""
        return self._default_request.model_copy()

    async def get_tunnel_url(self) -> str:
        """Get tunnel URL, starting the tunnel if needed."""
        async with self.tunnel_lock:
            if self.tunnel is None:
                if logger.isEnabledFor(logging.DEBUG):
                    self.tunnel = Tunnel(
                        local_port=self.interception_port,
                        log_level="debug",
                    )
                else:
                    self.tunnel = Tunnel(local_port=self.interception_port)
                url = await self.tunnel.start()
                logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self.tunnel.url is not None, "Tunnel started but URL is None"
                return self.tunnel.url

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout using SandboxManager."""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id
        state["trajectory_id"] = rollout_id  # For error tracking

        await self.ensure_interception_server()

        # Auto-start Prime Tunnel if no interception URL provided
        if self.interception_url is None:
            tunnel_url = await self.get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = (
                f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)

        # Build custom request for this rollout
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
            labels=self.labels if self.labels else [],
        )

        logger.debug(
            f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
            f"docker_image={docker_image}"
        )

        # Acquire sandbox through manager
        managed_sandbox = await self.sandbox_manager.acquire(
            rollout_id=rollout_id,
            request=sandbox_request,
        )
        state["sandbox_id"] = managed_sandbox.id
        state["managed_sandbox"] = managed_sandbox
        logger.debug(f"Created sandbox {managed_sandbox.id}")

        # Wait for sandbox to be ready
        await self.sandbox_manager.wait_for_ready(managed_sandbox.id)

        await self.post_sandbox_setup(state)

        request_id_queue: asyncio.Queue = asyncio.Queue()
        state["request_id_queue"] = request_id_queue
        state["agent_completed"] = False
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_id_queue,
        }

        await self.start_agent(state)

        return state

    async def get_docker_image(self, state: State) -> str:
        """Get the Docker image for the sandbox. Override for per-task images."""
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "600")
        env_vars.setdefault("HTTPX_TIMEOUT", "600")
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Hook for post-sandbox setup. Override to upload files, run commands, etc.

        Use self.sandbox_manager.execute_command() for commands (with retry and recording)
        and self.sandbox_manager.client for other operations like upload_file.
        """
        pass

    async def start_agent(self, state: State) -> None:
        """Start the agent command using background job via sandbox manager."""
        sandbox_id = state["sandbox_id"]

        background_job = await self.sandbox_manager.start_background_job(
            sandbox_id,
            self.run_command,
        )
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()

        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(state)
        )

    async def wait_for_completion(self, state: State) -> None:
        """Poll for agent completion using sandbox manager's background job API."""
        sandbox_id = state.get("sandbox_id")
        background_job: BackgroundJob | None = state.get("background_job")

        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(state, sandbox_id, background_job),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent timed out after {self.timeout_seconds}s")
            state["agent_timed_out"] = True
        except asyncio.CancelledError:
            logger.debug("Completion wait task cancelled")
            raise
        except Exception as e:
            logger.debug(f"Completion wait ended: {e}")
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(
        self, state: State, sandbox_id: str, background_job: BackgroundJob
    ) -> None:
        """Poll until background job completes via sandbox manager, capturing output."""
        while True:
            background_job = await self.sandbox_manager.poll_background_job(
                sandbox_id, background_job
            )
            if background_job.completed:
                state["agent_exit_code"] = background_job.exit_code
                state["agent_stdout"] = background_job.stdout
                state["agent_stderr"] = background_job.stderr
                logger.debug(f"Agent completed with exit_code={background_job.exit_code}")
                return
            await asyncio.sleep(1)

    async def check_agent_completed(self, state: State) -> bool:
        """Check if agent process has completed."""
        return state.get("agent_completed", False)

    async def get_prompt_messages(self, state: State) -> Messages:
        """Wait for agent to make an API request OR agent completion."""
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
            model = state.get("model") or model
            oai_tools = intercept.get("tools") or oai_tools

        response: ModelResponse | None = None
        error: BaseException | None = None

        try:
            if intercept and intercept.get("stream"):
                response = await self._get_streaming_model_response(
                    state=state,
                    prompt=prompt,
                    intercept=intercept,
                    client=client,
                    model=model,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args,
                )
            else:
                response = await super().get_model_response(
                    state=state,
                    prompt=prompt,
                    client=client,
                    model=model,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args,
                    message_type=message_type,
                )
        except BaseException as e:
            error = e
            raise
        finally:
            if intercept:
                future = intercept.get("response_future")
                if future and not future.done():
                    if error is not None:
                        future.set_exception(error)
                    elif response is not None:
                        future.set_result(response)
                state["current_request_id"] = None

        return response

    async def _get_streaming_model_response(
        self,
        state: State,
        prompt: Messages,
        intercept: dict,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> ChatCompletion:
        """Handle streaming API call, forwarding chunks and accumulating response."""
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])

        client = client or state["client"]
        model = model or state["model"]
        sampling_args = sampling_args or state.get("sampling_args") or {}

        if "max_tokens" in sampling_args:
            sampling_args = dict(sampling_args)
            max_tokens = sampling_args.pop("max_tokens")
            if "max_completion_tokens" not in sampling_args:
                sampling_args["max_completion_tokens"] = max_tokens

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": prompt,
            "stream": True,
        }
        if oai_tools:
            create_kwargs["tools"] = oai_tools
        create_kwargs.update(sampling_args)

        stream = await client.chat.completions.create(**create_kwargs)

        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}
        finish_reason = None
        completion_id = None
        created_time = int(time.time())
        stream_ended = False

        try:
            async for chunk in stream:
                await chunk_queue.put(chunk)

                if not completion_id and chunk.id:
                    completion_id = chunk.id
                if chunk.created:
                    created_time = chunk.created

                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    delta = choice.delta
                    if delta:
                        if delta.content:
                            accumulated_content += delta.content

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in accumulated_tool_calls:
                                    accumulated_tool_calls[idx] = {
                                        "id": tc.id or "",
                                        "type": tc.type or "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.id:
                                    accumulated_tool_calls[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        accumulated_tool_calls[idx]["function"][
                                            "name"
                                        ] = tc.function.name
                                    if tc.function.arguments:
                                        accumulated_tool_calls[idx]["function"][
                                            "arguments"
                                        ] += tc.function.arguments

            await chunk_queue.put(None)
            stream_ended = True
        finally:
            if not stream_ended:
                try:
                    chunk_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        tool_calls_list = None
        if accumulated_tool_calls:
            tool_calls_list = [
                ChatCompletionMessageToolCall(
                    id=tc_data["id"],
                    type="function",
                    function=Function(
                        name=tc_data["function"]["name"],
                        arguments=tc_data["function"]["arguments"],
                    ),
                )
                for idx, tc_data in sorted(accumulated_tool_calls.items())
            ]

        message = ChatCompletionMessage(
            role="assistant",
            content=accumulated_content if accumulated_content else None,
            tool_calls=tool_calls_list,
        )

        result = ChatCompletion(
            id=completion_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            choices=[
                Choice(
                    finish_reason=finish_reason or "stop",
                    index=0,
                    message=message,
                )
            ],
            created=created_time,
            model=model,
            object="chat.completion",
        )

        rollout_id = intercept.get("rollout_id", "?")
        self.log_response(rollout_id, result.model_dump())

        return result

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

            app = web.Application()
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self.handle_intercepted_request,
            )
            app.router.add_get(
                "/health",
                lambda _: web.json_response({"status": "ok"}),
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)
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
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        self.log_request(rollout_id, request_body)

        is_streaming = request_body.get("stream", False)
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        chunk_queue: asyncio.Queue | None = asyncio.Queue() if is_streaming else None

        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "stream": is_streaming,
            "chunk_queue": chunk_queue,
            "response_future": asyncio.Future(),
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        if is_streaming:
            return await self._handle_streaming_response(request, rollout_id, intercept)
        else:
            try:
                response_future = cast(
                    asyncio.Future[Any], intercept["response_future"]
                )
                response = await response_future
            except asyncio.CancelledError:
                return web.json_response({"error": "Rollout cancelled"}, status=499)
            except Exception as e:
                logger.error(f"Error processing intercepted request: {e}")
                return web.json_response({"error": str(e)}, status=500)

            response_dict = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )

            self.log_response(rollout_id, response_dict)
            return web.json_response(response_dict)

    async def _handle_streaming_response(
        self, http_request: Any, rollout_id: str, intercept: dict
    ) -> Any:
        """Handle streaming SSE response to the agent."""
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])
        response_future = cast(asyncio.Future[Any], intercept["response_future"])

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(http_request)

        try:
            while True:
                chunk = await chunk_queue.get()

                if chunk is None:
                    await response.write(b"data: [DONE]\n\n")
                    break

                chunk_dict = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                )
                chunk_json = json.dumps(chunk_dict)
                await response.write(f"data: {chunk_json}\n\n".encode())

            await response_future

        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Streaming cancelled")
        except ClientConnectionResetError:
            # Client disconnected before streaming completed - this is expected
            logger.debug(f"[{rollout_id}] Client disconnected during streaming")
        except Exception as e:
            logger.error(f"[{rollout_id}] Streaming error: {e}")

        try:
            await response.write_eof()
        except Exception as e:
            # Client may have disconnected before we could finish writing
            # This is expected behavior when the agent closes the connection early
            logger.debug(f"[{rollout_id}] Could not write EOF (client disconnected): {e}")
        return response

    @vf.teardown
    async def teardown_tunnel(self):
        """Stop Prime Tunnel and HTTP interception server."""
        async with self.tunnel_lock:
            if self.tunnel is not None:
                try:
                    await self.tunnel.stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self.tunnel = None

        async with self.server_lock:
            if self.server_runner is not None:
                try:
                    await self.server_runner.cleanup()
                    logger.debug("Stopped HTTP interception server")
                except RuntimeError as e:
                    if "Event loop is closed" not in str(e):
                        raise
                    logger.debug("HTTP server cleanup skipped (event loop closed)")
                finally:
                    self.server_runner = None
                    self.server_site = None
                    self.interception_server = None

    @vf.teardown
    async def teardown_sandboxes(self):
        """Release all active sandboxes through the manager."""
        # Print summary before cleanup
        self.print_sandbox_summary()
        await self.sandbox_manager.release_all()

    @vf.teardown
    async def teardown_sandbox_client(self):
        """Teardown the sandbox manager's client."""
        self.sandbox_manager.teardown()

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

        state.pop("background_job", None)

        rollout_id = state.get("rollout_id")
        if rollout_id:
            for request_id in list(self.intercepts.keys()):
                intercept = self.intercepts.get(request_id)
                if intercept and intercept.get("rollout_id") == rollout_id:
                    chunk_queue = intercept.get("chunk_queue")
                    if chunk_queue is not None:
                        try:
                            chunk_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass
                    future = intercept.get("response_future")
                    if future and not future.done():
                        future.cancel()
                    del self.intercepts[request_id]

            if rollout_id in self.active_rollouts:
                del self.active_rollouts[rollout_id]

    @vf.cleanup
    async def destroy_sandbox(self, state: State):
        """Release sandbox through the manager."""
        await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                await self.sandbox_manager.release(sandbox_id)
                logger.debug(f"Released sandbox {sandbox_id}")
            except Exception as e:
                logger.warning(f"Failed to release sandbox {sandbox_id}: {e}")

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
        """Override for custom post-rollout logic."""
        pass

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Generate a response from the environment."""
        return []

    # -------------------------------------------------------------------------
    # Error Tracking and Filtering Methods
    # -------------------------------------------------------------------------

    def get_sandbox_errors_for_rollout(self, rollout_id: str) -> list:
        """Get all sandbox errors associated with a specific rollout."""
        return self.sandbox_manager.get_errors_for_rollout(rollout_id)

    def get_sandbox(self, sandbox_id: str) -> ManagedSandbox | None:
        """Get the managed sandbox object by ID."""
        resource = self.sandbox_manager.get_resource(sandbox_id)
        if resource is None:
            return None
        return resource  # type: ignore

    def is_sandbox_healthy(self, sandbox_id: str) -> bool:
        """Check if a sandbox is in a healthy state."""
        sandbox = self.get_sandbox(sandbox_id)
        if sandbox is None:
            return False
        return sandbox.state == ResourceState.READY

    def get_all_sandbox_errors(self) -> list:
        """Get all sandbox errors across all rollouts."""
        return self.sandbox_manager.get_all_errors()

    def get_rollouts_with_errors(self) -> set[str]:
        """Get set of rollout IDs that had sandbox errors."""
        errors = self.sandbox_manager.get_all_errors()
        return {e.rollout_id for e in errors if e.rollout_id is not None}

    def get_sandbox_summary(self) -> dict[str, Any]:
        """Get summary of sandbox lifecycle metrics."""
        return self.sandbox_manager.get_summary()

    def print_sandbox_summary(self) -> None:
        """Print a summary of sandbox lifecycle metrics."""
        self.sandbox_manager.print_summary(title="SANDBOX LIFECYCLE SUMMARY")

    def filter_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Filter evaluation results to exclude rollouts with infrastructure errors."""
        if "outputs" not in results:
            return results

        error_rollouts = self.get_rollouts_with_errors()

        if not error_rollouts:
            results["filtered"] = {
                "total_rollouts": len(results["outputs"]),
                "excluded_rollouts": 0,
                "excluded_rollout_ids": [],
                "clean_rollouts": len(results["outputs"]),
            }
            return results

        original_outputs = results["outputs"]
        filtered_outputs = []
        excluded_ids = []

        for output in original_outputs:
            state = output.get("state", {})
            rollout_id = state.get("trajectory_id") or state.get("rollout_id")

            if rollout_id in error_rollouts:
                excluded_ids.append(rollout_id)
                logger.info(
                    f"Filtering out rollout {rollout_id} due to infrastructure error"
                )
            else:
                filtered_outputs.append(output)

        filtered_results = dict(results)
        filtered_results["outputs"] = filtered_outputs

        if filtered_outputs:
            rewards = [
                o.get("state", {}).get("reward", 0.0)
                for o in filtered_outputs
                if o.get("state", {}).get("reward") is not None
            ]
            if rewards:
                filtered_results["mean_reward"] = sum(rewards) / len(rewards)

        filtered_results["filtered"] = {
            "total_rollouts": len(original_outputs),
            "excluded_rollouts": len(excluded_ids),
            "excluded_rollout_ids": excluded_ids,
            "clean_rollouts": len(filtered_outputs),
            "errors": [
                {
                    "rollout_id": e.rollout_id,
                    "phase": e.phase,
                    "error": str(e.error),
                }
                for e in self.sandbox_manager.get_all_errors()
            ],
        }

        logger.info(
            f"Filtered results: {len(excluded_ids)} rollouts excluded due to "
            f"infrastructure errors, {len(filtered_outputs)} clean rollouts remain"
        )

        return filtered_results
