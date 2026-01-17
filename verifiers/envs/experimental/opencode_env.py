import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web

from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.types import (
    Messages,
    State,
)

logger = logging.getLogger(__name__)


class OpenCodeEnv(CliAgentEnv):
    """
    Environment for running OpenCode (or similar agents using OpenAI Responses API)
    inside sandboxes. Extends CliAgentEnv to intercept Responses API requests
    with streaming support.

    The Responses API differs from Chat Completions:
    - Endpoint: /v1/responses
    - Request format: {input, instructions, model, stream, tools, ...}
    - Streaming: SSE events like response.created, response.output_text.delta,
      response.function_call_arguments.delta, response.completed, etc.
    """

    def __init__(
        self,
        run_command: str = "opencode",
        opencode_version: str | None = None,
        **kwargs,
    ):
        """
        Initialize OpenCodeEnv.

        Args:
            run_command: Command to run the agent (default: "opencode")
            opencode_version: Specific opencode version to install (e.g., "0.2.0")
            **kwargs: Additional arguments passed to CliAgentEnv
        """
        super().__init__(run_command=run_command, **kwargs)
        self.opencode_version = opencode_version
        # Track pending streaming requests
        self.streaming_requests: dict[str, dict[str, Any]] = {}

    async def ensure_interception_server(self):
        """Start shared HTTP server with Responses API endpoint support."""
        async with self.server_lock:
            if self.interception_server is not None:
                return

            app = web.Application()
            # Add Responses API endpoint (primary for OpenCode)
            app.router.add_post(
                "/rollout/{rollout_id}/v1/responses",
                self.handle_responses_api_request,
            )
            # Keep Chat Completions endpoint for compatibility
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self.handle_intercepted_request,
            )
            # Add models endpoint (some agents query this)
            app.router.add_get(
                "/rollout/{rollout_id}/v1/models",
                self.handle_models_request,
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)
            await site.start()

            self.interception_server = app
            self.server_runner = runner
            self.server_site = site

            logger.debug(
                f"Started interception server on port {self.interception_port} "
                f"(Responses API + Chat Completions)"
            )

    async def handle_models_request(self, request: Any) -> Any:
        """Handle /v1/models request - return available models."""
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        # Return a simple models list
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": "gpt-4o",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                },
            ],
        })

    def convert_responses_input_to_messages(
        self, request_body: dict
    ) -> tuple[Messages, str | None]:
        """
        Convert Responses API input format to Chat Completions messages format.

        Responses API accepts:
        - input: str (simple text)
        - input: list[{role, content}] (message array)
        - instructions: str (system prompt)

        Returns (messages, instructions)
        """
        input_data = request_body.get("input", [])
        instructions = request_body.get("instructions")

        messages: Messages = []

        # Add instructions as system message if present
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Convert input to messages
        if isinstance(input_data, str):
            # Simple string input -> user message
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # Message array - convert role names if needed
            for msg in input_data:
                role = msg.get("role", "user")
                # Responses API uses "developer" for system-like messages
                if role == "developer":
                    role = "system"
                content = msg.get("content", "")
                messages.append({"role": role, "content": content})

        return messages, instructions

    def convert_tools_for_chat_completions(
        self, tools: list[dict] | None
    ) -> list[dict] | None:
        """
        Convert Responses API tools format to Chat Completions tools format.

        Responses API tools may have slightly different structure.
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                # Standard function tool - should be compatible
                converted.append(tool)
            elif tool_type == "computer_use_preview":
                # Computer use tool - skip for now (not supported in standard API)
                logger.debug(f"Skipping computer_use_preview tool")
                continue
            elif tool_type == "web_search":
                # Web search tool - skip (built-in to Responses API)
                logger.debug(f"Skipping web_search tool")
                continue
            elif tool_type == "code_interpreter":
                # Code interpreter - skip (built-in to Responses API)
                logger.debug(f"Skipping code_interpreter tool")
                continue
            else:
                # Unknown tool type - try to pass through
                converted.append(tool)

        return converted if converted else None

    def convert_chat_response_to_responses_format(
        self, chat_response: dict, request_id: str
    ) -> dict:
        """
        Convert Chat Completions response to Responses API format.

        Chat Completions response:
        {
            "id": "chatcmpl-xxx",
            "choices": [{
                "message": {"role": "assistant", "content": "...", "tool_calls": [...]},
                "finish_reason": "stop"
            }],
            ...
        }

        Responses API response:
        {
            "id": "resp_xxx",
            "object": "response",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "..."}]
            }],
            "status": "completed",
            ...
        }
        """
        choices = chat_response.get("choices", [])
        if not choices:
            return {
                "id": f"resp_{request_id}",
                "object": "response",
                "created_at": int(time.time()),
                "model": chat_response.get("model", "unknown"),
                "output": [],
                "status": "completed",
                "usage": chat_response.get("usage", {}),
            }

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        output = []

        # Convert text content
        text_content = message.get("content")
        if text_content:
            output.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text_content,
                    }
                ],
                "status": "completed",
            })

        # Convert tool calls
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            output.append({
                "type": "function_call",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
                "status": "completed",
            })

        # Map finish reason to status
        status = "completed"
        if finish_reason == "tool_calls":
            status = "completed"
        elif finish_reason == "length":
            status = "incomplete"
        elif finish_reason == "content_filter":
            status = "failed"

        return {
            "id": f"resp_{request_id}",
            "object": "response",
            "created_at": int(time.time()),
            "model": chat_response.get("model", "unknown"),
            "output": output,
            "status": status,
            "usage": chat_response.get("usage", {}),
        }

    async def generate_streaming_events(
        self, chat_response: dict, request_id: str
    ) -> list[dict]:
        """
        Generate Responses API streaming events from a complete Chat Completions response.

        This converts the full response into a sequence of SSE events that mimic
        real streaming from the Responses API.
        """
        events = []
        sequence_number = 0

        # response.created
        events.append({
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": {
                "id": f"resp_{request_id}",
                "object": "response",
                "status": "in_progress",
                "output": [],
            },
        })
        sequence_number += 1

        choices = chat_response.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            # Handle text content with streaming deltas
            text_content = message.get("content", "")
            if text_content:
                output_item_id = f"item_{uuid.uuid4().hex[:8]}"

                # response.output_item.added
                events.append({
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": output_item_id,
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress",
                    },
                })
                sequence_number += 1

                # response.content_part.added
                content_part_id = f"part_{uuid.uuid4().hex[:8]}"
                events.append({
                    "type": "response.content_part.added",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                    },
                })
                sequence_number += 1

                # Stream text in chunks (simulate streaming)
                chunk_size = 20  # Characters per chunk
                for i in range(0, len(text_content), chunk_size):
                    chunk = text_content[i:i + chunk_size]
                    events.append({
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": chunk,
                    })
                    sequence_number += 1

                # response.output_text.done
                events.append({
                    "type": "response.output_text.done",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text_content,
                })
                sequence_number += 1

                # response.content_part.done
                events.append({
                    "type": "response.content_part.done",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": text_content,
                    },
                })
                sequence_number += 1

                # response.output_item.done
                events.append({
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": output_item_id,
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                        "status": "completed",
                    },
                })
                sequence_number += 1

            # Handle tool calls
            tool_calls = message.get("tool_calls", [])
            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                func_name = func.get("name", "")
                func_args = func.get("arguments", "{}")

                output_index = 1 + tc_idx if text_content else tc_idx

                # response.output_item.added for function call
                events.append({
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": func_name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                })
                sequence_number += 1

                # Stream function call arguments
                chunk_size = 50
                for i in range(0, len(func_args), chunk_size):
                    chunk = func_args[i:i + chunk_size]
                    events.append({
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "call_id": call_id,
                        "delta": chunk,
                    })
                    sequence_number += 1

                # response.function_call_arguments.done
                events.append({
                    "type": "response.function_call_arguments.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "call_id": call_id,
                    "arguments": func_args,
                })
                sequence_number += 1

                # response.output_item.done for function call
                events.append({
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": func_name,
                        "arguments": func_args,
                        "status": "completed",
                    },
                })
                sequence_number += 1

        # response.completed
        final_response = self.convert_chat_response_to_responses_format(
            chat_response, request_id
        )
        events.append({
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": final_response,
        })

        return events

    def log_responses_request(self, rollout_id: str, body: dict) -> None:
        """Log Responses API request details."""
        logger.debug(f"[{rollout_id}] <- INTERCEPTED RESPONSES API REQUEST")
        input_data = body.get("input", [])
        if isinstance(input_data, str):
            logger.debug(f"  [input] {self.truncate(input_data)}")
        else:
            for msg in input_data:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, str):
                    logger.debug(f"  [{role}] {self.truncate(content)}")
                else:
                    logger.debug(f"  [{role}] {self.truncate(str(content))}")
        if body.get("instructions"):
            logger.debug(f"  [instructions] {self.truncate(body['instructions'])}")
        if body.get("tools"):
            logger.debug(f"  [tools] {len(body['tools'])} tool(s)")
        logger.debug(f"  [stream] {body.get('stream', False)}")

    async def handle_responses_api_request(self, request: Any) -> Any:
        """
        Handle Responses API request: intercept, convert to Chat Completions,
        get model response, convert back, and optionally stream.
        """
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        self.log_responses_request(rollout_id, request_body)

        is_streaming = request_body.get("stream", False)
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Convert Responses API request to Chat Completions format
        messages, instructions = self.convert_responses_input_to_messages(request_body)
        tools = self.convert_tools_for_chat_completions(request_body.get("tools"))

        # Create intercept data in Chat Completions format
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": messages,
            "model": request_body.get("model"),
            "tools": tools,
            "response_future": asyncio.Future(),
            "original_request": request_body,
            "is_streaming": is_streaming,
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        try:
            response_future = cast(asyncio.Future[Any], intercept["response_future"])
            response = await response_future
        except asyncio.CancelledError:
            return web.json_response({"error": "Rollout cancelled"}, status=499)
        except Exception as e:
            logger.error(f"Error processing Responses API request: {e}")
            return web.json_response({"error": str(e)}, status=500)

        # Convert response to dict
        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )

        if is_streaming:
            # Generate and stream SSE events
            return await self.stream_responses_events(response_dict, request_id, request)
        else:
            # Return non-streaming Responses API format
            responses_format = self.convert_chat_response_to_responses_format(
                response_dict, request_id
            )
            self.log_response(rollout_id, response_dict)
            return web.json_response(responses_format)

    async def stream_responses_events(
        self, chat_response: dict, request_id: str, http_request: Any
    ) -> web.StreamResponse:
        """
        Stream Responses API events as Server-Sent Events.
        """
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(http_request)

        events = await self.generate_streaming_events(chat_response, request_id)

        for event in events:
            event_data = json.dumps(event)
            # SSE format: data: <json>\n\n
            sse_line = f"data: {event_data}\n\n"
            await response.write(sse_line.encode("utf-8"))
            # Small delay to simulate real streaming
            await asyncio.sleep(0.01)

        # Send [DONE] marker
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()

        return response

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for OpenCode."""
        env_vars = await super().build_env_vars(state)

        # OpenCode uses standard OpenAI environment variables
        # OPENAI_BASE_URL is already set by parent
        # Add any OpenCode-specific variables
        env_vars.setdefault("OPENAI_API_KEY", "dummy-key-for-interception")

        return env_vars

    async def post_sandbox_setup(
        self, state: State, sandbox_client: Any
    ) -> None:
        """Install OpenCode if needed after sandbox creation."""
        sandbox_id = state["sandbox_id"]

        if self.opencode_version:
            # Install specific version of opencode
            install_cmd = f"go install github.com/opencode-ai/opencode@v{self.opencode_version}"
        else:
            # Install latest version
            install_cmd = "go install github.com/opencode-ai/opencode@latest"

        # Check if opencode is already available
        check_result = await sandbox_client.execute_command(
            sandbox_id, "which opencode || echo 'not found'"
        )

        if "not found" in check_result.get("stdout", ""):
            logger.debug(f"Installing opencode in sandbox {sandbox_id}")
            try:
                await sandbox_client.execute_command(
                    sandbox_id,
                    install_cmd,
                    timeout=300,
                )
            except Exception as e:
                logger.warning(f"Failed to install opencode: {e}")

        await super().post_sandbox_setup(state, sandbox_client)


class ResponsesAPIStreamBuffer:
    """
    Buffer for accumulating Responses API streaming events and reconstructing
    the complete response.
    """

    def __init__(self):
        self.events: list[dict] = []
        self.text_content: str = ""
        self.function_calls: dict[str, dict] = {}  # call_id -> {name, arguments}
        self.response_id: str | None = None
        self.model: str | None = None
        self.status: str = "in_progress"

    def add_event(self, event: dict) -> None:
        """Add a streaming event and update internal state."""
        self.events.append(event)
        event_type = event.get("type", "")

        if event_type == "response.created":
            resp = event.get("response", {})
            self.response_id = resp.get("id")
            self.model = resp.get("model")

        elif event_type == "response.output_text.delta":
            self.text_content += event.get("delta", "")

        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id", "")
            if call_id not in self.function_calls:
                self.function_calls[call_id] = {"name": "", "arguments": ""}
            self.function_calls[call_id]["arguments"] += event.get("delta", "")

        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                call_id = item.get("call_id", "")
                self.function_calls[call_id] = {
                    "name": item.get("name", ""),
                    "arguments": "",
                }

        elif event_type == "response.completed":
            self.status = "completed"

        elif event_type == "response.failed":
            self.status = "failed"

    def to_chat_completion_format(self) -> dict:
        """Convert buffered response to Chat Completions format."""
        message: dict[str, Any] = {"role": "assistant"}

        if self.text_content:
            message["content"] = self.text_content
        else:
            message["content"] = None

        if self.function_calls:
            tool_calls = []
            for call_id, call_data in self.function_calls.items():
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": call_data["arguments"],
                    },
                })
            message["tool_calls"] = tool_calls

        finish_reason = "stop"
        if self.function_calls:
            finish_reason = "tool_calls"
        elif self.status == "incomplete":
            finish_reason = "length"

        return {
            "id": self.response_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model or "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {},
        }
