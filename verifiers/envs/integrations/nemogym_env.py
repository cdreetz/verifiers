"""NeMo Gym integration for Verifiers.

Wraps any NeMo Gym resources server (SimpleResourcesServer or GymnasiumServer)
as a v1 ``vf.Taskset`` that can be used with ``vf.Env``.

Usage::

    from verifiers.envs.integrations.nemogym_env import NemoGymTaskset
    import verifiers as vf

    taskset = NemoGymTaskset(
        server_cls=MyResourcesServer,
        data_path="path/to/example.jsonl",
    )
    env = vf.Env(taskset=taskset)
"""

import inspect
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
from httpx import ASGITransport
from pydantic_core import PydanticUndefined

from verifiers.v1 import Taskset, Toolset

logger = logging.getLogger(__name__)

try:
    from nemo_gym.base_resources_server import BaseResourcesServerConfig
    from nemo_gym.openai_utils import (
        NeMoGymResponse,
        NeMoGymResponseCreateParamsNonStreaming,
        NeMoGymResponseOutputMessage,
        NeMoGymResponseOutputText,
    )
    from nemo_gym.server_utils import ServerClient
except ImportError as e:
    raise ImportError(
        "NemoGymTaskset requires nemo-gym. Install with: uv add 'verifiers[nemogym]'"
    ) from e

_GYMNASIUM_SERVER_CLS: type | None = None

_STANDARD_ROUTES = frozenset(
    {
        "/seed_session",
        "/verify",
        "/aggregate_metrics",
        "/reset",
        "/step",
        "/openapi.json",
        "/docs",
        "/redoc",
    }
)


def _get_gymnasium_server_cls() -> type | None:
    global _GYMNASIUM_SERVER_CLS
    if _GYMNASIUM_SERVER_CLS is not None:
        return _GYMNASIUM_SERVER_CLS
    try:
        from resources_servers.gymnasium.base import GymnasiumServer

        _GYMNASIUM_SERVER_CLS = GymnasiumServer
        return GymnasiumServer
    except ImportError:
        return None


def _is_gymnasium_server(server_cls: type) -> bool:
    gym_cls = _get_gymnasium_server_cls()
    if gym_cls is None:
        return False
    return issubclass(server_cls, gym_cls)


def _discover_tool_routes(app: object) -> list[str]:
    """Find non-standard POST routes registered on a FastAPI app."""
    routes: list[str] = []
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path and "POST" in methods and path not in _STANDARD_ROUTES:
            routes.append(path)
    return sorted(routes)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _extract_system_prompt(messages: list[dict[str, str]]) -> str | None:
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            return msg.get("content", "")
    return None


def _extract_user_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [msg for msg in messages if msg.get("role") not in ("system", "developer")]


def _completion_to_nemogym_response(
    completion: list[dict[str, Any]],
) -> NeMoGymResponse:
    """Convert a verifiers completion (list of message dicts) to NeMoGymResponse."""
    text_parts: list[str] = []
    for msg in completion:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                text_parts.append(content)

    combined_text = "\n".join(text_parts) if text_parts else ""
    return NeMoGymResponse(
        id=f"resp_{uuid.uuid4().hex[:8]}",
        created_at=0.0,
        model="verifiers",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=combined_text,
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _default_for_annotation(ann: type | None) -> Any:
    """Produce a zero-value default for a type annotation."""
    import typing

    if ann is None:
        return ""
    if ann is str:
        return ""
    if ann is int:
        return 0
    if ann is float:
        return 0.0
    if ann is bool:
        return False
    origin = getattr(ann, "__origin__", None)
    if origin is list or ann is list:
        return []
    if origin is dict or ann is dict:
        return {}
    if origin is typing.Literal:
        args = typing.get_args(ann)
        return args[0] if args else ""
    if hasattr(ann, "model_fields"):
        return _build_pydantic_stub(ann)
    return MagicMock()


def _build_pydantic_stub(model_cls: type) -> Any:
    """Recursively construct a Pydantic model with zero-value defaults."""
    kwargs: dict[str, Any] = {}
    for fname, finfo in model_cls.model_fields.items():
        if finfo.default is not PydanticUndefined:
            continue
        kwargs[fname] = _default_for_annotation(finfo.annotation)
    return model_cls(**kwargs)


def _make_server(server_cls: type, server_config: dict[str, Any] | None = None) -> Any:
    """Instantiate a NeMo Gym server class with minimal config."""
    cfg = server_config or {}
    config_cls: type = BaseResourcesServerConfig
    for field_name, field_info in server_cls.model_fields.items():
        if field_name == "config" and field_info.annotation is not None:
            config_cls = field_info.annotation
            break

    base_fields = {"host": "", "port": 0, "entrypoint": "", "name": ""}
    init_kwargs = {k: cfg.get(k, v) for k, v in base_fields.items()}
    init_kwargs.update({k: v for k, v in cfg.items() if k not in base_fields})

    for fname, finfo in config_cls.model_fields.items():
        if fname in init_kwargs:
            continue
        if finfo.default is not PydanticUndefined:
            continue
        init_kwargs[fname] = _default_for_annotation(finfo.annotation)

    config = config_cls(**init_kwargs)
    return server_cls(config=config, server_client=MagicMock(spec=ServerClient))


def _jsonl_to_task_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert NeMo Gym JSONL rows to verifiers task row format."""
    task_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        rcp = row.get("responses_create_params", {})
        messages = rcp.get("input", [])
        system_prompt = _extract_system_prompt(messages)
        user_messages = _extract_user_messages(messages)
        prompt = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in user_messages
        ]

        info: dict[str, Any] = {"nemogym_row_index": i}
        for key in (
            "verifier_metadata",
            "answer",
            "question",
            "metadata",
            "agent_ref",
        ):
            if key in row:
                info[key] = row[key]
        nemogym_tools = rcp.get("tools")
        if nemogym_tools:
            info["nemogym_tools"] = nemogym_tools
        info["responses_create_params"] = rcp

        task_row: dict[str, Any] = {
            "prompt": prompt,
            "example_id": i,
            "info": json.dumps(info),
        }
        if system_prompt:
            task_row["system_prompt"] = system_prompt
        if "answer" in row:
            task_row["answer"] = str(row["answer"])
        task_rows.append(task_row)
    return task_rows


def _make_tool_callable_from_route(app: object, route_path: str) -> Callable[..., Any]:
    """Create an async callable that dispatches to a NeMo Gym server endpoint."""

    async def tool_fn(**kwargs: Any) -> str:
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(
            transport=transport, base_url="http://nemogym"
        ) as client:
            resp = await client.post(route_path, json=kwargs)
            return resp.text

    route_name = route_path.lstrip("/")
    tool_fn.__name__ = route_name
    tool_fn.__qualname__ = route_name

    endpoint_fn = _find_endpoint(app, route_path)
    if endpoint_fn:
        _apply_pydantic_signature(tool_fn, endpoint_fn)

    return tool_fn


def _find_endpoint(app: object, route_path: str) -> Callable[..., Any] | None:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) == route_path:
            return getattr(route, "endpoint", None)
    return None


def _apply_pydantic_signature(
    tool_fn: Callable[..., Any], endpoint_fn: Callable[..., Any]
) -> None:
    """Extract parameter schema from endpoint's Pydantic body model."""
    sig = inspect.signature(endpoint_fn)
    body_param = None
    for p in sig.parameters.values():
        if p.name in ("self", "request"):
            continue
        if p.annotation is not inspect.Parameter.empty:
            body_param = p
            break
    if body_param is None or not hasattr(body_param.annotation, "model_fields"):
        return
    fields = body_param.annotation.model_fields
    new_params = []
    for fname, finfo in fields.items():
        ann = finfo.annotation if finfo.annotation is not None else str
        default = (
            finfo.default
            if finfo.default is not PydanticUndefined
            else inspect.Parameter.empty
        )
        new_params.append(
            inspect.Parameter(
                fname,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=ann,
            )
        )
    if new_params:
        tool_fn.__signature__ = inspect.Signature(new_params)  # type: ignore[attr-defined]

    doc = getattr(endpoint_fn, "__doc__", None)
    if doc:
        tool_fn.__doc__ = doc


def _build_tool_callables(
    app: object,
    tool_routes: list[str],
    nemogym_tools: list[dict[str, Any]] | None = None,
) -> list[Callable[..., Any]]:
    """Build tool callables from JSONL tool defs (preferred) or routes."""
    if nemogym_tools:
        route_set = {r.lstrip("/") for r in tool_routes}
        callables: list[Callable[..., Any]] = []
        for tool_def in nemogym_tools:
            name = tool_def.get("name", "")
            if name not in route_set:
                continue
            fn = _make_jsonl_tool_callable(app, name, tool_def)
            callables.append(fn)
        return callables
    return [_make_tool_callable_from_route(app, route) for route in tool_routes]


def _make_jsonl_tool_callable(
    app: object, name: str, tool_def: dict[str, Any]
) -> Callable[..., Any]:
    """Create a tool callable from a JSONL tool definition."""
    route_path = f"/{name}"

    def _make_fn(path: str = route_path) -> Callable[..., Any]:
        async def tool_fn(**kwargs: Any) -> str:
            transport = ASGITransport(app=app)  # type: ignore[arg-type]
            async with httpx.AsyncClient(
                transport=transport, base_url="http://nemogym"
            ) as client:
                resp = await client.post(path, json=kwargs)
                return resp.text

        return tool_fn

    fn = _make_fn()
    fn.__name__ = name  # type: ignore[attr-defined]
    fn.__qualname__ = name  # type: ignore[attr-defined]

    params_schema = tool_def.get("parameters", {})
    properties = params_schema.get("properties", {})
    required = set(params_schema.get("required", []))
    sig_params = []
    for pname in properties:
        default = inspect.Parameter.empty if pname in required else None
        sig_params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=str,
            )
        )
    if sig_params:
        fn.__signature__ = inspect.Signature(sig_params)  # type: ignore[attr-defined]

    desc = tool_def.get("description", "")
    if desc:
        fn.__doc__ = desc

    return fn


class NemoGymTaskset(Taskset):
    """V1 Taskset wrapping any NeMo Gym resources server.

    Auto-detects the server type (SimpleResourcesServer vs GymnasiumServer)
    and configures tools, rewards, and multi-turn handlers accordingly.

    Args:
        server_cls: The NeMo Gym server class (e.g. ``BlackjackEnv``).
        data_path: Path to the NeMo Gym JSONL data file.
        server_config: Optional dict of server config overrides.
        max_steps: Maximum steps for gymnasium environments (default 50).
    """

    def __init__(
        self,
        server_cls: type,
        data_path: str | Path,
        server_config: dict[str, Any] | None = None,
        max_steps: int = 50,
        **kwargs: Any,
    ):
        self._server_cls = server_cls
        self._data_path = Path(data_path)
        self._server_config = server_config
        self._max_steps = max_steps
        self._is_gymnasium = _is_gymnasium_server(server_cls)

        self._server = _make_server(server_cls, server_config)
        self._app = self._server.setup_webserver()

        raw_rows = _load_jsonl(self._data_path)
        task_rows = _jsonl_to_task_rows(raw_rows)
        self._raw_rows = raw_rows
        self._task_rows = task_rows

        system_prompt = None
        if task_rows:
            system_prompt = task_rows[0].get("system_prompt")

        rewards: list[Any] = []
        toolsets: list[Any] = []
        setups: list[Any] = []
        stops: list[Any] = []
        cleanups: list[Any] = []

        if self._is_gymnasium:
            rewards.append(self._gymnasium_reward)
            setups.append(self._gymnasium_setup)
            stops.append(self._gymnasium_stop)
            cleanups.append(self._gymnasium_cleanup)
            kwargs.setdefault("user", self._gymnasium_user)
        else:
            rewards.append(self._verify_reward)
            tool_routes = _discover_tool_routes(self._app)
            if tool_routes:
                first_info = (
                    json.loads(task_rows[0].get("info", "{}")) if task_rows else {}
                )
                tool_callables = _build_tool_callables(
                    self._app, tool_routes, first_info.get("nemogym_tools")
                )
                if tool_callables:
                    toolsets.append(Toolset(tools=tool_callables))

        def source() -> Any:
            yield from self._task_rows

        super().__init__(
            source=source,
            system_prompt=system_prompt,
            rewards=rewards,
            toolsets=toolsets if toolsets else None,
            setups=setups,
            stops=stops,
            cleanups=cleanups,
            **kwargs,
        )

    # -- Gymnasium handlers --

    async def _gymnasium_setup(self, task: Any, state: Any) -> None:
        session_id = uuid.uuid4().hex
        state.setdefault("runtime", {})
        state["runtime"]["nemogym_session_id"] = session_id
        state["runtime"]["nemogym_terminated"] = False
        state["runtime"]["nemogym_reward"] = 0.0

        metadata = self._extract_metadata(task)
        obs, _ = await self._server.reset(metadata, session_id=session_id)
        if obs:
            state["prompt"] = [{"role": "user", "content": obs}]

    async def _gymnasium_user(
        self, task: Any, state: Any, transcript: Any = None
    ) -> list[dict[str, str]] | None:
        runtime = state.get("runtime", {})
        if runtime.get("nemogym_terminated", False):
            return None

        session_id = runtime.get("nemogym_session_id", "")
        if not transcript:
            return None

        last_content = self._last_assistant_content(transcript)
        if last_content is None:
            return None

        response = _completion_to_nemogym_response(
            [{"role": "assistant", "content": last_content}]
        )
        metadata = self._extract_metadata(task)
        obs, reward, terminated, truncated, _ = await self._server.step(
            response, metadata, session_id=session_id
        )

        runtime["nemogym_reward"] = runtime.get("nemogym_reward", 0.0) + reward
        if terminated or truncated:
            runtime["nemogym_terminated"] = True
            return None

        if obs:
            return [{"role": "user", "content": obs}]
        return None

    async def _gymnasium_stop(self, task: Any, state: Any) -> bool:
        runtime = state.get("runtime", {})
        if runtime.get("nemogym_terminated", False):
            return True
        turn = state.get("turn", 0)
        return turn >= self._max_steps

    async def _gymnasium_reward(self, task: Any, state: Any) -> float:
        return float(state.get("runtime", {}).get("nemogym_reward", 0.0))

    async def _gymnasium_cleanup(self, task: Any, state: Any) -> None:
        session_id = state.get("runtime", {}).get("nemogym_session_id")
        if session_id and hasattr(self._server, "close_session"):
            await self._server.close_session(session_id)

    # -- SimpleResourcesServer reward --

    async def _verify_reward(self, task: Any, state: Any) -> float:
        completion = state.get("completion", [])
        if not completion:
            return 0.0

        response = _completion_to_nemogym_response(completion)

        info_str = task.get("info")
        info = json.loads(info_str) if isinstance(info_str, str) else (info_str or {})
        rcp = info.get("responses_create_params", {})
        rcp_obj = NeMoGymResponseCreateParamsNonStreaming(**rcp)

        verify_body: dict[str, Any] = {
            "responses_create_params": rcp_obj.model_dump(),
            "response": response.model_dump(),
        }

        raw_idx = info.get("nemogym_row_index", 0)
        if raw_idx < len(self._raw_rows):
            raw_row = self._raw_rows[raw_idx]
            for k, v in raw_row.items():
                if k not in ("responses_create_params", "agent_ref"):
                    verify_body[k] = v

        transport = ASGITransport(app=self._app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(
            transport=transport, base_url="http://nemogym"
        ) as client:
            resp = await client.post("/verify", json=verify_body)
            if resp.status_code == 200:
                result = resp.json()
                return float(result.get("reward", 0.0))
            logger.warning(
                "verify returned status %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            return 0.0

    # -- Helpers --

    @staticmethod
    def _extract_metadata(task: Any) -> dict[str, Any]:
        info_str = task.get("info")
        if not info_str:
            return {}
        info = json.loads(info_str) if isinstance(info_str, str) else info_str
        rcp = info.get("responses_create_params", {})
        return {k: v for k, v in rcp.items() if k != "input"}

    @staticmethod
    def _last_assistant_content(transcript: Any) -> str | None:
        for msg in reversed(transcript):
            role = (
                msg.get("role", "")
                if isinstance(msg, dict)
                else getattr(msg, "role", "")
            )
            if role == "assistant":
                return (
                    msg.get("content", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", "")
                )
        return None
