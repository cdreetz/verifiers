"""Monkey-patching integration for Braintrust tracing in verifiers v1.

Activate with a single call at the top of your environment file::

    from verifiers.v1.experimental.braintrust_tracing import setup_v1_tracing
    setup_v1_tracing()

This patches the v1 ``Env``, ``Harness``, and ``Runtime`` classes in-place so
that all subclasses automatically produce Braintrust traces.  Every patch is a
safe wrap that calls the original method — no replacements.

The patching is idempotent — calling ``setup_v1_tracing()`` multiple times is
safe and has no additional effect after the first call.

Span hierarchy produced per rollout::

    rollout (type=task)              ← Harness.run() wrap
    ├── model_request (type=llm)     ← Runtime.submit_model_request() wrap
    ├── tool_call:search (type=tool) ← Runtime._call_tool() wrap
    ├── tool_call:python (type=tool)
    ├── model_request (type=llm)
    ├── tool_call:submit (type=tool)
    └── scoring (type=score)         ← Runtime.score_rollout() wrap

    group (for grouped rollouts)     ← Env._run_group_states() wrap
    ├── rollout 1 (child)
    ├── rollout 2 (child)
    └── group_scoring
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import Any

import verifiers.v1.experimental.braintrust_tracing.braintrust_tracing as _bt

_log = logging.getLogger(__name__)

_PATCH_MARKER = "__verifiers_v1_bt_patched__"

# ContextVar to pass the current rollout span from Harness.run() down to
# Runtime.submit_model_request() and Runtime._call_tool().  Each coroutine
# gets its own isolated value, so concurrent rollouts don't interfere.
_current_rollout_span: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_v1_current_rollout_span", default=None
)


def _is_patched(cls: type) -> bool:
    return cls.__dict__.get(_PATCH_MARKER, False)


def _mark_patched(cls: type) -> None:
    setattr(cls, _PATCH_MARKER, True)


# ---------------------------------------------------------------------------
# Harness patches — rollout span + scoring sub-span
# ---------------------------------------------------------------------------


def _patch_harness() -> None:
    from verifiers.v1.harness import Harness

    if _is_patched(Harness):
        return

    _orig_run = Harness.run

    async def _traced_run(self: Any, task: Any, state: Any = None) -> Any:
        model = ""
        env_id = ""
        example_id = ""
        try:
            taskset = getattr(self, "taskset", None)
            env_id = getattr(taskset, "taskset_id", "") if taskset else ""
            if state is not None:
                runtime = state.get("runtime", {})
                if isinstance(runtime, dict):
                    model = runtime.get("model", "")
            if isinstance(task, dict):
                example_id = task.get("example_id", task.get("task_id", ""))
        except Exception:
            pass

        bt_span = _bt.rollout_started(
            env_id=str(env_id),
            model=str(model),
            example_id=str(example_id),
            trajectory_id="",
        )
        _current_rollout_span.set(bt_span)
        t0 = time.monotonic()
        error_msg = ""
        result_state = None
        try:
            result_state = await _orig_run(self, task, state)
            return result_state
        except Exception as exc:
            error_msg = repr(exc)[:500]
            raise
        except BaseException as exc:
            error_msg = repr(exc)[:500]
            raise
        finally:
            dur = time.monotonic() - t0
            reward = None
            num_turns = 0
            stop_condition = ""
            input_tokens = 0.0
            output_tokens = 0.0
            if result_state is not None:
                reward = result_state.get("reward")
                trajectory = result_state.get("trajectory", [])
                num_turns = len(trajectory) if isinstance(trajectory, list) else 0
                stop_condition = result_state.get("stop_condition", "")
                timing = result_state.get("timing")
                if timing is not None:
                    try:
                        tokens = (
                            timing.get("tokens")
                            if isinstance(timing, dict)
                            else getattr(timing, "tokens", None)
                        )
                        if tokens is not None:
                            if isinstance(tokens, dict):
                                input_tokens = float(tokens.get("prompt_tokens", 0))
                                output_tokens = float(
                                    tokens.get("completion_tokens", 0)
                                )
                            else:
                                input_tokens = float(
                                    getattr(tokens, "prompt_tokens", 0)
                                )
                                output_tokens = float(
                                    getattr(tokens, "completion_tokens", 0)
                                )
                    except Exception:
                        pass
            _bt.rollout_completed(
                bt_span,
                reward=reward,
                num_turns=num_turns,
                duration_s=dur,
                stop_condition=str(stop_condition),
                error=error_msg,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            _current_rollout_span.set(None)

    Harness.run = _traced_run  # type: ignore[assignment]
    _mark_patched(Harness)


# ---------------------------------------------------------------------------
# Runtime patches — model_request, tool_call, scoring spans
# ---------------------------------------------------------------------------


def _patch_runtime() -> None:
    from verifiers.v1.runtime import Runtime

    if _is_patched(Runtime):
        return

    # -- submit_model_request: model_request span --
    _orig_submit = Runtime.submit_model_request

    async def _traced_submit_model_request(
        self: Any,
        prompt: Any,
        task: Any,
        state: Any,
        tool_defs: Any = None,
        extras: Any = None,
    ) -> Any:
        bt_parent = _current_rollout_span.get(None)
        model = ""
        turn_index = 0
        try:
            model = self.model(state)
        except Exception:
            pass
        try:
            trajectory = state.get("trajectory", [])
            turn_index = len(trajectory) if isinstance(trajectory, list) else 0
        except Exception:
            pass

        bt_span = _bt.model_request_span(
            bt_parent,
            model=str(model),
            turn_index=turn_index,
            messages=prompt,
        )
        t0 = time.monotonic()
        error_msg = ""
        response = None
        try:
            response = await _orig_submit(self, prompt, task, state, tool_defs, extras)
            return response
        except Exception as exc:
            error_msg = repr(exc)[:500]
            raise
        except BaseException as exc:
            error_msg = repr(exc)[:500]
            raise
        finally:
            dur = time.monotonic() - t0
            input_tok, output_tok = 0.0, 0.0
            if not error_msg and response is not None:
                try:
                    from verifiers.utils.response_utils import (
                        parse_response_tokens,
                    )

                    tokens = None
                    if asyncio.iscoroutinefunction(parse_response_tokens):
                        # Can't await in finally, extract from state instead
                        pass
                    # Extract tokens from the trajectory step that was just appended
                    trajectory = state.get("trajectory", [])
                    if isinstance(trajectory, list) and trajectory:
                        last_step = trajectory[-1]
                        if isinstance(last_step, dict):
                            tokens = last_step.get("tokens")
                            if isinstance(tokens, dict):
                                input_tok = float(tokens.get("prompt_tokens", 0))
                                output_tok = float(tokens.get("completion_tokens", 0))
                except Exception:
                    pass
            _bt.model_request_completed(
                bt_span,
                duration_s=dur,
                input_tokens=input_tok,
                output_tokens=output_tok,
                response=response if not error_msg else None,
                error=error_msg,
            )

    # -- _call_tool: tool_call span --
    _orig_call_tool = Runtime._call_tool

    async def _traced_call_tool(
        self: Any,
        tool_name: str,
        task: Any,
        state: Any,
        exposed: bool,
        **kwargs: Any,
    ) -> Any:
        bt_parent = _current_rollout_span.get(None)
        bt_span = _bt.tool_call_started(
            bt_parent,
            tool_name=tool_name,
            tool_call_id="",
            tool_args=kwargs if kwargs else None,
        )
        t0 = time.monotonic()
        err_msg = ""
        result = None
        try:
            result = await _orig_call_tool(
                self, tool_name, task, state, exposed, **kwargs
            )
            return result
        except Exception as exc:
            err_msg = repr(exc)[:500]
            raise
        finally:
            result_str = None
            if not err_msg and result is not None:
                try:
                    result_str = str(result)[:2000]
                except Exception:
                    pass
            _bt.tool_call_completed(
                bt_span,
                duration_s=time.monotonic() - t0,
                result=result_str,
                error=err_msg,
            )

    # -- score_rollout: scoring span --
    _orig_score_rollout = Runtime.score_rollout

    async def _traced_score_rollout(self: Any, task: Any, state: Any) -> Any:
        bt_parent = _current_rollout_span.get(None)
        trajectory_id = ""
        try:
            trajectory_id = str(state.get("trajectory_id", ""))
        except Exception:
            pass
        bt_span = _bt.scoring_started(bt_parent, trajectory_id=trajectory_id)
        t0 = time.monotonic()
        try:
            result = await _orig_score_rollout(self, task, state)
            return result
        finally:
            reward = None
            try:
                reward = state.get("reward")
            except Exception:
                pass
            _bt.scoring_completed(
                bt_span,
                duration_s=time.monotonic() - t0,
                reward=reward,
            )

    Runtime.submit_model_request = _traced_submit_model_request  # type: ignore[assignment]
    Runtime._call_tool = _traced_call_tool  # type: ignore[assignment]
    Runtime.score_rollout = _traced_score_rollout  # type: ignore[assignment]
    _mark_patched(Runtime)


# ---------------------------------------------------------------------------
# Env patches — group span + generate run tags
# ---------------------------------------------------------------------------


def _patch_env() -> None:
    from verifiers.v1.env import Env

    if _is_patched(Env):
        return

    _orig_run_group = Env._run_group_states

    async def _traced_run_group_states(
        self: Any,
        group_inputs: list,
        client: Any,
        model: str,
        sampling_args: Any,
    ) -> list:
        t0 = time.monotonic()
        example_id = ""
        try:
            if group_inputs:
                example_id = group_inputs[0].get("example_id", "")
        except Exception:
            pass
        env_id = ""
        try:
            taskset = getattr(self, "taskset", None)
            env_id = getattr(taskset, "taskset_id", "") if taskset else ""
        except Exception:
            pass

        bt_group = _bt.group_started(
            env_id=str(env_id),
            model=str(model),
            example_id=str(example_id),
            group_size=len(group_inputs),
        )
        group_states: list = []
        try:
            group_states = await _orig_run_group(
                self, group_inputs, client, model, sampling_args
            )
            return group_states
        finally:
            rewards = []
            for s in group_states:
                try:
                    r = s.get("reward")
                    if r is not None:
                        rewards.append(float(r))
                except Exception:
                    pass
            _bt.group_completed(
                bt_group,
                duration_s=time.monotonic() - t0,
                avg_reward=sum(rewards) / len(rewards) if rewards else None,
                group_size=len(group_inputs),
            )

    Env._run_group_states = _traced_run_group_states  # type: ignore[assignment]
    _mark_patched(Env)


def _patch_environment_generate() -> None:
    """Patch Environment.generate() for run-level tags.

    This reuses the same pattern as the v0 integration.  Since v1's ``Env``
    inherits from ``Environment`` and does not override ``generate()``, the
    patch applies to v1 environments automatically.
    """
    from verifiers.envs.environment import Environment

    # Check for both v0 and v1 patch markers to avoid double-patching
    if _is_patched(Environment):
        return
    if Environment.__dict__.get("__verifiers_bt_patched__", False):
        return

    _orig_generate = Environment.generate

    async def _traced_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _bt.enabled():
            run_tags = _bt.get_run_tags() or _bt.set_run_tags()
            if run_tags:
                _log.info("Braintrust run tag: %s", run_tags[0])
        try:
            return await _orig_generate(self, *args, **kwargs)
        finally:
            _bt.clear_run_tags()

    Environment.generate = _traced_generate  # type: ignore[assignment]
    _mark_patched(Environment)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_v1_tracing(
    *,
    api_key: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Activate Braintrust tracing for verifiers v1 environments.

    Call this once at the top of your environment file::

        from verifiers.v1.experimental.braintrust_tracing import setup_v1_tracing
        setup_v1_tracing()

    This patches ``Env``, ``Harness``, and ``Runtime`` so that every rollout,
    model request, tool call, and scoring phase is traced as nested spans in
    Braintrust.  The patching is idempotent.

    Args:
        api_key: Braintrust API key.  Falls back to ``BRAINTRUST_API_KEY``
            env var.
        project: Braintrust project name.  Falls back to
            ``VF_BRAINTRUST_PROJECT`` env var, then ``"verifiers"``.
        tags: Optional run-level tags attached to every trace.  When ``None``
            a unique tag is auto-generated per ``generate()`` call.

    Returns:
        ``True`` if tracing is active (API key configured and ``braintrust``
        installed), ``False`` otherwise.
    """
    import os

    if api_key is not None:
        os.environ["BRAINTRUST_API_KEY"] = api_key
    if project is not None:
        os.environ["VF_BRAINTRUST_PROJECT"] = project

    if api_key is not None or project is not None:
        _bt._INSTANCE = None

    _patch_harness()
    _patch_runtime()
    _patch_env()
    _patch_environment_generate()

    if tags is not None:
        _bt.set_run_tags(tags)

    is_active = _bt.enabled()
    if is_active:
        _log.info("Braintrust v1 tracing integration activated")
    else:
        _log.debug(
            "Braintrust v1 tracing integration installed but inactive "
            "(set BRAINTRUST_API_KEY to enable)"
        )
    return is_active
