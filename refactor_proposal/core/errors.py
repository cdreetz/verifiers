"""
Error hierarchy for verifiers.

Design:
- All errors inherit from VerifierError
- Errors are specific enough to handle programmatically
- Include context for debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class VerifierError(Exception):
    """Base class for all verifier errors."""

    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class RolloutError(VerifierError):
    """Error during rollout execution."""

    pass


class LLMError(RolloutError):
    """Error from LLM API call."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        status_code: int | None = None,
        **context: Any,
    ):
        super().__init__(message, model=model, status_code=status_code, **context)
        self.model = model
        self.status_code = status_code


class ToolError(RolloutError):
    """Error during tool execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        **context: Any,
    ):
        super().__init__(
            message, tool_name=tool_name, tool_call_id=tool_call_id, **context
        )
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id


class TimeoutError(RolloutError):
    """Operation timed out."""

    def __init__(
        self,
        message: str,
        *,
        timeout_ms: float | None = None,
        operation: str | None = None,
        **context: Any,
    ):
        super().__init__(message, timeout_ms=timeout_ms, operation=operation, **context)
        self.timeout_ms = timeout_ms
        self.operation = operation


class TruncationError(RolloutError):
    """Rollout was truncated (hit limits)."""

    def __init__(
        self,
        message: str,
        *,
        turn_count: int | None = None,
        token_count: int | None = None,
        limit_type: str | None = None,
        **context: Any,
    ):
        super().__init__(
            message,
            turn_count=turn_count,
            token_count=token_count,
            limit_type=limit_type,
            **context,
        )
        self.turn_count = turn_count
        self.token_count = token_count
        self.limit_type = limit_type


class ScoringError(VerifierError):
    """Error during scoring/reward computation."""

    pass


class RewardFunctionError(ScoringError):
    """Error in a reward function."""

    def __init__(
        self,
        message: str,
        *,
        function_name: str | None = None,
        **context: Any,
    ):
        super().__init__(message, function_name=function_name, **context)
        self.function_name = function_name


class ParseError(ScoringError):
    """Error parsing completion."""

    def __init__(
        self,
        message: str,
        *,
        parser_type: str | None = None,
        content_preview: str | None = None,
        **context: Any,
    ):
        super().__init__(
            message, parser_type=parser_type, content_preview=content_preview, **context
        )
        self.parser_type = parser_type
        self.content_preview = content_preview


class ResourceError(VerifierError):
    """Error with resource management."""

    pass


class PoolExhaustedError(ResourceError):
    """Resource pool is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        pool_size: int | None = None,
        wait_time_ms: float | None = None,
        **context: Any,
    ):
        super().__init__(
            message, pool_size=pool_size, wait_time_ms=wait_time_ms, **context
        )
        self.pool_size = pool_size
        self.wait_time_ms = wait_time_ms


class ResourceUnhealthyError(ResourceError):
    """Resource failed health check."""

    def __init__(
        self,
        message: str,
        *,
        resource_id: str | None = None,
        **context: Any,
    ):
        super().__init__(message, resource_id=resource_id, **context)
        self.resource_id = resource_id


class ConfigurationError(VerifierError):
    """Invalid configuration."""

    pass


class EnvironmentNotFoundError(ConfigurationError):
    """Environment could not be loaded."""

    def __init__(
        self,
        message: str,
        *,
        env_id: str | None = None,
        **context: Any,
    ):
        super().__init__(message, env_id=env_id, **context)
        self.env_id = env_id
