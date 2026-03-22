"""Sandbox-specific error types with context for attribution."""

from dataclasses import dataclass, field
from typing import Any


class SandboxError(Exception):
    """Base class for all sandbox errors."""

    def __init__(
        self,
        message: str,
        sandbox_id: str | None = None,
        rollout_id: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.sandbox_id = sandbox_id
        self.rollout_id = rollout_id
        self.cause = cause

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.sandbox_id:
            parts.append(f"sandbox_id={self.sandbox_id}")
        if self.rollout_id:
            parts.append(f"rollout_id={self.rollout_id}")
        return " ".join(parts)


class SandboxCreationError(SandboxError):
    """Failed to create sandbox."""

    pass


class SandboxNotReadyError(SandboxError):
    """Sandbox failed to become ready after creation."""

    pass


class SandboxSetupError(SandboxError):
    """Post-creation setup hook failed."""

    pass


class SandboxExecutionError(SandboxError):
    """Command execution failed."""

    def __init__(
        self,
        message: str,
        command: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.command = command


class SandboxOOMError(SandboxExecutionError):
    """Sandbox ran out of memory."""

    pass


class SandboxTimeoutError(SandboxExecutionError):
    """Sandbox-level timeout (not command timeout)."""

    pass


class CommandTimeoutError(SandboxExecutionError):
    """Command timed out."""

    def __init__(
        self,
        message: str,
        timeout: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.timeout = timeout


class SandboxHealthError(SandboxError):
    """Health check failed."""

    pass


@dataclass(slots=True)
class SandboxFailureInfo:
    """Tracks how a sandbox failed for attribution."""

    oom: bool = False
    timeout: bool = False  # sandbox-level timeout
    command_timeout: bool = False
    creation_failed: bool = False
    setup_failed: bool = False
    not_ready: bool = False
    health_check_failed: bool = False
    error_message: str | None = None
    failed_command: str | None = None
    failed_at: float | None = None

    @property
    def has_failure(self) -> bool:
        return any((
            self.oom,
            self.timeout,
            self.command_timeout,
            self.creation_failed,
            self.setup_failed,
            self.not_ready,
            self.health_check_failed,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "oom": self.oom,
            "timeout": self.timeout,
            "command_timeout": self.command_timeout,
            "creation_failed": self.creation_failed,
            "setup_failed": self.setup_failed,
            "not_ready": self.not_ready,
            "health_check_failed": self.health_check_failed,
            "error_message": self.error_message,
            "failed_command": self.failed_command,
            "failed_at": self.failed_at,
        }
