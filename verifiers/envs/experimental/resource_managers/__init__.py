"""Experimental resource management abstractions for verifiers.

Key patterns inspired by frontier-evals' Alcatraz:
- AsyncExitStack for guaranteed LIFO cleanup
- Centralized limits via TypedDict
- File-based GC leader election
- Port allocation with context managers
"""

from verifiers.envs.experimental.resource_managers.errors import (
    CommandTimeoutError,
    SandboxCreationError,
    SandboxError,
    SandboxExecutionError,
    SandboxFailureInfo,
    SandboxHealthError,
    SandboxNotReadyError,
    SandboxOOMError,
    SandboxSetupError,
    SandboxTimeoutError,
)
from verifiers.envs.experimental.resource_managers.base import (
    ManagedResource,
    ResourceError,
    ResourceManager,
    ResourceState,
    managed_resources,
)
from verifiers.envs.experimental.resource_managers.retry import (
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    is_connection_error,
    is_transient_error,
)
from verifiers.envs.experimental.resource_managers.limits import (
    AGGRESSIVE_LIMITS,
    CONSERVATIVE_LIMITS,
    DEFAULT_SANDBOX_LIMITS,
    SandboxLimits,
    merge_limits,
)
from verifiers.envs.experimental.resource_managers.sandbox_manager import (
    BackgroundJob,
    FileLock,
    ManagedSandbox,
    SandboxManager,
    SetupCallback,
    ThreadedAsyncSandboxClient,
    allocate_port,
    get_available_ports,
    sandbox_manager,
)
from verifiers.envs.experimental.resource_managers.recorder import (
    CommandEvent,
    EventType,
    FilesystemEvent,
    InMemoryRecorder,
    NullRecorder,
    RecordedEvent,
    Recorder,
    StateChangeEvent,
    TurnEvent,
)

__all__ = [
    # Base resource classes
    "ManagedResource",
    "ResourceError",
    "ResourceManager",
    "ResourceState",
    "managed_resources",
    # Retry configuration
    "RetryConfig",
    "DEFAULT_RETRY_CONFIG",
    "AGGRESSIVE_RETRY_CONFIG",
    "CONSERVATIVE_RETRY_CONFIG",
    "is_transient_error",
    "is_connection_error",
    # Limits configuration
    "SandboxLimits",
    "DEFAULT_SANDBOX_LIMITS",
    "AGGRESSIVE_LIMITS",
    "CONSERVATIVE_LIMITS",
    "merge_limits",
    # Sandbox error types
    "SandboxError",
    "SandboxCreationError",
    "SandboxNotReadyError",
    "SandboxSetupError",
    "SandboxExecutionError",
    "SandboxOOMError",
    "SandboxTimeoutError",
    "CommandTimeoutError",
    "SandboxHealthError",
    "SandboxFailureInfo",
    # Sandbox manager classes
    "BackgroundJob",
    "FileLock",
    "ManagedSandbox",
    "SandboxManager",
    "SetupCallback",
    "ThreadedAsyncSandboxClient",
    "allocate_port",
    "get_available_ports",
    "sandbox_manager",
    # Recorder classes
    "CommandEvent",
    "EventType",
    "FilesystemEvent",
    "InMemoryRecorder",
    "NullRecorder",
    "RecordedEvent",
    "Recorder",
    "StateChangeEvent",
    "TurnEvent",
]
