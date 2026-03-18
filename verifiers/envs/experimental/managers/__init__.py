"""Experimental resource management abstractions for verifiers."""

from verifiers.envs.experimental.managers.resource_manager import (
    ManagedResource,
    ResourceError,
    ResourceManager,
    ResourceState,
)
from verifiers.envs.experimental.managers.sandbox_manager import (
    ManagedSandbox,
    SandboxManager,
)

__all__ = [
    # Base classes
    "ManagedResource",
    "ResourceError",
    "ResourceManager",
    "ResourceState",
    # Sandbox classes
    "ManagedSandbox",
    "SandboxManager",
]
