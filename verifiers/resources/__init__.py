"""
Resource management for verifiers.

This module provides abstractions for managing resources like sandboxes,
containers, and connections with support for different allocation modes.
"""

from verifiers.resources.resource_manager import (
    AllocationMode,
    BaseResourceManager,
    ManagerMetrics,
    ResourceHandle,
    ResourceManager,
    ResourceMetrics,
)
from verifiers.resources.sandbox_manager import (
    SandboxConfig,
    SandboxResource,
    SandboxResourceManager,
    SandboxState,
    ThreadedAsyncSandboxClient,
    create_sandbox_manager,
)

__all__ = [
    # Protocol and base
    "ResourceManager",
    "BaseResourceManager",
    "ResourceHandle",
    "ResourceMetrics",
    "ManagerMetrics",
    "AllocationMode",
    # Sandbox
    "SandboxResourceManager",
    "SandboxConfig",
    "SandboxResource",
    "SandboxState",
    "ThreadedAsyncSandboxClient",
    "create_sandbox_manager",
]
