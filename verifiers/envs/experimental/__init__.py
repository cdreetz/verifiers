"""Experimental environments and resource managers."""

# Use lazy imports to avoid circular dependencies
# These are imported when accessed via __getattr__

from . import resource_managers as resource_managers

__all__ = [
    # Environments
    "CliAgentEnv",
    "HarborEnv",
    "NewCliAgentEnv",
    "NewHarborEnv",
    "NewSandboxEnv",
    # Resource managers (re-exported from resource_managers submodule)
    "resource_managers",
    # Convenience re-exports from resource_managers
    "RetryConfig",
    "SandboxFailureInfo",
    "SandboxManager",
    "BackgroundJob",
]


def __getattr__(name: str):
    if name == "CliAgentEnv":
        from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
        return CliAgentEnv
    elif name == "HarborEnv":
        from verifiers.envs.experimental.harbor_env import HarborEnv
        return HarborEnv
    elif name == "NewCliAgentEnv":
        from verifiers.envs.experimental.new_cli_agent_env import NewCliAgentEnv
        return NewCliAgentEnv
    elif name == "NewHarborEnv":
        from verifiers.envs.experimental.new_harbor_env import NewHarborEnv
        return NewHarborEnv
    elif name == "NewSandboxEnv":
        from verifiers.envs.experimental.new_sandbox_env import NewSandboxEnv
        return NewSandboxEnv
    # Convenience re-exports from resource_managers
    elif name == "RetryConfig":
        from verifiers.envs.experimental.resource_managers import RetryConfig
        return RetryConfig
    elif name == "SandboxFailureInfo":
        from verifiers.envs.experimental.resource_managers import SandboxFailureInfo
        return SandboxFailureInfo
    elif name == "SandboxManager":
        from verifiers.envs.experimental.resource_managers import SandboxManager
        return SandboxManager
    elif name == "BackgroundJob":
        from verifiers.envs.experimental.resource_managers import BackgroundJob
        return BackgroundJob
    raise AttributeError(f"module 'verifiers.envs.experimental' has no attribute '{name}'")
