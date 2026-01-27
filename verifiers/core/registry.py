"""
LifecycleRegistry - Explicit lifecycle hook registration without reflection.

This replaces the reflection-based discovery of @stop, @cleanup, and @teardown
decorated methods with explicit registration, making the code more predictable
and easier to trace.

Usage:
    class MyEnv(MultiTurnEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Explicit registration instead of decorator discovery
            self.lifecycle.register_stop(self._game_won, priority=10)
            self.lifecycle.register_stop(self._time_limit, priority=5)
            self.lifecycle.register_cleanup(self._save_logs)

        async def _game_won(self, state: State) -> bool:
            return state.get("game_won", False)

        async def _time_limit(self, state: State) -> bool:
            return state.get("turns", 0) > 100

        async def _save_logs(self, state: State) -> None:
            await save_to_disk(state["logs"])

The registry maintains priority ordering and supports both the new explicit
API and backward compatibility with the existing decorator-based discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from verifiers.types import State

logger = logging.getLogger(__name__)


@dataclass
class StopCondition:
    """
    A registered stop condition with priority ordering.

    Attributes:
        func: Async function that takes State and returns bool
        priority: Higher priority runs first (default 0)
        name: Display name for logging (defaults to func.__name__)
    """

    func: Callable[["State"], Awaitable[bool]]
    priority: int = 0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.func, "__name__", "unknown")

    async def __call__(self, state: "State") -> bool:
        return await self.func(state)


@dataclass
class CleanupHook:
    """
    A registered cleanup hook with priority ordering.

    Attributes:
        func: Async function that takes State and returns None
        priority: Higher priority runs first (default 0)
        name: Display name for logging
    """

    func: Callable[["State"], Awaitable[None]]
    priority: int = 0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.func, "__name__", "unknown")

    async def __call__(self, state: "State") -> None:
        await self.func(state)


@dataclass
class TeardownHook:
    """
    A registered teardown hook with priority ordering.

    Attributes:
        func: Async function that takes no args and returns None
        priority: Higher priority runs first (default 0)
        name: Display name for logging
    """

    func: Callable[[], Awaitable[None]]
    priority: int = 0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.func, "__name__", "unknown")

    async def __call__(self) -> None:
        await self.func()


@dataclass
class LifecycleRegistry:
    """
    Registry for explicit lifecycle hook management.

    This provides a cleaner alternative to the reflection-based discovery
    of decorated methods. Hooks are registered explicitly and stored in
    priority-sorted order.

    Example:
        registry = LifecycleRegistry()
        registry.register_stop(has_error, priority=100)
        registry.register_stop(max_turns_reached, priority=50)
        registry.register_cleanup(save_state)

        # Check stop conditions
        if await registry.check_stop(state):
            print(f"Stopped by: {registry.last_stop_condition}")

        # Run cleanup
        await registry.run_cleanup(state)
    """

    _stop_conditions: list[StopCondition] = field(default_factory=list)
    _cleanup_hooks: list[CleanupHook] = field(default_factory=list)
    _teardown_hooks: list[TeardownHook] = field(default_factory=list)

    # Track which condition triggered stop
    last_stop_condition: Optional[str] = field(default=None)

    def _sort_by_priority(self, items: list) -> None:
        """Sort items by priority (descending) then by name (ascending)."""
        items.sort(key=lambda x: (-x.priority, x.name))

    def register_stop(
        self,
        func: Callable[["State"], Awaitable[bool]],
        priority: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a stop condition.

        Args:
            func: Async function that takes State and returns bool
            priority: Higher priority runs first (default 0)
            name: Optional display name (defaults to func.__name__)
        """
        condition = StopCondition(
            func=func,
            priority=priority,
            name=name or "",
        )
        self._stop_conditions.append(condition)
        self._sort_by_priority(self._stop_conditions)

    def register_cleanup(
        self,
        func: Callable[["State"], Awaitable[None]],
        priority: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a cleanup hook.

        Args:
            func: Async function that takes State and returns None
            priority: Higher priority runs first (default 0)
            name: Optional display name
        """
        hook = CleanupHook(
            func=func,
            priority=priority,
            name=name or "",
        )
        self._cleanup_hooks.append(hook)
        self._sort_by_priority(self._cleanup_hooks)

    def register_teardown(
        self,
        func: Callable[[], Awaitable[None]],
        priority: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a teardown hook.

        Args:
            func: Async function that takes no args and returns None
            priority: Higher priority runs first (default 0)
            name: Optional display name
        """
        hook = TeardownHook(
            func=func,
            priority=priority,
            name=name or "",
        )
        self._teardown_hooks.append(hook)
        self._sort_by_priority(self._teardown_hooks)

    def unregister_stop(self, name: str) -> bool:
        """
        Unregister a stop condition by name.

        Returns True if found and removed, False otherwise.
        """
        for i, condition in enumerate(self._stop_conditions):
            if condition.name == name:
                self._stop_conditions.pop(i)
                return True
        return False

    def unregister_cleanup(self, name: str) -> bool:
        """Unregister a cleanup hook by name."""
        for i, hook in enumerate(self._cleanup_hooks):
            if hook.name == name:
                self._cleanup_hooks.pop(i)
                return True
        return False

    def unregister_teardown(self, name: str) -> bool:
        """Unregister a teardown hook by name."""
        for i, hook in enumerate(self._teardown_hooks):
            if hook.name == name:
                self._teardown_hooks.pop(i)
                return True
        return False

    async def check_stop(self, state: "State") -> bool:
        """
        Check all stop conditions in priority order.

        Returns True if any condition returns True.
        Sets last_stop_condition to the name of the triggering condition.
        """
        self.last_stop_condition = None
        for condition in self._stop_conditions:
            try:
                if await condition(state):
                    self.last_stop_condition = condition.name
                    return True
            except Exception as e:
                logger.error(f"Error in stop condition '{condition.name}': {e}")
                # Continue checking other conditions
        return False

    async def run_cleanup(self, state: "State") -> None:
        """Run all cleanup hooks in priority order."""
        for hook in self._cleanup_hooks:
            try:
                await hook(state)
            except Exception as e:
                logger.error(f"Error in cleanup hook '{hook.name}': {e}")
                # Continue running other hooks

    async def run_teardown(self) -> None:
        """Run all teardown hooks in priority order."""
        for hook in self._teardown_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Error in teardown hook '{hook.name}': {e}")
                # Continue running other hooks

    @property
    def stop_conditions(self) -> list[StopCondition]:
        """Get all registered stop conditions (read-only view)."""
        return list(self._stop_conditions)

    @property
    def cleanup_hooks(self) -> list[CleanupHook]:
        """Get all registered cleanup hooks."""
        return list(self._cleanup_hooks)

    @property
    def teardown_hooks(self) -> list[TeardownHook]:
        """Get all registered teardown hooks."""
        return list(self._teardown_hooks)

    def clear(self) -> None:
        """Clear all registered hooks."""
        self._stop_conditions.clear()
        self._cleanup_hooks.clear()
        self._teardown_hooks.clear()
        self.last_stop_condition = None

    def merge_from(self, other: "LifecycleRegistry") -> None:
        """
        Merge hooks from another registry.

        Useful for composing environments or inheriting defaults.
        """
        for condition in other._stop_conditions:
            self._stop_conditions.append(condition)
        for hook in other._cleanup_hooks:
            self._cleanup_hooks.append(hook)
        for hook in other._teardown_hooks:
            self._teardown_hooks.append(hook)

        # Re-sort after merge
        self._sort_by_priority(self._stop_conditions)
        self._sort_by_priority(self._cleanup_hooks)
        self._sort_by_priority(self._teardown_hooks)

    def __repr__(self) -> str:
        return (
            f"LifecycleRegistry("
            f"stops={len(self._stop_conditions)}, "
            f"cleanups={len(self._cleanup_hooks)}, "
            f"teardowns={len(self._teardown_hooks)})"
        )


def create_default_registry() -> LifecycleRegistry:
    """
    Create a registry with common default stop conditions.

    This can be used as a starting point for new environments.
    """
    return LifecycleRegistry()
