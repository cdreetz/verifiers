"""Recorder abstraction for tracking resource state changes over time.

Inspired by frontier-evals' recorder pattern. Records events that occur
during resource lifecycle, particularly useful for:
- Debugging agent behavior
- Understanding what commands were executed
- Tracking filesystem changes turn-by-turn
- Analyzing resource utilization patterns

TODO: Complete implementation with:
- Filesystem diff recording
- Export to various formats (JSON, JSONL, etc.)
- Integration with observability tools
- Compression for long rollouts
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, Self


class EventType(Enum):
    """Types of events that can be recorded."""

    # Resource lifecycle
    RESOURCE_CREATED = auto()
    RESOURCE_READY = auto()
    RESOURCE_ERROR = auto()
    RESOURCE_DESTROYED = auto()

    # Command execution
    COMMAND_START = auto()
    COMMAND_SUCCESS = auto()
    COMMAND_TIMEOUT = auto()
    COMMAND_ERROR = auto()

    # Filesystem (TODO)
    FILESYSTEM_SNAPSHOT = auto()
    FILESYSTEM_DIFF = auto()

    # Agent turns
    TURN_START = auto()
    TURN_END = auto()


@dataclass(slots=True)
class RecordedEvent:
    """Base class for all recorded events."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    resource_id: str | None = None
    rollout_id: str | None = None
    turn_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CommandEvent(RecordedEvent):
    """Records a command execution."""

    command: str = ""
    working_dir: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None

    @classmethod
    def start(
        cls,
        command: str,
        resource_id: str | None = None,
        rollout_id: str | None = None,
        turn_number: int | None = None,
        working_dir: str | None = None,
    ) -> Self:
        """Create a command start event."""
        return cls(
            event_type=EventType.COMMAND_START,
            command=command,
            resource_id=resource_id,
            rollout_id=rollout_id,
            turn_number=turn_number,
            working_dir=working_dir,
        )

    @classmethod
    def success(
        cls,
        command: str,
        stdout: str,
        stderr: str | None = None,
        exit_code: int = 0,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a command success event."""
        return cls(
            event_type=EventType.COMMAND_SUCCESS,
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        command: str,
        error_message: str,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a command error event."""
        return cls(
            event_type=EventType.COMMAND_ERROR,
            command=command,
            duration_seconds=duration_seconds,
            metadata={"error": error_message},
            **kwargs,
        )


@dataclass(slots=True)
class StateChangeEvent(RecordedEvent):
    """Records a resource state transition."""

    old_state: str | None = None
    new_state: str = ""
    reason: str | None = None


@dataclass(slots=True)
class FilesystemEvent(RecordedEvent):
    """Records filesystem state or changes.

    TODO: Implement filesystem diffing:
    - Snapshot: capture full directory listing with sizes/hashes
    - Diff: compute changes since last snapshot
    - Consider using git-like tree hashing for efficiency
    """

    # For snapshots: list of (path, size, mtime, hash) tuples
    # For diffs: list of (action, path, details) tuples
    changes: list[dict[str, Any]] = field(default_factory=list)
    base_path: str = "/workspace"

    @classmethod
    def snapshot(
        cls,
        changes: list[dict[str, Any]],
        base_path: str = "/workspace",
        **kwargs: Any,
    ) -> Self:
        """Create a filesystem snapshot event."""
        return cls(
            event_type=EventType.FILESYSTEM_SNAPSHOT,
            changes=changes,
            base_path=base_path,
            **kwargs,
        )

    @classmethod
    def diff(
        cls,
        changes: list[dict[str, Any]],
        base_path: str = "/workspace",
        **kwargs: Any,
    ) -> Self:
        """Create a filesystem diff event."""
        return cls(
            event_type=EventType.FILESYSTEM_DIFF,
            changes=changes,
            base_path=base_path,
            **kwargs,
        )


@dataclass(slots=True)
class TurnEvent(RecordedEvent):
    """Records the start or end of an agent turn."""

    turn_number: int = 0
    action_count: int = 0  # Number of actions in this turn (for TURN_END)


class Recorder(Protocol):
    """Protocol for event recorders.

    Implementations can store events in memory, write to disk,
    send to observability systems, etc.
    """

    def record(self, event: RecordedEvent) -> None:
        """Record a single event."""
        ...

    def get_events(
        self,
        rollout_id: str | None = None,
        resource_id: str | None = None,
        event_type: EventType | None = None,
    ) -> list[RecordedEvent]:
        """Query recorded events with optional filters."""
        ...

    def clear(self, rollout_id: str | None = None) -> None:
        """Clear recorded events, optionally for a specific rollout."""
        ...


class InMemoryRecorder:
    """Simple in-memory recorder for testing and short runs.

    Events are stored in a list and can be queried/filtered.
    Not suitable for long runs with many events (memory growth).
    """

    def __init__(self, max_events: int | None = None):
        """Initialize recorder.

        Args:
            max_events: Optional limit on stored events (oldest dropped first).
        """
        self.events: list[RecordedEvent] = []
        self.max_events = max_events

    def record(self, event: RecordedEvent) -> None:
        """Record an event."""
        self.events.append(event)

        # Trim if over limit
        if self.max_events and len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_events(
        self,
        rollout_id: str | None = None,
        resource_id: str | None = None,
        event_type: EventType | None = None,
        turn_number: int | None = None,
    ) -> list[RecordedEvent]:
        """Get events matching filters."""
        result = self.events

        if rollout_id is not None:
            result = [e for e in result if e.rollout_id == rollout_id]
        if resource_id is not None:
            result = [e for e in result if e.resource_id == resource_id]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if turn_number is not None:
            result = [e for e in result if e.turn_number == turn_number]

        return result

    def get_commands(self, rollout_id: str | None = None) -> list[CommandEvent]:
        """Get all command events for a rollout."""
        events = self.get_events(rollout_id=rollout_id)
        return [e for e in events if isinstance(e, CommandEvent)]

    def get_turn_events(self, rollout_id: str, turn_number: int) -> list[RecordedEvent]:
        """Get all events for a specific turn."""
        return self.get_events(rollout_id=rollout_id, turn_number=turn_number)

    def clear(self, rollout_id: str | None = None) -> None:
        """Clear events."""
        if rollout_id is None:
            self.events.clear()
        else:
            self.events = [e for e in self.events if e.rollout_id != rollout_id]

    def to_dicts(self, rollout_id: str | None = None) -> list[dict[str, Any]]:
        """Export events as list of dicts (for JSON serialization)."""
        events = self.get_events(rollout_id=rollout_id)
        result = []
        for event in events:
            d = {
                "event_type": event.event_type.name,
                "timestamp": event.timestamp,
                "resource_id": event.resource_id,
                "rollout_id": event.rollout_id,
                "turn_number": event.turn_number,
                "metadata": event.metadata,
            }
            # Add type-specific fields
            if isinstance(event, CommandEvent):
                d.update({
                    "command": event.command,
                    "working_dir": event.working_dir,
                    "stdout": event.stdout,
                    "stderr": event.stderr,
                    "exit_code": event.exit_code,
                    "duration_seconds": event.duration_seconds,
                })
            elif isinstance(event, StateChangeEvent):
                d.update({
                    "old_state": event.old_state,
                    "new_state": event.new_state,
                    "reason": event.reason,
                })
            elif isinstance(event, FilesystemEvent):
                d.update({
                    "changes": event.changes,
                    "base_path": event.base_path,
                })
            elif isinstance(event, TurnEvent):
                d.update({
                    "action_count": event.action_count,
                })
            result.append(d)
        return result


class NullRecorder:
    """No-op recorder that discards all events.

    Useful as a default when recording is not needed.
    """

    def record(self, event: RecordedEvent) -> None:
        """Discard the event."""
        pass

    def get_events(self, **kwargs: Any) -> list[RecordedEvent]:
        """Always returns empty list."""
        return []

    def clear(self, rollout_id: str | None = None) -> None:
        """No-op."""
        pass


# TODO: Implement these recorders
#
# class JSONLRecorder(Recorder):
#     """Appends events as JSONL to a file. Good for long runs."""
#     pass
#
# class RolloutRecorder:
#     """Groups events by rollout with turn-based organization.
#
#     Provides convenient access patterns like:
#     - Get all commands for turn N
#     - Get filesystem diff between turns
#     - Summarize a rollout's resource usage
#     """
#     pass
#
# class FilesystemTracker:
#     """Utility for capturing filesystem state and computing diffs.
#
#     Methods:
#     - snapshot(sandbox_id, path) -> FilesystemEvent
#     - diff(sandbox_id, path, since_event) -> FilesystemEvent
#     """
#     pass
