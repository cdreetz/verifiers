"""Core types with clear schemas - no magic forwarding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable message type."""
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None

    def to_openai(self) -> dict[str, Any]:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return d


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Immutable tool call."""
    id: str
    name: str
    arguments: str  # JSON string

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Single step in a multi-turn trajectory."""
    messages: tuple[Message, ...]  # Messages added this step
    token_count: int
    step_type: Literal["assistant", "environment", "tool"]


@dataclass(slots=True)
class RolloutInput:
    """Input to a rollout - explicit fields, no dict inheritance."""
    example_id: int
    prompt: tuple[Message, ...]
    answer: str
    task: str = ""
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutResult:
    """Output of a rollout - mutable during execution, frozen after."""
    input: RolloutInput
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    completion: tuple[Message, ...] = ()

    # Scoring (filled in by rubric)
    reward: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    stop_conditions: dict[str, bool] = field(default_factory=dict)

    # Status
    is_completed: bool = False
    is_truncated: bool = False
    error: Exception | None = None

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def all_messages(self) -> tuple[Message, ...]:
        """Flatten trajectory into message sequence."""
        msgs = list(self.input.prompt)
        for step in self.trajectory:
            msgs.extend(step.messages)
        return tuple(msgs)

    @property
    def assistant_messages(self) -> tuple[Message, ...]:
        """Just the assistant responses."""
        return tuple(m for m in self.all_messages if m.role == Role.ASSISTANT)
