import os
import sys
import warnings
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
)

from verifiers.errors import Error

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# openai types
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,  # noqa: F401
)
from openai.types.chat.chat_completion_role import ChatCompletionRole  # noqa: F401
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.completion import Completion
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)
from pydantic import BaseModel

# Environment variable to control deprecation warnings for State forwarding
# Set VF_SUPPRESS_STATE_WARNINGS=1 to disable these warnings
_SUPPRESS_STATE_WARNINGS = os.environ.get("VF_SUPPRESS_STATE_WARNINGS", "0") == "1"

# typing aliases
ChatMessage = ChatCompletionMessageParam
MessageType = Literal["chat", "completion"]
ModelResponse = Completion | ChatCompletion | None

ChatMessages = list[ChatMessage]
Message = str | ChatMessage

Messages = str | list[ChatMessage]
Info = dict[str, Any]

SamplingArgs = dict[str, Any]
IndividualRewardFunc = Callable[..., float | Awaitable[float]]
GroupRewardFunc = Callable[..., list[float] | Awaitable[list[float]]]
RewardFunc = IndividualRewardFunc | GroupRewardFunc


class TrajectoryStepTokens(TypedDict):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool
    is_truncated: bool


class TrajectoryStep(TypedDict):
    prompt: Messages
    completion: Messages
    response: ModelResponse
    tokens: TrajectoryStepTokens | None
    reward: float | None
    advantage: float | None
    is_truncated: bool
    trajectory_id: str
    extras: dict[str, Any]


class BaseRolloutInput(TypedDict):
    prompt: Messages
    example_id: int
    task: str


class RolloutInput(BaseRolloutInput, total=False):
    # required: prompt, example_id, task
    # optional: answer, info
    answer: str
    info: Info


class RolloutTiming(TypedDict, total=False):
    start_time: float
    generation_ms: float
    scoring_ms: float
    total_ms: float


class State(dict):
    """
    Rollout state container.

    State is a dict subclass that stores all data for a single rollout.
    For backward compatibility, it supports implicit forwarding of INPUT_FIELDS
    to state["input"], but this behavior is deprecated.

    Preferred usage (explicit):
        state["input"]["prompt"]  # explicit access to input fields
        state.get_prompt()        # explicit accessor method

    Deprecated usage (implicit forwarding):
        state["prompt"]           # implicitly forwards to state["input"]["prompt"]

    To suppress deprecation warnings, set VF_SUPPRESS_STATE_WARNINGS=1 in your
    environment, or call State.suppress_warnings().
    """

    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]

    # Class-level flag to control deprecation warnings
    _warnings_suppressed: bool = _SUPPRESS_STATE_WARNINGS
    _warning_shown: set = set()  # Track which warnings have been shown

    # Type hints for IDE support (these are stored in dict, not as attributes)
    input: RolloutInput
    client: AsyncOpenAI
    model: str
    sampling_args: SamplingArgs | None
    # created during rollout
    is_completed: bool
    is_truncated: bool
    stop_condition: str | None
    oai_tools: list[ChatCompletionToolParam]
    trajectory: list[TrajectoryStep]
    completion: Messages | None
    reward: float | None
    advantage: float | None
    metrics: dict[str, float] | None
    timing: RolloutTiming | None
    error: Error | None

    @classmethod
    def suppress_warnings(cls, suppress: bool = True) -> None:
        """
        Suppress or enable deprecation warnings for implicit forwarding.

        Args:
            suppress: If True, suppress warnings. If False, enable them.
        """
        cls._warnings_suppressed = suppress

    def _warn_forwarding(self, key: str) -> None:
        """Emit a deprecation warning for implicit forwarding."""
        if self._warnings_suppressed:
            return
        # Only warn once per key to avoid spam
        if key in State._warning_shown:
            return
        State._warning_shown.add(key)
        warnings.warn(
            f"Implicit access to state['{key}'] is deprecated. "
            f"Use state['input']['{key}'] or state.get_{key}() instead. "
            f"Set VF_SUPPRESS_STATE_WARNINGS=1 to suppress this warning.",
            DeprecationWarning,
            stacklevel=3,
        )

    def __getitem__(self, key: str) -> Any:
        # forward to input if exists (with deprecation warning)
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                self._warn_forwarding(key)
                return input_obj[key]
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        # forward to input if exists (with deprecation warning)
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                self._warn_forwarding(key)
                input_obj[key] = value
                return
        super().__setitem__(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    # Explicit accessor methods (preferred over implicit forwarding)
    def get_prompt(self) -> Messages:
        """Get the prompt from input (explicit accessor)."""
        return self["input"]["prompt"]

    def get_answer(self) -> Optional[str]:
        """Get the answer from input (explicit accessor)."""
        return self["input"].get("answer")

    def get_task(self) -> str:
        """Get the task from input (explicit accessor)."""
        return self["input"].get("task", "default")

    def get_info(self) -> dict:
        """Get the info dict from input (explicit accessor)."""
        return self["input"].get("info", {})

    def get_example_id(self) -> int:
        """Get the example_id from input (explicit accessor)."""
        return self["input"]["example_id"]

    def get_input(self) -> RolloutInput:
        """Get the full input object (explicit accessor)."""
        return self["input"]

    # Serialization helpers
    def to_dict(self) -> dict[str, Any]:
        """
        Convert State to a plain dict for serialization.

        This flattens the structure and handles non-serializable objects.
        """
        result = dict(self)
        # Remove non-serializable objects
        result.pop("client", None)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        """Create a State from a dict."""
        return cls(data)


# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]


class GenerateMetadata(TypedDict):
    """Pydantic model for generation metadata."""

    env_id: str
    env_args: dict
    model: str
    base_url: str
    num_examples: int
    rollouts_per_example: int
    sampling_args: SamplingArgs
    date: str
    time_ms: float
    avg_reward: float
    avg_metrics: dict[str, float]
    state_columns: list[str]
    path_to_save: Path


class GenerateOutputs(TypedDict):
    """TypedDict for generation outputs."""

    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    task: list[str]
    info: list[Info]
    example_id: list[int]
    reward: list[float]
    metrics: dict[str, list[float]]
    stop_conditions: list[str | None]
    is_truncated: list[bool]
    metadata: GenerateMetadata


class RolloutScore(TypedDict):
    """TypedDict for rollout scores."""

    reward: float
    metrics: dict[str, float]


class RolloutScores(TypedDict):
    """TypedDict for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]]


class ProcessedOutputs(TypedDict):
    """TypedDict for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]
    is_truncated: list[bool]


Endpoint = TypedDict("Endpoint", {"key": str, "url": str, "model": str})
Endpoints = dict[str, Endpoint]


class ClientConfig(BaseModel):
    """Pydantic model for OpenAI client configuration."""

    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    timeout: float = 3600.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = {}


class EvalConfig(BaseModel):
    """Pydantic model for evaluation configuration."""

    # environment
    env_id: str
    env_args: dict
    env_dir_path: str
    # evaluation
    model: str
    client_config: ClientConfig
    sampling_args: SamplingArgs
    num_examples: int
    rollouts_per_example: int
    max_concurrent: int
    max_concurrent_generation: int | None = None
    max_concurrent_scoring: int | None = None
    independent_scoring: bool = False
    extra_env_kwargs: dict = {}
    # logging
    print_results: bool = False
    verbose: bool = False
    # saving
    state_columns: list[str] | None = None
    save_results: bool = False
    save_every: int = -1
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None
