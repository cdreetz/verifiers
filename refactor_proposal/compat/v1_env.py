"""
Backward compatibility shims for v1 Environment API.

This module provides the old interface wrapping the new internals,
allowing existing environments to work unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence
from datasets import Dataset

from ..core.types import RolloutInput, RolloutResult, Message, Role
from ..core.protocols import LLMClient
from ..env.base import Environment as NewEnvironment, Context, EnvironmentConfig
from ..scoring.rubric import Rubric as NewRubric


class Environment:
    """
    v1-compatible Environment class.

    This wraps the new Environment with the old interface.
    Existing code like:

        env = SingleTurnEnv(
            dataset=dataset,
            rubric=rubric,
            system_prompt="...",
        )
        outputs = await env.generate(inputs, client, model="gpt-4")

    Will continue to work.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        rubric: Any = None,  # Old Rubric type
        parser: Any = None,
        system_prompt: str | None = None,
        few_shot: list[dict[str, Any]] | None = None,
        sampling_args: dict[str, Any] | None = None,
        message_type: str = "chat",
        **kwargs: Any,
    ):
        self._dataset = dataset
        self._eval_dataset = eval_dataset
        self._rubric = rubric
        self._parser = parser
        self._system_prompt = system_prompt
        self._few_shot = few_shot or []
        self._sampling_args = sampling_args or {}
        self._message_type = message_type
        self._kwargs = kwargs

        # Will be set by load_environment
        self._env_id: str = ""
        self._env_args: dict[str, Any] = {}

        # Create new-style environment internally
        self._new_env: NewEnvironment | None = None

    @property
    def env_id(self) -> str:
        return self._env_id

    @env_id.setter
    def env_id(self, value: str) -> None:
        self._env_id = value

    @property
    def env_args(self) -> dict[str, Any]:
        return self._env_args

    @env_args.setter
    def env_args(self, value: dict[str, Any]) -> None:
        self._env_args = value

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    @property
    def eval_dataset(self) -> Dataset | None:
        return self._eval_dataset

    def _ensure_new_env(self) -> NewEnvironment:
        """Lazily create the new environment."""
        if self._new_env is None:
            self._new_env = self._build_new_env()
        return self._new_env

    def _build_new_env(self) -> NewEnvironment:
        """Build the new-style environment from old config."""
        # Convert old rubric to new rubric
        new_rubric = self._convert_rubric(self._rubric) if self._rubric else None

        return NewEnvironment(
            env_id=self._env_id,
            turn_handler=None,  # Override in subclasses
            scorer=new_rubric,
            parser=self._parser,
            system_prompt=self._system_prompt,
            config=EnvironmentConfig(),
        )

    def _convert_rubric(self, old_rubric: Any) -> NewRubric | None:
        """Convert old-style rubric to new-style."""
        # This would handle the conversion
        # For now, return None and let subclasses handle
        return None

    def _convert_input(self, example: dict[str, Any], idx: int) -> RolloutInput:
        """Convert dataset example to RolloutInput."""
        # Build prompt from example
        prompt_messages: list[Message] = []

        # Add few-shot examples
        for shot in self._few_shot:
            if "user" in shot:
                prompt_messages.append(Message(role=Role.USER, content=shot["user"]))
            if "assistant" in shot:
                prompt_messages.append(
                    Message(role=Role.ASSISTANT, content=shot["assistant"])
                )

        # Add the actual prompt
        if "prompt" in example:
            prompt = example["prompt"]
            if isinstance(prompt, str):
                prompt_messages.append(Message(role=Role.USER, content=prompt))
            elif isinstance(prompt, list):
                for msg in prompt:
                    role = Role(msg.get("role", "user"))
                    prompt_messages.append(
                        Message(role=role, content=msg.get("content", ""))
                    )

        return RolloutInput(
            example_id=example.get("example_id", idx),
            prompt=tuple(prompt_messages),
            answer=example.get("answer", ""),
            task=example.get("task", ""),
            info=example.get("info", {}),
        )

    def _convert_output(self, result: RolloutResult) -> dict[str, Any]:
        """Convert RolloutResult to old-style output dict."""
        return {
            "prompt": [m.to_openai() for m in result.input.prompt],
            "completion": [m.to_openai() for m in result.completion],
            "answer": result.input.answer,
            "task": result.input.task,
            "info": result.input.info,
            "example_id": result.input.example_id,
            "reward": result.reward,
            "metrics": result.metrics,
            "stop_conditions": result.stop_conditions,
            "is_truncated": result.is_truncated,
            "state": self._build_legacy_state(result),
            "metadata": {
                "duration_ms": result.duration_ms,
                "env_id": self._env_id,
            },
        }

    def _build_legacy_state(self, result: RolloutResult) -> dict[str, Any]:
        """Build legacy State dict from RolloutResult."""
        # This provides the dict-like state that old code expects
        return {
            "input": {
                "prompt": [m.to_openai() for m in result.input.prompt],
                "answer": result.input.answer,
                "task": result.input.task,
                "info": result.input.info,
                "example_id": result.input.example_id,
            },
            "trajectory": [
                {
                    "messages": [m.to_openai() for m in step.messages],
                    "step_type": step.step_type,
                }
                for step in result.trajectory
            ],
            "completion": [m.to_openai() for m in result.completion],
            "reward": result.reward,
            "metrics": result.metrics,
            "is_completed": result.is_completed,
            "is_truncated": result.is_truncated,
        }

    async def generate(
        self,
        inputs: Dataset | Sequence[dict[str, Any]] | None = None,
        client: Any = None,
        model: str = "",
        sampling_args: dict[str, Any] | None = None,
        max_concurrent: int = 64,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate rollouts - main v1 API method.

        This wraps the new streaming API with the old batch return.
        """
        # Use dataset if inputs not provided
        if inputs is None:
            inputs = self._dataset

        if inputs is None:
            raise ValueError("No inputs provided and no dataset set")

        # Convert inputs
        rollout_inputs = [
            self._convert_input(ex, i)
            for i, ex in enumerate(inputs)
        ]

        # Build context
        merged_sampling = {**self._sampling_args, **(sampling_args or {})}
        ctx = Context(
            client=client,
            model=model,
            sampling_args=merged_sampling,
        )

        # Run generation
        env = self._ensure_new_env()
        results = await env.generate_all(
            rollout_inputs,
            ctx,
            max_concurrent=max_concurrent,
        )

        # Convert outputs
        return [self._convert_output(r) for r in results]


class SingleTurnEnv(Environment):
    """v1-compatible SingleTurnEnv."""

    pass  # Base Environment already handles single-turn


class MultiTurnEnv(Environment):
    """v1-compatible MultiTurnEnv with env_response hook."""

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Override this to provide environment responses.

        This is the v1 API - subclasses override this method.
        """
        return []

    def _build_new_env(self) -> NewEnvironment:
        """Build new env with turn handler that calls env_response."""

        async def turn_handler(
            result: RolloutResult, ctx: Context
        ) -> list[Message]:
            # Convert to old format for env_response
            messages = [m.to_openai() for m in result.all_messages]
            state = self._build_legacy_state(result)

            # Call the old-style method
            response = await self.env_response(messages, state)

            # Convert back to new format
            return [
                Message(
                    role=Role(msg.get("role", "user")),
                    content=msg.get("content", ""),
                )
                for msg in response
            ]

        new_rubric = self._convert_rubric(self._rubric) if self._rubric else None

        return NewEnvironment(
            env_id=self._env_id,
            turn_handler=turn_handler,
            scorer=new_rubric,
            parser=self._parser,
            system_prompt=self._system_prompt,
            config=EnvironmentConfig(max_turns=10),
        )
