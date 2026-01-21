"""
Clean scoring system with proper async handling.

Key design:
- All reward functions are async (no maybe_await)
- Parameter injection via explicit registration, not inspection magic
- Batching support built-in for group-based rewards
- Clear separation of individual vs group scoring
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any
from enum import Enum

from ..core.types import RolloutResult
from ..core.protocols import Scorer, Parser, Context
from ..async_engine.batch import GroupBatcher, BatchConfig


# Type aliases for reward functions
RewardFunc = Callable[..., Awaitable[float]]
GroupRewardFunc = Callable[..., Awaitable[list[float]]]


class RewardAggregation(Enum):
    """How to combine multiple reward functions."""

    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_MEAN = "weighted_mean"
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"


@dataclass
class RewardComponent:
    """A single reward function with its configuration."""

    name: str
    func: RewardFunc
    weight: float = 1.0
    is_metric_only: bool = False  # If True, doesn't contribute to reward
    required_fields: list[str] = field(default_factory=list)  # Fields to inject


@dataclass
class GroupRewardComponent:
    """A group-based reward function."""

    name: str
    func: GroupRewardFunc
    weight: float = 1.0


class Rubric(Scorer):
    """
    Scores rollouts using composed reward functions.

    Usage:
        async def correct_answer(completion: str, answer: str) -> float:
            return 1.0 if completion.strip() == answer.strip() else 0.0

        async def length_penalty(completion: str) -> float:
            return min(1.0, 100 / len(completion))

        rubric = Rubric(
            components=[
                RewardComponent("correct", correct_answer, weight=1.0),
                RewardComponent("length", length_penalty, weight=0.1, is_metric_only=True),
            ]
        )

        result = await rubric.score(rollout_result, ctx)
    """

    def __init__(
        self,
        components: list[RewardComponent] | None = None,
        group_components: list[GroupRewardComponent] | None = None,
        parser: Parser | None = None,
        aggregation: RewardAggregation = RewardAggregation.WEIGHTED_SUM,
        shared_objects: dict[str, Any] | None = None,
    ):
        self.components = components or []
        self.group_components = group_components or []
        self.parser = parser
        self.aggregation = aggregation
        self.shared_objects = shared_objects or {}

        # For group batching
        self._group_batcher: GroupBatcher | None = None
        if self.group_components:
            self._setup_group_batcher()

    def _setup_group_batcher(self) -> None:
        """Set up batcher for group-based rewards."""

        async def process_group(
            example_id: str, results: list[RolloutResult]
        ) -> list[dict[str, float]]:
            # Run all group reward functions
            group_rewards: dict[str, list[float]] = {}

            for comp in self.group_components:
                kwargs = self._build_group_kwargs(results, comp)
                rewards = await comp.func(**kwargs)
                group_rewards[comp.name] = rewards

            # Combine into per-result dicts
            return [
                {name: rewards[i] for name, rewards in group_rewards.items()}
                for i in range(len(results))
            ]

        self._group_batcher = GroupBatcher(
            group_func=process_group,
            key_func=lambda r: str(r.input.example_id),
            config=BatchConfig(max_batch_size=64, max_wait_ms=100),
        )

    async def score(self, result: RolloutResult, ctx: Context) -> RolloutResult:
        """Score a single rollout."""
        # Parse completion if parser provided
        parsed = {}
        if self.parser and result.completion:
            content = result.completion[-1].content if result.completion else ""
            parsed = self.parser.parse(content)

        # Run individual reward components
        individual_rewards: dict[str, float] = {}
        tasks = []

        for comp in self.components:
            kwargs = self._build_kwargs(result, parsed, comp, ctx)
            tasks.append(self._run_component(comp, kwargs))

        component_results = await asyncio.gather(*tasks, return_exceptions=True)

        for comp, res in zip(self.components, component_results):
            if isinstance(res, Exception):
                individual_rewards[comp.name] = 0.0
                result.metrics[f"{comp.name}_error"] = 1.0
            else:
                individual_rewards[comp.name] = res

        # Get group rewards if applicable
        group_rewards: dict[str, float] = {}
        if self._group_batcher:
            group_rewards = await self._group_batcher.submit(result)

        # Combine all rewards
        all_rewards = {**individual_rewards, **group_rewards}

        # Store all as metrics
        result.metrics.update(all_rewards)

        # Compute final reward (excluding metric-only components)
        reward_components = []
        for comp in self.components:
            if not comp.is_metric_only and comp.name in all_rewards:
                reward_components.append((all_rewards[comp.name], comp.weight))

        for comp in self.group_components:
            if comp.name in all_rewards:
                reward_components.append((all_rewards[comp.name], comp.weight))

        result.reward = self._aggregate(reward_components)
        return result

    async def score_batch(
        self, results: list[RolloutResult], ctx: Context
    ) -> list[RolloutResult]:
        """Score a batch of rollouts."""
        # Run individual scoring in parallel
        tasks = [self.score(r, ctx) for r in results]
        return await asyncio.gather(*tasks)

    async def _run_component(
        self, comp: RewardComponent, kwargs: dict[str, Any]
    ) -> float:
        """Run a single reward component."""
        return await comp.func(**kwargs)

    def _build_kwargs(
        self,
        result: RolloutResult,
        parsed: dict[str, str],
        comp: RewardComponent,
        ctx: Context,
    ) -> dict[str, Any]:
        """Build kwargs for a reward function."""
        kwargs: dict[str, Any] = {}

        # Standard fields
        if "completion" in comp.required_fields or not comp.required_fields:
            content = result.completion[-1].content if result.completion else ""
            kwargs["completion"] = content

        if "answer" in comp.required_fields or not comp.required_fields:
            kwargs["answer"] = result.input.answer

        if "result" in comp.required_fields:
            kwargs["result"] = result

        if "parsed" in comp.required_fields:
            kwargs["parsed"] = parsed

        if "context" in comp.required_fields:
            kwargs["context"] = ctx

        # Shared objects
        for key, value in self.shared_objects.items():
            if key in comp.required_fields:
                kwargs[key] = value

        return kwargs

    def _build_group_kwargs(
        self,
        results: list[RolloutResult],
        comp: GroupRewardComponent,
    ) -> dict[str, Any]:
        """Build kwargs for a group reward function."""
        return {
            "completions": [
                r.completion[-1].content if r.completion else ""
                for r in results
            ],
            "answers": [r.input.answer for r in results],
            "results": results,
        }

    def _aggregate(self, components: list[tuple[float, float]]) -> float:
        """Aggregate reward components."""
        if not components:
            return 0.0

        if self.aggregation == RewardAggregation.WEIGHTED_SUM:
            return sum(r * w for r, w in components)

        elif self.aggregation == RewardAggregation.WEIGHTED_MEAN:
            total_weight = sum(w for _, w in components)
            if total_weight == 0:
                return 0.0
            return sum(r * w for r, w in components) / total_weight

        elif self.aggregation == RewardAggregation.MIN:
            return min(r for r, _ in components)

        elif self.aggregation == RewardAggregation.MAX:
            return max(r for r, _ in components)

        elif self.aggregation == RewardAggregation.PRODUCT:
            result = 1.0
            for r, _ in components:
                result *= r
            return result

        return 0.0


# --- Convenience for creating rubrics ---


def reward(
    name: str | None = None,
    weight: float = 1.0,
    metric_only: bool = False,
):
    """
    Decorator to create a RewardComponent from an async function.

    Usage:
        @reward(weight=1.0)
        async def correct(completion: str, answer: str) -> float:
            return 1.0 if completion == answer else 0.0

        rubric = Rubric(components=[correct])
    """

    def decorator(func: RewardFunc) -> RewardComponent:
        import inspect

        func_name = name or func.__name__
        sig = inspect.signature(func)
        required = list(sig.parameters.keys())

        return RewardComponent(
            name=func_name,
            func=func,
            weight=weight,
            is_metric_only=metric_only,
            required_fields=required,
        )

    return decorator


def group_reward(name: str | None = None, weight: float = 1.0):
    """Decorator for group-based reward functions."""

    def decorator(func: GroupRewardFunc) -> GroupRewardComponent:
        return GroupRewardComponent(
            name=name or func.__name__,
            func=func,
            weight=weight,
        )

    return decorator
