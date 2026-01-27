"""
Tests for the new core abstractions: ExecutionContext, LifecycleRegistry, ResourcePool.
"""

import asyncio

import pytest

import verifiers as vf
from verifiers.core import (
    CleanupHook,
    ExecutionContext,
    LifecycleRegistry,
    StopCondition,
    TeardownHook,
)
from verifiers.utils.async_utils import ResourcePool, create_semaphore


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_create_context_basic(self):
        """Test basic context creation."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
            sampling_args={"temperature": 0.7},
        )

        assert ctx.client == mock_client
        assert ctx.model == "gpt-4"
        assert ctx.sampling_args == {"temperature": 0.7}
        assert ctx.gen_concurrency is None
        assert ctx.score_concurrency is None

    def test_context_with_concurrency(self):
        """Test context with concurrency limits."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
            gen_concurrency=32,
            score_concurrency=16,
        )

        assert ctx.gen_concurrency == 32
        assert ctx.score_concurrency == 16
        # Semaphores should be created
        assert ctx._gen_sem is not None
        assert ctx._score_sem is not None

    @pytest.mark.asyncio
    async def test_generation_limit(self):
        """Test generation concurrency limit."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
            gen_concurrency=2,
        )

        # Track concurrent access
        max_concurrent = 0
        current_concurrent = 0

        async def task():
            nonlocal max_concurrent, current_concurrent
            async with ctx.generation_limit():
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1

        # Run 5 tasks with concurrency limit of 2
        await asyncio.gather(*[task() for _ in range(5)])

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_scoring_limit(self):
        """Test scoring concurrency limit."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
            score_concurrency=3,
        )

        max_concurrent = 0
        current_concurrent = 0

        async def task():
            nonlocal max_concurrent, current_concurrent
            async with ctx.scoring_limit():
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1

        await asyncio.gather(*[task() for _ in range(10)])

        assert max_concurrent <= 3

    def test_with_sampling_args(self):
        """Test creating new context with updated sampling args."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
            sampling_args={"temperature": 0.7},
            gen_concurrency=32,
        )

        new_ctx = ctx.with_sampling_args(temperature=0.9, top_p=0.95)

        # Original unchanged
        assert ctx.sampling_args == {"temperature": 0.7}

        # New context has merged args
        assert new_ctx.sampling_args == {"temperature": 0.9, "top_p": 0.95}
        assert new_ctx.client == ctx.client
        assert new_ctx.model == ctx.model
        # Semaphores should be shared
        assert new_ctx._gen_sem is ctx._gen_sem

    def test_with_model(self):
        """Test creating new context with different model."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
        )

        new_ctx = ctx.with_model("gpt-4-turbo")

        assert ctx.model == "gpt-4"
        assert new_ctx.model == "gpt-4-turbo"

    def test_to_dict(self):
        """Test serialization to dict."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        ctx = ExecutionContext(
            client=mock_client,
            model="gpt-4",
            sampling_args={"temperature": 0.7},
            gen_concurrency=32,
            score_concurrency=16,
            interleaved_rollouts=True,
        )

        data = ctx.to_dict()

        assert data["model"] == "gpt-4"
        assert data["sampling_args"] == {"temperature": 0.7}
        assert data["gen_concurrency"] == 32
        assert data["score_concurrency"] == 16
        assert data["interleaved_rollouts"] is True
        # Client should not be in dict (non-serializable)
        assert "client" not in data


class TestLifecycleRegistry:
    """Tests for LifecycleRegistry."""

    def test_create_registry(self):
        """Test basic registry creation."""
        registry = LifecycleRegistry()

        assert len(registry.stop_conditions) == 0
        assert len(registry.cleanup_hooks) == 0
        assert len(registry.teardown_hooks) == 0

    def test_register_stop(self):
        """Test registering stop conditions."""
        registry = LifecycleRegistry()

        async def condition1(state):
            return False

        async def condition2(state):
            return True

        registry.register_stop(condition1, priority=5)
        registry.register_stop(condition2, priority=10)

        assert len(registry.stop_conditions) == 2
        # Higher priority first
        assert registry.stop_conditions[0].name == "condition2"
        assert registry.stop_conditions[1].name == "condition1"

    @pytest.mark.asyncio
    async def test_check_stop(self):
        """Test checking stop conditions."""
        registry = LifecycleRegistry()

        async def never_stop(state):
            return False

        async def always_stop(state):
            return True

        registry.register_stop(never_stop, priority=10)
        registry.register_stop(always_stop, priority=5)

        state = vf.State(input={"prompt": [], "example_id": 0, "task": "test"})

        result = await registry.check_stop(state)

        assert result is True
        assert registry.last_stop_condition == "always_stop"

    @pytest.mark.asyncio
    async def test_check_stop_all_false(self):
        """Test when no stop conditions trigger."""
        registry = LifecycleRegistry()

        async def never_stop1(state):
            return False

        async def never_stop2(state):
            return False

        registry.register_stop(never_stop1)
        registry.register_stop(never_stop2)

        state = vf.State(input={"prompt": [], "example_id": 0, "task": "test"})

        result = await registry.check_stop(state)

        assert result is False
        assert registry.last_stop_condition is None

    def test_register_cleanup(self):
        """Test registering cleanup hooks."""
        registry = LifecycleRegistry()

        async def cleanup1(state):
            pass

        async def cleanup2(state):
            pass

        registry.register_cleanup(cleanup1, priority=0)
        registry.register_cleanup(cleanup2, priority=10)

        assert len(registry.cleanup_hooks) == 2
        # Higher priority first
        assert registry.cleanup_hooks[0].name == "cleanup2"

    @pytest.mark.asyncio
    async def test_run_cleanup(self):
        """Test running cleanup hooks."""
        registry = LifecycleRegistry()
        cleanup_order = []

        async def cleanup1(state):
            cleanup_order.append("cleanup1")

        async def cleanup2(state):
            cleanup_order.append("cleanup2")

        registry.register_cleanup(cleanup1, priority=5)
        registry.register_cleanup(cleanup2, priority=10)

        state = vf.State(input={"prompt": [], "example_id": 0, "task": "test"})

        await registry.run_cleanup(state)

        # Higher priority runs first
        assert cleanup_order == ["cleanup2", "cleanup1"]

    def test_unregister_stop(self):
        """Test unregistering stop conditions."""
        registry = LifecycleRegistry()

        async def condition(state):
            return True

        registry.register_stop(condition, name="my_condition")
        assert len(registry.stop_conditions) == 1

        result = registry.unregister_stop("my_condition")
        assert result is True
        assert len(registry.stop_conditions) == 0

        # Try to unregister non-existent
        result = registry.unregister_stop("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clearing all hooks."""
        registry = LifecycleRegistry()

        async def stop_cond(state):
            return True

        async def cleanup(state):
            pass

        async def teardown():
            pass

        registry.register_stop(stop_cond)
        registry.register_cleanup(cleanup)
        registry.register_teardown(teardown)

        assert len(registry.stop_conditions) == 1
        assert len(registry.cleanup_hooks) == 1
        assert len(registry.teardown_hooks) == 1

        registry.clear()

        assert len(registry.stop_conditions) == 0
        assert len(registry.cleanup_hooks) == 0
        assert len(registry.teardown_hooks) == 0

    def test_merge_from(self):
        """Test merging registries."""
        registry1 = LifecycleRegistry()
        registry2 = LifecycleRegistry()

        async def stop1(state):
            return True

        async def stop2(state):
            return False

        registry1.register_stop(stop1, priority=5)
        registry2.register_stop(stop2, priority=10)

        registry1.merge_from(registry2)

        assert len(registry1.stop_conditions) == 2
        # After merge and re-sort, priority 10 should be first
        assert registry1.stop_conditions[0].name == "stop2"


class TestResourcePool:
    """Tests for ResourcePool."""

    def test_create_pool(self):
        """Test basic pool creation."""
        pool = ResourcePool(gen_limit=32, score_limit=16)

        assert pool.gen_limit == 32
        assert pool.score_limit == 16

    @pytest.mark.asyncio
    async def test_generation_concurrency(self):
        """Test generation concurrency control."""
        pool = ResourcePool(gen_limit=2)

        max_concurrent = 0
        current_concurrent = 0

        async def task():
            nonlocal max_concurrent, current_concurrent
            async with pool.generation():
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1

        await asyncio.gather(*[task() for _ in range(5)])

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_scoring_concurrency(self):
        """Test scoring concurrency control."""
        pool = ResourcePool(score_limit=3)

        max_concurrent = 0
        current_concurrent = 0

        async def task():
            nonlocal max_concurrent, current_concurrent
            async with pool.scoring():
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1

        await asyncio.gather(*[task() for _ in range(10)])

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test usage statistics tracking."""
        pool = ResourcePool(gen_limit=5, score_limit=5)

        async def gen_task():
            async with pool.generation():
                await asyncio.sleep(0.001)

        async def score_task():
            async with pool.scoring():
                await asyncio.sleep(0.001)

        await asyncio.gather(*[gen_task() for _ in range(3)])
        await asyncio.gather(*[score_task() for _ in range(2)])

        stats = pool.stats
        assert stats["gen_acquired"] == 3
        assert stats["gen_released"] == 3
        assert stats["score_acquired"] == 2
        assert stats["score_released"] == 2

    def test_reset_stats(self):
        """Test resetting statistics."""
        pool = ResourcePool(gen_limit=5)
        pool._stats["gen_acquired"] = 10
        pool._stats["gen_released"] = 10

        pool.reset_stats()

        assert pool.stats["gen_acquired"] == 0
        assert pool.stats["gen_released"] == 0


class TestCreateSemaphore:
    """Tests for create_semaphore utility."""

    def test_with_limit(self):
        """Test creating semaphore with limit."""
        sem = create_semaphore(10)
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 10

    def test_without_limit(self):
        """Test creating null context without limit."""
        sem = create_semaphore(None)
        # Should be a NullAsyncContext (or similar no-op)
        assert sem is not None

    def test_with_zero_limit(self):
        """Test creating null context with zero limit."""
        sem = create_semaphore(0)
        # Should be a NullAsyncContext (or similar no-op)
        assert sem is not None


class TestStateDeprecationWarnings:
    """Tests for State deprecation warnings."""

    def test_suppress_warnings_class_method(self):
        """Test class method to suppress warnings."""
        # Suppress warnings
        vf.State.suppress_warnings(True)
        assert vf.State._warnings_suppressed is True

        # Enable warnings
        vf.State.suppress_warnings(False)
        assert vf.State._warnings_suppressed is False

        # Reset to suppressed for other tests
        vf.State.suppress_warnings(True)

    def test_explicit_accessors(self):
        """Test explicit accessor methods don't warn."""
        state = vf.State(
            input={"prompt": [{"role": "user", "content": "test"}], "example_id": 42, "task": "test", "answer": "42"}
        )

        # These should work without warnings
        assert state.get_prompt() == [{"role": "user", "content": "test"}]
        assert state.get_example_id() == 42
        assert state.get_task() == "test"
        assert state.get_answer() == "42"
        assert state.get_info() == {}

    def test_to_dict(self):
        """Test to_dict serialization."""
        state = vf.State(
            input={"prompt": [], "example_id": 0, "task": "test"},
            reward=0.5,
            is_completed=True,
        )

        data = state.to_dict()

        assert data["input"] == {"prompt": [], "example_id": 0, "task": "test"}
        assert data["reward"] == 0.5
        assert data["is_completed"] is True

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "input": {"prompt": [], "example_id": 0, "task": "test"},
            "reward": 0.75,
        }

        state = vf.State.from_dict(data)

        assert state["input"]["example_id"] == 0
        assert state["reward"] == 0.75
