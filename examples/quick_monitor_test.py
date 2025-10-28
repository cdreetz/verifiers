#!/usr/bin/env python3
"""
Quick test of rollout monitoring.

This script demonstrates the monitor without running a full rollout.
It just shows environment setup and the monitoring capabilities.
"""

import asyncio
from src.rollout_monitor import load_monitored_environment, RolloutMonitor


async def test_monitor_setup():
    """Test monitor setup and basic event logging."""

    print("\n" + "="*80)
    print("Testing Rollout Monitor - Setup Only")
    print("="*80 + "\n")

    # Create a monitor
    monitor = RolloutMonitor(output_mode="console", verbose=True)

    # Test event logging
    print("\n1. Testing event logging:\n")

    monitor.log_event("sandbox_created", {
        "name": "test-sandbox-001",
        "competition_id": "spaceship-titanic",
        "gpu_device_id": None,
        "sandbox_prepared": True
    })

    monitor.log_event("turn_start", {
        "turn": 1,
        "max_turns": 20,
        "sandbox_name": "test-sandbox-001"
    })

    monitor.log_event("assistant_message", {
        "content": "I'll start by reading the instructions to understand the task.",
        "turn": 1
    })

    monitor.log_event("tool_call", {
        "tool_name": "bash",
        "arguments": {"command": "cat /home/instructions_obfuscated.txt"},
        "turn": 1
    })

    monitor.log_event("tool_result", {
        "tool_name": "bash",
        "result": "Instructions for the competition...\n\nYour task is to predict...",
        "turn": 1
    })

    monitor.log_event("completion", {
        "sandbox_name": "test-sandbox-001",
        "score": 0.75,
        "competition_report": {
            "any_medal": False,
            "above_median": True,
            "score": 0.75
        },
        "turns_used": 15
    })

    print("\n" + "="*80)
    print("Monitor test complete!")
    print("="*80)

    print("\n2. To use with actual rollouts:\n")
    print("""
    from src.rollout_monitor import load_monitored_environment

    # Simple console monitoring
    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="console"
    )

    # Full monitoring with logs
    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="all",
        monitor_file="my_rollout.log",
        max_turns=20
    )

    # Then use with verifiers as normal
    # results = await verifiers.evaluate(agent, env)
    """)


async def test_environment_load():
    """Test loading monitored environment (setup only, no execution)."""

    print("\n" + "="*80)
    print("Testing Environment Load")
    print("="*80 + "\n")

    try:
        print("Loading monitored environment...")
        env = load_monitored_environment(
            competition_ids=["spaceship-titanic"],
            monitor_output="console",
            max_turns=5
        )
        print("✓ Environment loaded successfully!")
        print(f"  - Dataset size: {len(env.dataset)}")
        print(f"  - Max turns: {env.max_turns}")
        print(f"  - Tools: {len(env._tools)} tools registered")
        print(f"  - Monitor enabled: Yes")

    except Exception as e:
        print(f"✗ Error loading environment: {e}")
        print("  (This is expected if Kaggle credentials are not set up)")


async def main():
    """Run all tests."""
    await test_monitor_setup()
    print("\n")
    await test_environment_load()

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("""
1. See examples/monitor_rollout_example.py for detailed examples
2. See docs/ROLLOUT_MONITORING.md for full documentation
3. Try running a real rollout with monitoring:

   from src.rollout_monitor import load_monitored_environment
   from verifiers import OpenAIAgent

   env = load_monitored_environment(
       competition_ids=["spaceship-titanic"],
       monitor_output="all",
       max_turns=20
   )

   agent = OpenAIAgent(model="gpt-4")
   results = await verifiers.evaluate(agent, env)
    """)


if __name__ == "__main__":
    asyncio.run(main())
