"""
Example: Real-time rollout monitoring

This script demonstrates how to use the rollout monitor to observe
agent behavior in real-time during MLEB competition rollouts.
"""

import asyncio
from src.rollout_monitor import load_monitored_environment


async def run_monitored_rollout():
    """Run a single rollout with real-time monitoring."""

    # Example 1: Console output only (default)
    print("=" * 80)
    print("EXAMPLE 1: Console monitoring")
    print("=" * 80)

    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="console",  # Pretty console output
        max_turns=5,  # Short demo
        use_gpu=False
    )

    # Run evaluation (this would typically be done by verifiers)
    # For demo purposes, we'll just show the setup
    print("\nEnvironment loaded with monitoring enabled!")
    print("When you run this with an actual agent, you'll see:")
    print("  - Sandbox creation events")
    print("  - Turn-by-turn progress")
    print("  - Assistant messages")
    print("  - Tool calls with arguments")
    print("  - Tool results")
    print("  - Final scores and reports")


async def run_with_file_logging():
    """Run rollout with file logging."""

    print("\n" + "=" * 80)
    print("EXAMPLE 2: File logging")
    print("=" * 80)

    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="file",  # Log to file only
        monitor_file="rollout_logs/spaceship_titanic.log",
        max_turns=5
    )

    print("\nEnvironment loaded with file logging!")
    print("Logs will be saved to: rollout_logs/spaceship_titanic.log")


async def run_with_json_logging():
    """Run rollout with JSON logging for programmatic analysis."""

    print("\n" + "=" * 80)
    print("EXAMPLE 3: JSON logging")
    print("=" * 80)

    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="json",  # JSON format for parsing
        monitor_file="rollout_logs/spaceship_titanic.jsonl",
        max_turns=5
    )

    print("\nEnvironment loaded with JSON logging!")
    print("Logs will be saved to: rollout_logs/spaceship_titanic.jsonl")
    print("Each line is a JSON event that can be parsed programmatically")


async def run_with_all_outputs():
    """Run rollout with all monitoring options enabled."""

    print("\n" + "=" * 80)
    print("EXAMPLE 4: All outputs (console + file)")
    print("=" * 80)

    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="all",  # Console and file
        monitor_file="rollout_logs/spaceship_titanic_all.log",
        monitor_verbose=True,  # Show state updates
        max_turns=5
    )

    print("\nEnvironment loaded with full monitoring!")
    print("You'll see console output AND file logging")


async def run_multiple_competitions():
    """Monitor multiple competitions simultaneously."""

    print("\n" + "=" * 80)
    print("EXAMPLE 5: Multiple competitions")
    print("=" * 80)

    env = load_monitored_environment(
        competition_ids=["spaceship-titanic", "leaf-classification"],
        monitor_output="all",
        monitor_file="rollout_logs/multi_competition.log",
        max_concurrent_sandboxes=2,
        max_turns=5
    )

    print("\nEnvironment loaded for multiple competitions!")
    print("Each sandbox will be monitored with its name in the output")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MLEB Rollout Monitor - Examples")
    print("=" * 80)

    # Run examples
    await run_monitored_rollout()
    await run_with_file_logging()
    await run_with_json_logging()
    await run_with_all_outputs()
    await run_multiple_competitions()

    print("\n" + "=" * 80)
    print("INTEGRATION WITH VERIFIERS")
    print("=" * 80)
    print("""
To use with verifiers evaluation:

1. In your evaluation script:
   ```python
   from src.rollout_monitor import load_monitored_environment

   env = load_monitored_environment(
       competition_ids=["spaceship-titanic"],
       monitor_output="all",
       max_turns=20
   )

   # Use with verifiers as normal
   from verifiers import verifiers
   results = await verifiers.evaluate(agent, env)
   ```

2. Run with vf-eval:
   ```bash
   # Modify mleb.py temporarily to use load_monitored_environment
   # Or create a separate config file
   vf-eval --agent openai/gpt-4 --env mleb --env-args '{"competition_ids": ["spaceship-titanic"]}'
   ```

3. Analysis after rollout:
   ```python
   # Parse JSON logs
   import json

   with open('rollout_log.jsonl') as f:
       events = [json.loads(line) for line in f]

   # Analyze tool usage
   tool_calls = [e for e in events if e['event_type'] == 'tool_call']
   print(f"Total tool calls: {len(tool_calls)}")

   # Find errors
   errors = [e for e in events if e['event_type'] == 'error']
   ```
    """)


if __name__ == "__main__":
    asyncio.run(main())
