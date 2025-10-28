"""
Real-time rollout monitoring for MLEB environments.

This module provides a wrapper around MLEBenchEnv that captures and displays
agent messages, tool calls, and state updates in real-time without modifying
the verifiers library.

Usage:
    from src.rollout_monitor import load_monitored_environment

    env = load_monitored_environment(
        competition_ids=["spaceship-titanic"],
        monitor_output="console",  # or "file", "json", "all"
        monitor_file="rollout_log.txt"
    )
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.mleb import MLEBenchEnv, load_environment
from verifiers.types import Messages, State


class RolloutMonitor:
    """Monitors and logs rollout events in real-time."""

    def __init__(
        self,
        output_mode: str = "console",  # "console", "file", "json", "all"
        log_file: Optional[str] = None,
        verbose: bool = True
    ):
        self.output_mode = output_mode
        self.log_file = log_file
        self.verbose = verbose
        self.console = Console()

        if self.output_mode in ["file", "json", "all"] and not log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"rollout_log_{timestamp}.txt"

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to all configured outputs."""
        timestamp = datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data
        }

        if self.output_mode in ["console", "all"]:
            self._log_to_console(event_type, data, timestamp)

        if self.output_mode in ["file", "all"]:
            self._log_to_file(event, format="text")

        if self.output_mode == "json":
            self._log_to_file(event, format="json")

    def _log_to_console(self, event_type: str, data: Dict[str, Any], timestamp: str):
        """Pretty print events to console using rich."""

        if event_type == "sandbox_created":
            self.console.print(Panel(
                f"[bold green]Sandbox Created[/bold green]\n"
                f"Name: {data['name']}\n"
                f"Competition: {data['competition_id']}\n"
                f"GPU: {data.get('gpu_device_id', 'None')}",
                title=f"[dim]{timestamp}[/dim]",
                border_style="green"
            ))

        elif event_type == "turn_start":
            self.console.print(f"\n[bold blue]{'='*60}[/bold blue]")
            self.console.print(f"[bold blue]Turn {data['turn']}/{data['max_turns']} - {data['sandbox_name']}[/bold blue]")
            self.console.print(f"[bold blue]{'='*60}[/bold blue]\n")

        elif event_type == "assistant_message":
            content = data.get('content', '')
            if content:
                self.console.print(Panel(
                    content,
                    title="[cyan]Assistant Message[/cyan]",
                    border_style="cyan"
                ))

        elif event_type == "tool_call":
            table = Table(title="Tool Call", show_header=True, header_style="bold magenta")
            table.add_column("Tool", style="magenta")
            table.add_column("Arguments", style="white")

            tool_name = data.get('tool_name', 'unknown')
            args = data.get('arguments', {})

            # Format args nicely, truncate if too long
            args_str = json.dumps(args, indent=2)
            if len(args_str) > 500:
                args_str = args_str[:500] + "\n... (truncated)"

            table.add_row(tool_name, args_str)
            self.console.print(table)

        elif event_type == "tool_result":
            result = data.get('result', '')
            tool_name = data.get('tool_name', 'unknown')

            # Truncate long results
            result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "\n... (truncated)"

            self.console.print(Panel(
                result_str,
                title=f"[yellow]Tool Result: {tool_name}[/yellow]",
                border_style="yellow"
            ))

        elif event_type == "state_update":
            if self.verbose:
                updates = data.get('updates', {})
                if updates:
                    self.console.print(f"[dim]State updated: {list(updates.keys())}[/dim]")

        elif event_type == "completion":
            score = data.get('score', 0)
            report = data.get('competition_report', {})

            self.console.print(Panel(
                f"[bold]Competition Complete![/bold]\n"
                f"Score: {score}\n"
                f"Any Medal: {report.get('any_medal', False)}\n"
                f"Above Median: {report.get('above_median', False)}",
                title="[green]Rollout Complete[/green]",
                border_style="green"
            ))

        elif event_type == "error":
            self.console.print(Panel(
                f"[bold red]Error:[/bold red] {data.get('error', 'Unknown error')}",
                border_style="red"
            ))

    def _log_to_file(self, event: Dict[str, Any], format: str = "text"):
        """Log event to file."""
        if not self.log_file:
            return

        with open(self.log_file, "a") as f:
            if format == "json":
                f.write(json.dumps(event) + "\n")
            else:
                # Human-readable text format
                f.write(f"\n{'='*80}\n")
                f.write(f"[{event['timestamp']}] {event['event_type'].upper()}\n")
                f.write(f"{'-'*80}\n")
                f.write(json.dumps(event['data'], indent=2))
                f.write(f"\n{'='*80}\n\n")


class MonitoredMLEBenchEnv(MLEBenchEnv):
    """MLEBenchEnv with real-time monitoring capabilities."""

    def __init__(
        self,
        monitor: Optional[RolloutMonitor] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.monitor = monitor or RolloutMonitor()

    async def setup_state(self, state: State, **kwargs) -> State:
        """Override to log sandbox creation."""
        state = await super().setup_state(state, **kwargs)

        self.monitor.log_event("sandbox_created", {
            "name": state["sandbox_name"],
            "competition_id": state["info"]["competition_id"],
            "gpu_device_id": state.get("gpu_device_id"),
            "sandbox_prepared": state.get("sandbox_prepared", False)
        })

        return state

    async def env_response(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> tuple[Messages, State]:
        """Override to log turns and tool responses."""
        current_turn = state.get("turn", 0)

        self.monitor.log_event("turn_start", {
            "turn": current_turn,
            "max_turns": self.max_turns,
            "sandbox_name": state.get("sandbox_name", "unknown")
        })

        # Log the last message if it's from assistant with tool calls
        if messages and messages[-1].get("role") == "assistant":
            msg = messages[-1]

            # Log assistant content
            if msg.get("content"):
                self.monitor.log_event("assistant_message", {
                    "content": msg["content"],
                    "turn": current_turn
                })

            # Log tool calls
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    self.monitor.log_event("tool_call", {
                        "tool_name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments"),
                        "turn": current_turn
                    })

        # Call parent method
        tool_messages, state = await super().env_response(messages, state, **kwargs)

        # Log tool results
        for tool_msg in tool_messages:
            if tool_msg.get("role") == "tool":
                self.monitor.log_event("tool_result", {
                    "tool_name": tool_msg.get("name", "unknown"),
                    "result": tool_msg.get("content", ""),
                    "turn": current_turn
                })

        return tool_messages, state

    async def is_completed(
        self,
        messages: Messages,
        state: State,
        **kwargs: Any
    ) -> bool:
        """Override to log completion."""
        completed = await super().is_completed(messages, state, **kwargs)

        if completed:
            self.monitor.log_event("completion", {
                "sandbox_name": state.get("sandbox_name", "unknown"),
                "score": state.get("info", {}).get("score", 0),
                "competition_report": state.get("competition_report", {}),
                "turns_used": state.get("turn", 0)
            })

        return completed


def load_monitored_environment(
    competition_ids: List[str],
    monitor_output: str = "console",  # "console", "file", "json", "all"
    monitor_file: Optional[str] = None,
    monitor_verbose: bool = True,
    **env_kwargs
) -> MonitoredMLEBenchEnv:
    """
    Load an MLEB environment with real-time monitoring.

    Args:
        competition_ids: List of competition IDs to load
        monitor_output: Output mode - "console", "file", "json", or "all"
        monitor_file: Path to log file (auto-generated if None)
        monitor_verbose: Whether to show verbose state updates
        **env_kwargs: Additional arguments passed to load_environment

    Returns:
        MonitoredMLEBenchEnv instance with monitoring enabled

    Example:
        >>> env = load_monitored_environment(
        ...     competition_ids=["spaceship-titanic"],
        ...     monitor_output="all",
        ...     monitor_file="my_rollout.log",
        ...     max_turns=20
        ... )
    """
    # Create monitor
    monitor = RolloutMonitor(
        output_mode=monitor_output,
        log_file=monitor_file,
        verbose=monitor_verbose
    )

    # Get base env config from load_environment
    # We need to extract the dataset and other setup
    base_env = load_environment(competition_ids=competition_ids, **env_kwargs)

    # Create monitored version with same config
    monitored_env = MonitoredMLEBenchEnv(
        monitor=monitor,
        max_concurrent_sandboxes=env_kwargs.get('max_concurrent_sandboxes', 2),
        max_turns=env_kwargs.get('max_turns', 20),
        tools=env_kwargs.get('tools'),
        sandbox_backend=env_kwargs.get('sandbox_backend', 'docker'),
        use_gpu=env_kwargs.get('use_gpu', False),
        num_gpu_partitions=env_kwargs.get('num_gpu_partitions', 4),
    )

    # Copy dataset and rubric from base env
    monitored_env.dataset = base_env.dataset
    monitored_env.rubric = base_env.rubric
    monitored_env._tools = base_env._tools

    return monitored_env
