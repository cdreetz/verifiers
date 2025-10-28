# Real-Time Rollout Monitoring

Monitor MLEB agent rollouts in real-time without modifying the verifiers library.

## Features

- **Real-time visibility**: See agent messages, tool calls, and results as they happen
- **Multiple output formats**: Console (pretty-printed), text files, or JSON
- **No verifiers modifications**: Pure wrapper approach
- **Concurrent support**: Monitor multiple competitions simultaneously
- **Rich formatting**: Beautiful console output with colors and panels

## Quick Start

### Basic Usage

```python
from src.rollout_monitor import load_monitored_environment

# Replace load_environment with load_monitored_environment
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="console",  # See output in terminal
    max_turns=20
)

# Use with verifiers as normal
```

### Output Modes

#### 1. Console Output (Default)
```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="console"
)
```
- Pretty-printed output with colors
- Panels for messages, tool calls, and results
- Progress indicators

#### 2. File Logging
```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="file",
    monitor_file="my_rollout.log"
)
```
- Human-readable text format
- Includes timestamps and event types
- Good for review after completion

#### 3. JSON Logging
```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="json",
    monitor_file="rollout.jsonl"
)
```
- One JSON event per line
- Easy to parse programmatically
- Great for analysis and metrics

#### 4. All Outputs
```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="all",  # Console + file
    monitor_file="rollout.log"
)
```

## Event Types

The monitor captures these events:

| Event | Description |
|-------|-------------|
| `sandbox_created` | Sandbox initialization complete |
| `turn_start` | New turn begins |
| `assistant_message` | Agent sends a message |
| `tool_call` | Agent calls a tool |
| `tool_result` | Tool execution result |
| `state_update` | Internal state changes |
| `completion` | Rollout complete with scores |
| `error` | Any errors encountered |

## Examples

### Monitor Multiple Competitions

```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic", "leaf-classification"],
    monitor_output="all",
    max_concurrent_sandboxes=2
)
```

### Verbose State Tracking

```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="console",
    monitor_verbose=True  # Show state updates
)
```

### Custom Log Location

```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="all",
    monitor_file="logs/my_experiment_001.log"
)
```

## Analyzing Logs

### Parse JSON Logs

```python
import json

# Load all events
with open('rollout.jsonl') as f:
    events = [json.loads(line) for line in f]

# Count tool calls
tool_calls = [e for e in events if e['event_type'] == 'tool_call']
print(f"Total tool calls: {len(tool_calls)}")

# Tool usage breakdown
from collections import Counter
tools_used = Counter(e['data']['tool_name'] for e in tool_calls)
print(tools_used)

# Find errors
errors = [e for e in events if e['event_type'] == 'error']
for error in errors:
    print(f"Error at {error['timestamp']}: {error['data']['error']}")

# Get final score
completion = [e for e in events if e['event_type'] == 'completion'][0]
print(f"Final score: {completion['data']['score']}")
```

### Extract Tool Call Sequences

```python
import json

def get_tool_sequence(log_file):
    """Extract sequence of tool calls from a rollout."""
    with open(log_file) as f:
        events = [json.loads(line) for line in f]

    tool_calls = [e for e in events if e['event_type'] == 'tool_call']
    return [tc['data']['tool_name'] for tc in tool_calls]

sequence = get_tool_sequence('rollout.jsonl')
print(f"Tool sequence: {' -> '.join(sequence)}")
```

## Integration with Existing Code

### Option 1: Modify Your Script

```python
# Before
from src.mleb import load_environment
env = load_environment(competition_ids=["spaceship-titanic"])

# After
from src.rollout_monitor import load_monitored_environment
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="all"
)
```

### Option 2: Direct Wrapper

```python
from src.mleb import load_environment
from src.rollout_monitor import RolloutMonitor, MonitoredMLEBenchEnv

# Load base environment
base_env = load_environment(competition_ids=["spaceship-titanic"])

# Wrap with monitor
monitor = RolloutMonitor(output_mode="console")
monitored_env = MonitoredMLEBenchEnv(
    monitor=monitor,
    **base_env.__dict__
)
```

## Performance Considerations

- **Console output**: Minimal overhead, but can slow down if printing very large results
- **File logging**: Negligible overhead
- **JSON logging**: Minimal overhead, recommended for production
- **Truncation**: Long results are automatically truncated in console output to prevent slowdown

## Troubleshooting

### Console output too verbose

```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="console",
    monitor_verbose=False  # Hide state updates
)
```

### Want to see raw tool results

Check the log files - they contain full, untruncated results:

```python
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="all",  # Console (truncated) + file (full)
)
```

### Monitor multiple experiments

Use unique log files:

```python
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
env = load_monitored_environment(
    competition_ids=["spaceship-titanic"],
    monitor_output="json",
    monitor_file=f"logs/experiment_{timestamp}.jsonl"
)
```

## See Also

- `examples/monitor_rollout_example.py` - Complete examples
- `src/rollout_monitor.py` - Implementation details
- `src/mleb.py` - Base MLEB environment
