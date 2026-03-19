# New OpenCode Math Environment (Experimental)

This environment runs the OpenCode agent on math problems using the **experimental resource manager** for better sandbox lifecycle tracking.

## Differences from `opencode_math`

1. **Does not depend on `research-environments`** - Standalone implementation
2. **Uses `NewCliAgentEnv`** - Experimental environment with `SandboxManager`
3. **Better resource tracking** - Atomic sandbox tracking, per-rollout error attribution

## Usage

```python
from environments.new_opencode_math.new_opencode_math import load_environment

env = load_environment()

# Or with custom settings
env = load_environment(
    dataset_name="PrimeIntellect/INTELLECT-3-RL",
    dataset_subset="math",
    max_turns=50,
    cpu_cores=2,
    memory_gb=2,
)
```

## Evaluation

```bash
cd environments/new_opencode_math
vf-eval --model openai/gpt-4o-mini --num-examples 5
```

## Features

- Uses `SandboxManager` for centralized resource lifecycle management
- Prints sandbox lifecycle summary at teardown
- Tracks errors per rollout for better debugging
- State machine for sandbox lifecycle (CREATING -> READY -> DESTROYED)

## Comparison with Original

To compare resource tracking between original and experimental:

```python
# Original (opencode_math) - uses research-environments
# Experimental (new_opencode_math) - uses SandboxManager

# The experimental version will print a summary like:
# ============================================================
# SANDBOX SUMMARY
# ============================================================
# Total sandboxes: 10
# States:
#   DESTROYED: 10
# No errors
# ============================================================
```
