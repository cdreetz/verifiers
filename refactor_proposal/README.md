# Verifiers v2 Architecture Proposal

## Design Principles

1. **Async-first**: Everything is async. No `maybe_await` hacks.
2. **Composition over inheritance**: Small, focused components that compose.
3. **Explicit state**: No magic dict forwarding. Clear schemas.
4. **Resource lifecycle**: First-class support for sandboxes, connections, etc.
5. **Backpressure-aware**: Proper flow control for high-throughput generation.

## Directory Structure

```
verifiers/
├── __init__.py                 # Public API (backward compat shims)
├── py.typed                    # PEP 561 marker
│
├── core/                       # Foundation layer (no deps on other verifiers modules)
│   ├── __init__.py
│   ├── types.py                # Core types: Message, Trajectory, RolloutResult
│   ├── state.py                # State container with schema validation
│   ├── errors.py               # Exception hierarchy
│   └── protocols.py            # Abstract protocols (interfaces)
│
├── async_engine/               # The async runtime
│   ├── __init__.py
│   ├── executor.py             # Main executor with backpressure
│   ├── batch.py                # Batching and grouping utilities
│   ├── semaphore.py            # Weighted semaphore for resource limits
│   ├── retry.py                # Retry policies with backoff
│   ├── stream.py               # Async iterators and streams
│   └── cancel.py               # Cancellation and timeout handling
│
├── resources/                  # Resource lifecycle management
│   ├── __init__.py
│   ├── pool.py                 # Generic async resource pool
│   ├── sandbox.py              # Sandbox resource type
│   ├── client.py               # LLM client resource type
│   └── lifecycle.py            # Setup/teardown coordination
│
├── env/                        # Environment layer
│   ├── __init__.py
│   ├── base.py                 # Base Environment protocol
│   ├── single_turn.py          # Single-turn implementation
│   ├── multi_turn.py           # Multi-turn with turn management
│   ├── tool.py                 # Tool-based environments
│   └── composite.py            # EnvGroup and mixtures
│
├── scoring/                    # Reward computation
│   ├── __init__.py
│   ├── rubric.py               # Core rubric (reward aggregation)
│   ├── functions.py            # Reward function registry and injection
│   ├── judge.py                # LLM-as-judge
│   ├── math.py                 # Math verification
│   └── group.py                # Group-based scoring
│
├── parsing/                    # Output parsing
│   ├── __init__.py
│   ├── parser.py               # Base parser protocol
│   ├── xml.py                  # XML tag extraction
│   ├── json.py                 # JSON extraction
│   └── think.py                # Think block handling
│
├── tools/                      # Tool system
│   ├── __init__.py
│   ├── schema.py               # OpenAI schema generation
│   ├── executor.py             # Tool execution with injection
│   ├── sandbox_tools.py        # Sandbox-backed tools
│   └── mcp.py                  # MCP protocol support
│
├── compat/                     # Backward compatibility layer
│   ├── __init__.py
│   ├── v1_env.py               # Shims for v1 Environment classes
│   ├── v1_state.py             # State forwarding behavior
│   ├── v1_rubric.py            # Old rubric interface
│   └── load_environment.py     # load_environment() contract
│
├── rl/                         # Training integration (unchanged structure)
│   ├── trainer/
│   └── inference/
│
└── utils/                      # Shared utilities
    ├── __init__.py
    ├── tokens.py               # Token counting
    ├── messages.py             # Message manipulation
    └── logging.py              # Structured logging
```

## Key Architectural Changes

### 1. Protocols Instead of Base Classes

```python
# Instead of inheriting from Environment, implement the protocol
class Environment(Protocol):
    async def rollout(self, input: RolloutInput, ctx: Context) -> RolloutResult: ...
    async def score(self, result: RolloutResult) -> ScoredResult: ...
```

### 2. Context Object for Cross-Cutting Concerns

```python
# Instead of threading state through everything
@dataclass
class Context:
    client: LLMClient
    resources: ResourceManager
    cancel_scope: CancelScope
    metrics: MetricsCollector
```

### 3. Explicit Resource Lifecycle

```python
async with ResourcePool(SandboxFactory(), max_size=10) as pool:
    async with pool.acquire() as sandbox:
        result = await execute_code(sandbox, code)
```

### 4. Stream-Based Generation

```python
# Instead of returning a big list, stream results
async for result in engine.generate(inputs):
    yield result  # Backpressure-aware
```
