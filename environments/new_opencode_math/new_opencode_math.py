"""OpenCode Math environment using the experimental NewCliAgentEnv with resource management.

This is a standalone version that doesn't depend on research-environments,
using the experimental resource manager for better sandbox lifecycle tracking.
"""

import logging
from typing import Any

from datasets import load_dataset, Dataset
from prime_sandboxes import AsyncSandboxClient

import verifiers as vf
from verifiers.envs.experimental.new_cli_agent_env import NewCliAgentEnv
from verifiers.types import State

logger = logging.getLogger("verifiers.envs.NewOpenCodeMathEnv")


def _build_run_command(agent_workdir: str) -> str:
    """Build the shell command to run the OpenCode agent."""
    return f"""
set -e

echo "=== Starting OpenCode agent ==="
echo "Base URL: $OPENAI_BASE_URL"
echo "Instruction file: /task/instruction.txt"

apt-get update && apt-get install -y curl

# Install OpenCode
echo "Installing OpenCode..."
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"

echo "OpenCode installed, checking version..."
opencode --version || echo "opencode not found in PATH"

# Create opencode config directory
mkdir -p ~/.config/opencode

# Create opencode.json config with intercepted provider
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{{
  "\\$schema": "https://opencode.ai/config.json",
  "provider": {{
    "intercepted": {{
      "npm": "@ai-sdk/openai-compatible",
      "name": "Intercepted",
      "options": {{
        "baseURL": "$OPENAI_BASE_URL",
        "apiKey": "intercepted",
        "timeout": 600000
      }},
      "models": {{
        "model": {{
          "name": "Intercepted Model",
          "modalities": {{
            "input": ["text", "image"],
            "output": ["text"]
          }}
        }}
      }}
    }}
  }},
  "model": "intercepted/model"
}}
EOFCONFIG

echo "OpenCode config:"
cat ~/.config/opencode/opencode.json

mkdir -p /logs/agent

# Read instruction from file
echo "=== Instruction ==="
cat /task/instruction.txt
echo "=== End Instruction ==="

# Run OpenCode with the instruction from file
cd {agent_workdir}
echo "Running opencode..."
opencode run "$(cat /task/instruction.txt)" 2>&1 | tee /logs/agent/opencode.txt
echo "OpenCode exit code: $?"
"""


class NewOpenCodeMathEnv(NewCliAgentEnv):
    """Solve math problems using OpenCode agent with experimental resource management.

    This environment uses the new SandboxManager for:
    - Atomic resource tracking (no sandbox leakage)
    - Better error tracking (errors associated with rollouts)
    - Lifecycle observability
    """

    DEFAULT_DATASET_NAME = "PrimeIntellect/INTELLECT-3-RL"
    DEFAULT_DATASET_SUBSET = "math"
    DEFAULT_DATASET_SPLIT = "train"
    DEFAULT_INSTRUCTION_PROMPT = "Solve the following problem. Put your final answer in \\boxed{}.\n\n"
    DEFAULT_AGENT_WORKDIR = "/app"

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_subset: str = DEFAULT_DATASET_SUBSET,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
        question_key: str = "question",
        answer_key: str = "answer",
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        rubric: vf.Rubric | None = None,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.instruction_prompt = instruction_prompt
        self.question_key = question_key
        self.answer_key = answer_key
        self.agent_workdir = agent_workdir

        # Load and prepare the dataset
        dataset = self._prepare_dataset()

        # Use MathRubric if no rubric provided
        if rubric is None:
            rubric = vf.MathRubric(max_workers=128)

        super().__init__(
            run_command=_build_run_command(agent_workdir),
            dataset=dataset,
            rubric=rubric,
            **kwargs,
        )

    def _prepare_dataset(self) -> Dataset:
        """Load and prepare the math dataset."""
        raw_dataset = load_dataset(
            self.dataset_name,
            self.dataset_subset,
            split=self.dataset_split,
        )

        def prepare_example(example: dict[str, Any], idx: int) -> dict[str, Any]:
            question = example.get(self.question_key, "")
            answer = example.get(self.answer_key, "")

            # Format the instruction
            full_instruction = f"{self.instruction_prompt}{question}"

            return {
                "example_id": idx,
                "prompt": [{"role": "user", "content": full_instruction}],
                "answer": answer,
                "info": {
                    "question": question,
                    "original_answer": answer,
                },
            }

        prepared = [
            prepare_example(ex, i) for i, ex in enumerate(raw_dataset)
        ]
        return Dataset.from_list(prepared)

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        """Build environment variables including the instruction."""
        env_vars = await super().build_env_vars(state)

        # Extract instruction from the prompt
        prompt = state.get("prompt", [])
        if prompt and isinstance(prompt, list) and len(prompt) > 0:
            instruction = prompt[0].get("content", "")
        else:
            instruction = str(prompt)

        env_vars["OPENCODE_INSTRUCTION"] = instruction
        return env_vars

    async def post_sandbox_setup(
        self, state: State, sandbox_client: AsyncSandboxClient
    ) -> None:
        """Write the instruction file to the sandbox before running the agent."""
        sandbox_id = state["sandbox_id"]

        # Extract instruction from the prompt
        prompt = state.get("prompt", [])
        if prompt and isinstance(prompt, list) and len(prompt) > 0:
            instruction = prompt[0].get("content", "")
        else:
            instruction = str(prompt)

        # Create /task directory and write instruction file
        await sandbox_client.run_command(sandbox_id, "mkdir -p /task")

        # Write instruction to file using heredoc (single-quoted delimiter prevents expansion)
        write_cmd = f"cat > /task/instruction.txt << 'INSTRUCTION_EOF'\n{instruction}\nINSTRUCTION_EOF"
        await sandbox_client.run_command(sandbox_id, write_cmd)

        logger.debug(f"Wrote instruction to /task/instruction.txt ({len(instruction)} chars)")


def load_environment(
    dataset_name: str = NewOpenCodeMathEnv.DEFAULT_DATASET_NAME,
    dataset_subset: str = NewOpenCodeMathEnv.DEFAULT_DATASET_SUBSET,
    dataset_split: str = NewOpenCodeMathEnv.DEFAULT_DATASET_SPLIT,
    instruction_prompt: str = NewOpenCodeMathEnv.DEFAULT_INSTRUCTION_PROMPT,
    question_key: str = "question",
    answer_key: str = "answer",
    agent_workdir: str = NewOpenCodeMathEnv.DEFAULT_AGENT_WORKDIR,
    docker_image: str = "python:3.11-slim",
    timeout_seconds: float = 900.0,
    cpu_cores: int = 1,
    memory_gb: int = 1,
    disk_size_gb: int = 2,
    timeout_minutes: int = 60,
    max_turns: int = 100,
) -> NewOpenCodeMathEnv:
    return NewOpenCodeMathEnv(
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
        instruction_prompt=instruction_prompt,
        question_key=question_key,
        answer_key=answer_key,
        agent_workdir=agent_workdir,
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        max_turns=max_turns,
    )
