import verifiers as vf
from verifiers import Messages, State
from verifiers.types import ChatCompletionToolParam
from typing import Callable, Any
from datasets import Dataset
from agents.function_schema import function_schema
import inspect

def convert_func_to_oai_tool_exclude_params(
    func: Callable, exclude_params: set[str]
) -> ChatCompletionToolParam:
    """
    Convert a function to an OpenAI tool schema, excluding specified parameters.

    Args:
        func: The function to convert
        exclude_params: Set of parameter names to exclude from the schema
    """
    # Get the function's signature and docstring
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Create a new signature excluding the specified parameters
    filtered_params = [
        param for name, param in sig.parameters.items()
        if name not in exclude_params
    ]
    new_sig = sig.replace(parameters=filtered_params)

    # Create a wrapper function with the filtered signature
    def schema_wrapper(*args, **kwargs):
        pass

    schema_wrapper.__name__ = func.__name__
    schema_wrapper.__doc__ = doc
    schema_wrapper.__signature__ = new_sig
    schema_wrapper.__annotations__ = {
        k: v for k, v in func.__annotations__.items()
        if k not in exclude_params
    }

    # Generate the schema using the wrapper
    function_schema_obj = function_schema(schema_wrapper)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": function_schema_obj.description or "",
            "parameters": function_schema_obj.params_json_schema,
            "strict": True,
        },
    }


def super_secret_guessing_box(guess: str, state: Any = None) -> str:
    """
    Tool for making a guess.

    Args:
        guess: Your guess

    Returns:
        The result of your guess
    """
    sandbox_id = state.get('sandbox_id') if isinstance(state, dict) else None
    if sandbox_id:
        return f"Correct! The answer is: {guess}. (Sandbox: {sandbox_id})"
    return "WRONG"


def another_tool(message: str, state: Any = None) -> str:
    """
    Send a message.

    Args:
        message: The message to send

    Returns:
        Confirmation message
    """
    sandbox_id = state.get('sandbox_id') if isinstance(state, dict) else None
    if sandbox_id:
        return f"Message '{message}' sent from sandbox {sandbox_id}"
    return f"Message '{message}' sent (no sandbox)"


class UpdateStateEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{str(e)}",
        excluded_params: set[str] | None = None,
        **kwargs,
    ):
        self.excluded_params = excluded_params or {"state"}

        super().__init__(
            tools=tools,
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs,
        )

        self.oai_tools = [
            convert_func_to_oai_tool_exclude_params(tool, self.excluded_params)
            for tool in self.tools
        ]

    async def setup_state(self, state: State, **kwargs) -> State:
        state["sandbox_id"] = "1"
        return state

    def update_tool_args(
        self, tool_args: dict, messages: Messages, state: State, **kwargs
    ) -> dict:
        tool_args["state"] = state
        return tool_args



def load_environment() -> vf.StatefulToolEnv:
    ds_rows = []
    ds_rows.append(
        {
            "prompt": [{"role":"user","content":"Guess my name"}],
            "answer": ""
        }
    )
    ds = Dataset.from_list(ds_rows)
    rubric = vf.Rubric()
    vf_env = UpdateStateEnv(
        dataset=ds,
        rubric=rubric,
        tools=[super_secret_guessing_box, another_tool],
        max_turns=3
    )
    return vf_env
