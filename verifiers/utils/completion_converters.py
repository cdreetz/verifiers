"""
Utilities for converting different API response formats to the verifiers completion format.

The verifiers completion format uses OpenAI's chat message format:
- List of dicts with "role" and "content" keys
- Optional "tool_calls" for assistant messages
- Optional "tool_call_id" for tool response messages
"""

from typing import Any, cast

from verifiers.types import ChatMessage, Messages


def anthropic_message_to_openai(message: dict[str, Any]) -> list[ChatMessage]:
    """
    Convert a single Anthropic message to OpenAI chat message format.

    Anthropic format:
    - role: "user" or "assistant"
    - content: str or list of content blocks
      - text blocks: {"type": "text", "text": "..."}
      - tool_use blocks: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
      - tool_result blocks: {"type": "tool_result", "tool_use_id": "...", "content": "..."}

    OpenAI format:
    - role: "system", "user", "assistant", or "tool"
    - content: str or list of content parts
    - tool_calls: list of tool call objects (for assistant messages)
    - tool_call_id: str (for tool response messages)

    Args:
        message: An Anthropic message dict

    Returns:
        List of OpenAI-formatted ChatMessage dicts (may be multiple if tool results present)
    """
    role = message.get("role", "user")
    content = message.get("content", "")

    # Simple string content
    if isinstance(content, str):
        return [cast(ChatMessage, {"role": role, "content": content})]

    # Content is a list of blocks
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                # Convert Anthropic tool_use to OpenAI tool_calls format
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _serialize_tool_input(block.get("input", {})),
                    },
                })
            elif block_type == "tool_result":
                # Tool results become separate tool messages in OpenAI format
                tool_result_content = block.get("content", "")
                if isinstance(tool_result_content, list):
                    # Extract text from content blocks
                    tool_result_content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in tool_result_content
                    )
                tool_results.append({
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": tool_result_content,
                    "is_error": block.get("is_error", False),
                })

    result: list[ChatMessage] = []

    # Handle tool results first (they go as separate "tool" messages in OpenAI)
    for tool_result in tool_results:
        tool_msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_result["tool_use_id"],
            "content": tool_result["content"],
        }
        result.append(cast(ChatMessage, tool_msg))

    # Create the main message if there's text or tool calls
    combined_text = "\n".join(text_parts) if text_parts else ""
    if combined_text or tool_calls or (not tool_results):
        main_msg: dict[str, Any] = {
            "role": role,
            "content": combined_text,
        }
        if tool_calls:
            main_msg["tool_calls"] = tool_calls
        result.append(cast(ChatMessage, main_msg))

    return result


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[ChatMessage]:
    """
    Convert a list of Anthropic messages to OpenAI chat message format.

    Args:
        messages: List of Anthropic message dicts
        system: Optional system prompt (string or Anthropic system content blocks)

    Returns:
        List of OpenAI-formatted ChatMessage dicts
    """
    result: list[ChatMessage] = []

    # Handle system message
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            # Extract text from system content blocks
            system_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in system
            )
        else:
            system_text = str(system)
        result.append(cast(ChatMessage, {"role": "system", "content": system_text}))

    # Convert each message
    for msg in messages:
        result.extend(anthropic_message_to_openai(msg))

    return result


def anthropic_to_completion(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> Messages:
    """
    Convert Anthropic API messages to verifiers completion format.

    This is an alias for anthropic_messages_to_openai for clarity.

    Args:
        messages: List of Anthropic message dicts
        system: Optional system prompt

    Returns:
        Messages in verifiers format (list of ChatMessage dicts)
    """
    return anthropic_messages_to_openai(messages, system)


def openai_response_to_completion(response: dict[str, Any]) -> Messages:
    """
    Convert an OpenAI Responses API response to verifiers completion format.

    The Responses API (used by OpenAI Agents SDK) returns structured output items
    rather than traditional chat completion choices.

    Response format:
    - output: list of output items
      - message items: {"type": "message", "role": "assistant", "content": [...]}
      - function_call items: {"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."}
      - function_call_output items: {"type": "function_call_output", "call_id": "...", "output": "..."}

    Args:
        response: An OpenAI Responses API response dict

    Returns:
        Messages in verifiers format (list of ChatMessage dicts)
    """
    result: list[ChatMessage] = []
    output = response.get("output", [])

    # Track pending tool calls to group with their message
    pending_tool_calls: list[dict[str, Any]] = []
    pending_text_content: list[str] = []

    for item in output:
        item_type = item.get("type", "")

        if item_type == "message":
            # Flush any pending content first
            if pending_text_content or pending_tool_calls:
                msg = _create_assistant_message(pending_text_content, pending_tool_calls)
                result.append(msg)
                pending_text_content = []
                pending_tool_calls = []

            # Process message content
            role = item.get("role", "assistant")
            content = item.get("content", [])

            if isinstance(content, str):
                result.append(cast(ChatMessage, {"role": role, "content": content}))
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "output_text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "refusal":
                            text_parts.append(f"[Refusal: {block.get('refusal', '')}]")
                combined = "\n".join(text_parts) if text_parts else ""
                result.append(cast(ChatMessage, {"role": role, "content": combined}))

        elif item_type == "function_call":
            # Accumulate tool calls
            pending_tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })

        elif item_type == "function_call_output":
            # Flush pending assistant message with tool calls first
            if pending_text_content or pending_tool_calls:
                msg = _create_assistant_message(pending_text_content, pending_tool_calls)
                result.append(msg)
                pending_text_content = []
                pending_tool_calls = []

            # Add tool response message
            result.append(cast(ChatMessage, {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            }))

        elif item_type == "reasoning":
            # Include reasoning as part of assistant content
            summary = item.get("summary", [])
            if summary:
                for s in summary:
                    if isinstance(s, dict) and s.get("type") == "summary_text":
                        pending_text_content.append(f"[Reasoning: {s.get('text', '')}]")
                    elif isinstance(s, str):
                        pending_text_content.append(f"[Reasoning: {s}]")

    # Flush any remaining content
    if pending_text_content or pending_tool_calls:
        msg = _create_assistant_message(pending_text_content, pending_tool_calls)
        result.append(msg)

    return result


def openai_response_output_to_completion(output: list[dict[str, Any]]) -> Messages:
    """
    Convert OpenAI Responses API output items directly to completion format.

    This is useful when you have just the output array without the full response.

    Args:
        output: List of output items from an OpenAI Responses API response

    Returns:
        Messages in verifiers format (list of ChatMessage dicts)
    """
    return openai_response_to_completion({"output": output})


def _create_assistant_message(
    text_parts: list[str],
    tool_calls: list[dict[str, Any]],
) -> ChatMessage:
    """Create an assistant message with optional tool calls."""
    combined_text = "\n".join(text_parts) if text_parts else ""
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": combined_text,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return cast(ChatMessage, msg)


def _serialize_tool_input(input_data: Any) -> str:
    """Serialize tool input to JSON string."""
    import json

    if isinstance(input_data, str):
        return input_data
    try:
        return json.dumps(input_data)
    except (TypeError, ValueError):
        return str(input_data)
