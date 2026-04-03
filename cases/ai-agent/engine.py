"""Core reasoning loop: send messages to Ollama, handle tool calls."""

import json

import ollama

from .tools import TOOL_DEFS, execute


def run_turn(
    messages: list,
    *,
    model: str,
    memory_file,
    today_file,
    max_iterations: int = 20,
) -> str:
    """Run one reasoning turn with tool calling. Returns final text."""
    for _ in range(max_iterations):
        resp = ollama.chat(model=model, messages=messages, tools=TOOL_DEFS, think=False)
        msg = resp.message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "(no response)"

        messages.append(msg)
        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)[:80]})")
            result = execute(
                name, args, memory_file=memory_file, today_file=today_file,
            )
            messages.append({"role": "tool", "content": result})

    return "(reached max iterations)"
