"""ReAct query loop as an explicit state machine with dependency injection.

Inspired by claude-code's query.ts:
  - QueryState holds all mutable loop state
  - QueryDeps injects all I/O (model calls, tool execution)
  - The loop is a pure while-true with continue sites
  - Abort support via asyncio.Event
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events yielded by the query loop (like claude-code's stream events)
# ---------------------------------------------------------------------------

@dataclass
class QueryEvent:
    """Base for all query loop events."""
    pass


@dataclass
class IterationStart(QueryEvent):
    iteration: int = 0
    max_iterations: int = 0


@dataclass
class ModelResponse(QueryEvent):
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ToolCallStart(QueryEvent):
    tool: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class ToolCallDone(QueryEvent):
    tool: str = ""
    result: str = ""


@dataclass
class QueryComplete(QueryEvent):
    text: str = ""
    tools_used: list[dict] = field(default_factory=list)
    tokens_used: int = 0
    iterations: int = 0


@dataclass
class QueryError(QueryEvent):
    error: str = ""
    recoverable: bool = False


# ---------------------------------------------------------------------------
# QueryDeps: all I/O injected (inspired by claude-code query/deps.ts)
# ---------------------------------------------------------------------------

class QueryDeps(Protocol):
    """Dependency injection for the query loop. Swap for testing."""

    async def call_model(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Call the LLM. Return dict with 'content', 'tool_calls', 'role'."""
        ...

    async def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name. Return result string."""
        ...


# ---------------------------------------------------------------------------
# QueryState: all mutable state for one query loop run
# ---------------------------------------------------------------------------

@dataclass
class QueryState:
    messages: list[dict]
    tools: list[dict]
    max_iterations: int = 20
    tools_used: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    iteration: int = 0


# ---------------------------------------------------------------------------
# The loop itself (inspired by claude-code query.ts queryLoop)
# ---------------------------------------------------------------------------

async def query_loop(
    state: QueryState,
    deps: QueryDeps,
    abort: asyncio.Event | None = None,
) -> AsyncIterator[QueryEvent]:
    """Run the ReAct loop as an async generator, yielding events.

    This is the core reasoning engine. It:
    1. Calls the model with current messages + tools
    2. If model returns tool_calls → execute tools → append results → continue
    3. If model returns text only → yield QueryComplete → return
    4. Checks abort between iterations

    The caller drives the loop by iterating over events.
    """
    while state.iteration < state.max_iterations:
        if abort and abort.is_set():
            yield QueryError(error="Query aborted by user", recoverable=False)
            return

        state.iteration += 1
        yield IterationStart(iteration=state.iteration, max_iterations=state.max_iterations)

        try:
            response = await deps.call_model(
                messages=state.messages,
                tools=state.tools if state.tools else None,
            )
        except Exception as e:
            yield QueryError(error=f"Model call failed: {e}", recoverable=True)
            return

        if response is None:
            yield QueryError(error="Model returned None after retries", recoverable=False)
            return

        content = _extract_content(response)
        tool_calls = _extract_tool_calls(response)

        yield ModelResponse(content=content, tool_calls=tool_calls)

        if not tool_calls:
            yield QueryComplete(
                text=content,
                tools_used=state.tools_used,
                tokens_used=state.total_tokens,
                iterations=state.iteration,
            )
            return

        state.messages.append(_response_to_dict(response))

        for call in tool_calls:
            fn_name = call.get("name", "")
            fn_args = call.get("args", {})

            yield ToolCallStart(tool=fn_name, args=fn_args)

            if abort and abort.is_set():
                result_str = "Aborted"
            else:
                result_str = await deps.execute_tool(fn_name, fn_args)

            state.tools_used.append({
                "tool": fn_name,
                "args": fn_args,
                "result": result_str[:2000],
            })
            state.messages.append({"role": "tool", "content": str(result_str)})

            yield ToolCallDone(tool=fn_name, result=result_str[:200])

    yield QueryComplete(
        text="Reached max iterations, please simplify.",
        tools_used=state.tools_used,
        tokens_used=state.total_tokens,
        iterations=state.iteration,
    )


# ---------------------------------------------------------------------------
# Helpers for normalizing model responses (ollama objects vs dicts)
# ---------------------------------------------------------------------------

def _extract_content(response) -> str:
    msg = response.message if hasattr(response, "message") else response.get("message", {})
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content", "")
    return content or ""


def _extract_tool_calls(response) -> list[dict]:
    msg = response.message if hasattr(response, "message") else response.get("message", {})
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls is None and isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return []

    result = []
    for call in tool_calls:
        fn = call.function if hasattr(call, "function") else call.get("function", {})
        fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        fn_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", {})
        result.append({"name": fn_name, "args": fn_args})
    return result


def _response_to_dict(response) -> dict:
    msg = response.message if hasattr(response, "message") else response.get("message", {})
    if isinstance(msg, dict):
        return msg
    d: dict = {"role": getattr(msg, "role", "assistant")}
    content = getattr(msg, "content", "")
    if content:
        d["content"] = content
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = tool_calls
    return d
