"""OllamaProvider: System 2 implementation using Ollama API with ReAct loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ollama as _ollama

from . import ReasoningResult

if TYPE_CHECKING:
    from ..memory import Context

log = logging.getLogger(__name__)


class OllamaProvider:
    """
    Lightweight System 2: direct Ollama API calls with self-built ReAct loop.
    Zero external framework dependency.
    """

    def __init__(
        self,
        model: str = "qwen3.5:27b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = _ollama.AsyncClient(host=base_url)

    async def reason(
        self,
        message: str,
        context: "Context",
        tools: list[dict],
        max_iterations: int = 20,
        tool_executor=None,
    ) -> ReasoningResult:
        messages = context.to_ollama_messages()
        messages.append({"role": "user", "content": message})
        tools_used: list[dict] = []
        total_tokens = 0

        for iteration in range(max_iterations):
            log.info("ReAct iteration %d/%d", iteration + 1, max_iterations)

            kwargs: dict = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools

            try:
                response = await self._client.chat(**kwargs)
            except Exception:
                log.exception("Ollama chat failed")
                return ReasoningResult(text="Ollama call failed", tools_used=tools_used)

            msg = response.message if hasattr(response, "message") else response.get("message", {})
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls is None and isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content", "")

            if not tool_calls:
                return ReasoningResult(
                    text=content or "",
                    tools_used=tools_used,
                    tokens_used=total_tokens,
                )

            messages.append(msg if isinstance(msg, dict) else _msg_to_dict(msg))

            for call in tool_calls:
                fn = call.function if hasattr(call, "function") else call.get("function", {})
                fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                fn_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", {})

                log.info("Tool call: %s(%s)", fn_name, fn_args)

                if tool_executor:
                    result_str = await tool_executor(fn_name, fn_args)
                else:
                    result_str = "Tool {} not available (no executor)".format(fn_name)

                tools_used.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result": str(result_str)[:2000],
                })
                messages.append({"role": "tool", "content": str(result_str)})

        return ReasoningResult(
            text="Reached max iterations, please simplify.",
            tools_used=tools_used,
            tokens_used=total_tokens,
        )

    async def health(self) -> bool:
        try:
            await self._client.list()
            return True
        except Exception:
            return False


def _msg_to_dict(msg) -> dict:
    """Convert an ollama message object to a plain dict."""
    d: dict = {"role": getattr(msg, "role", "assistant")}
    content = getattr(msg, "content", "")
    if content:
        d["content"] = content
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = tool_calls
    return d
