"""OllamaProvider: System 2 implementation using the query loop state machine.

The provider is now thin: it implements QueryDeps (model calls + retry)
and delegates the ReAct loop to agent/query.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import ollama as _ollama

from . import ReasoningResult
from ..agent.query import QueryState, QueryEvent, QueryComplete, QueryError, IterationStart, ToolCallStart, query_loop

if TYPE_CHECKING:
    from ..context import RenderedContext

log = logging.getLogger(__name__)


class OllamaProvider:
    """
    Lightweight System 2: Ollama API + injected query loop.

    Implements QueryDeps protocol so the query loop can call the model
    and execute tools without knowing about Ollama specifics.
    """

    def __init__(
        self,
        model: str = "qwen3.5:27b",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = _ollama.AsyncClient(host=base_url)
        self._abort: asyncio.Event | None = None

    # -- QueryDeps implementation ----------------------------------------------

    async def call_model(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict | None:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        return await self._chat_with_retry(**kwargs)

    async def execute_tool(self, name: str, args: dict) -> str:
        if self._tool_executor:
            return await self._tool_executor(name, args)
        return f"Tool {name} not available (no executor)"

    # -- ReasoningProvider interface -------------------------------------------

    async def reason(
        self,
        message: str,
        context: "RenderedContext",
        tools: list[dict],
        tool_executor=None,
        max_iterations: int = 20,
    ) -> ReasoningResult:
        self._tool_executor = tool_executor
        self._abort = asyncio.Event()

        state = QueryState(
            messages=list(context.messages),
            tools=tools,
            max_iterations=max_iterations,
        )

        final_text = ""
        final_tools: list[dict] = []
        final_tokens = 0

        async for event in query_loop(state, self, self._abort):
            if isinstance(event, IterationStart):
                log.info("ReAct iteration %d/%d", event.iteration, event.max_iterations)
            elif isinstance(event, ToolCallStart):
                log.info("Tool call: %s(%s)", event.tool, event.args)
            elif isinstance(event, QueryComplete):
                final_text = event.text
                final_tools = event.tools_used
                final_tokens = event.tokens_used
            elif isinstance(event, QueryError):
                log.error("Query error: %s", event.error)
                final_text = event.error

        self._tool_executor = None
        return ReasoningResult(
            text=final_text,
            tools_used=final_tools,
            tokens_used=final_tokens,
        )

    def abort(self) -> None:
        """Cancel the current reasoning loop (thread-safe)."""
        if self._abort:
            self._abort.set()

    # -- Retry + health --------------------------------------------------------

    async def _ensure_client(self) -> None:
        try:
            await self._client.list()
        except Exception:
            log.info("Reconnecting to Ollama at %s...", self.base_url)
            self._client = _ollama.AsyncClient(host=self.base_url)

    async def _chat_with_retry(self, **kwargs) -> object | None:
        for attempt in range(self._max_retries):
            try:
                return await self._client.chat(**kwargs)
            except Exception as e:
                log.warning("Ollama chat attempt %d/%d failed: %s", attempt + 1, self._max_retries, e)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    await self._ensure_client()
        log.error("All Ollama chat retries exhausted")
        return None

    async def health(self) -> bool:
        try:
            await self._client.list()
            return True
        except Exception:
            try:
                await self._ensure_client()
                await self._client.list()
                return True
            except Exception:
                return False
