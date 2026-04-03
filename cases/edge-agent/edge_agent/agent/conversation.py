"""ConversationEngine: owns message history and per-conversation state.

Inspired by claude-code's QueryEngine.ts:
  - Holds mutableMessages, tool pool, memory context
  - submit_message() is the single entry point for all user input
  - Decoupled from transport (WebSocket/CLI/API)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from ..context import ContextBuilder, OllamaContextRenderer, TurnInput
from ..tools import ToolPool
from ..memory import MemoryStore

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a single conversation turn."""
    text: str = ""
    tools_used: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0
    aborted: bool = False


class ConversationEngine:
    """Manages one conversation: message history, tools, memory.

    Transport-agnostic — the caller (EdgeAgent, WebSocketChannel, CLI)
    provides the reasoning provider.
    """

    def __init__(
        self,
        tool_pool: ToolPool,
        memory: MemoryStore,
        reason_fn: Callable[..., Awaitable[Any]],
        max_iterations: int = 20,
        context_builder: ContextBuilder | None = None,
        context_renderer: OllamaContextRenderer | None = None,
    ) -> None:
        self._tools = tool_pool
        self._memory = memory
        self._reason_fn = reason_fn
        self._max_iterations = max_iterations
        self._context_builder = context_builder or ContextBuilder(memory=memory, tool_pool=tool_pool)
        self._context_renderer = context_renderer or OllamaContextRenderer()
        self._abort_event: asyncio.Event | None = None
        self.turns: list[TurnResult] = []

    async def submit_message(
        self,
        text: str,
        *,
        channel: str = "cli",
        sender: str = "",
        visual_context: str = "",
        metadata: dict | None = None,
    ) -> TurnResult:
        """Process one user message through the reasoning pipeline.

        Returns TurnResult with the agent's response.
        """
        self._abort_event = asyncio.Event()
        t0 = time.monotonic()

        turn_input = TurnInput(
            text=text,
            channel=channel,
            sender=sender,
            visual_context=visual_context,
            metadata=metadata or {},
        )
        snapshot = self._context_builder.build(turn_input)
        ctx = self._context_renderer.render(snapshot)
        tool_defs = self._tools.definitions()

        async def tool_executor(name: str, args: dict) -> str:
            return await self._tools.execute(name, args)

        result = await self._reason_fn(
            message=text,
            context=ctx,
            tools=tool_defs,
            tool_executor=tool_executor,
            max_iterations=self._max_iterations,
        )

        elapsed = (time.monotonic() - t0) * 1000

        reply_text = result.text if hasattr(result, "text") else str(result)
        tools_used = result.tools_used if hasattr(result, "tools_used") else []

        self._memory.add_user(text)
        if reply_text:
            self._memory.add_assistant(reply_text)

        turn = TurnResult(
            text=reply_text,
            tools_used=tools_used,
            elapsed_ms=elapsed,
            aborted=self._abort_event.is_set() if self._abort_event else False,
        )
        self.turns.append(turn)

        log.info(
            "Turn completed: %d chars, %d tools, %.0fms",
            len(reply_text), len(tools_used), elapsed,
        )
        return turn

    def abort(self) -> None:
        """Cancel the current turn."""
        if self._abort_event:
            self._abort_event.set()
