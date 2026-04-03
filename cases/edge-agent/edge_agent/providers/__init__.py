"""Provider protocol definitions for System 1 (Perception) and System 2 (Reasoning)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..context import RenderedContext

ToolExecutor = Callable[[str, dict], Awaitable[str]]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerceptionResult:
    text: str = ""
    is_listening: bool = True
    audio_chunks: list[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    text: str = ""
    tools_used: list[dict] = field(default_factory=list)
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# System 1: real-time multimodal perception (streaming)
# ---------------------------------------------------------------------------

@runtime_checkable
class PerceptionProvider(Protocol):

    async def start(self, system_prompt: str) -> None: ...
    async def pause(self) -> None: ...
    async def resume(self) -> None: ...
    async def feed(self, audio_b64: str, frame_b64: str = "") -> PerceptionResult: ...
    async def inject_context(self, text: str) -> None: ...
    async def reset(self) -> None: ...
    async def health(self) -> bool: ...


# ---------------------------------------------------------------------------
# System 2: deep reasoning with tool calling (request-response)
# ---------------------------------------------------------------------------

@runtime_checkable
class ReasoningProvider(Protocol):

    async def reason(
        self,
        message: str,
        context: "RenderedContext",
        tools: list[dict],
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = 20,
    ) -> ReasoningResult: ...

    async def health(self) -> bool: ...
