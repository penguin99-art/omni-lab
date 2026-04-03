"""Context model types for turn-time assembly and provider rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnInput:
    """Single user turn entering the runtime."""

    text: str
    channel: str = "cli"
    sender: str = ""
    visual_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHit:
    """A relevant memory returned by the memory store."""

    text: str
    score: float
    source: str = "memory"


@dataclass(frozen=True)
class ContextSnapshot:
    """Provider-agnostic context assembled for one reasoning turn."""

    identity: str = ""
    user_profile: str = ""
    recent_turns: list[dict] = field(default_factory=list)
    relevant_memories: list[MemoryHit] = field(default_factory=list)
    visual_context: str = ""
    channel_context: str = ""
    tool_names: list[str] = field(default_factory=list)
    runtime_notes: list[str] = field(default_factory=list)
    turn: TurnInput | None = None


@dataclass(frozen=True)
class RenderedContext:
    """Provider-ready prompt/messages plus optional debug metadata."""

    system_prompt: str
    messages: list[dict]
    debug: dict[str, Any] = field(default_factory=dict)
