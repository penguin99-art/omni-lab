"""Event definitions and EventBus for the edge-agent framework."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Type

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Event:
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Perception events (System 1 output)
# ---------------------------------------------------------------------------

@dataclass
class Utterance(Event):
    """System 1 produced a spoken reply."""
    text: str = ""


@dataclass
class UserSpeech(Event):
    """User spoke (transcribed by Web Speech API or Whisper)."""
    text: str = ""


@dataclass
class VisualScene(Event):
    """Visual scene description updated."""
    description: str = ""


@dataclass
class Silence(Event):
    """Prolonged silence detected."""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Routing events
# ---------------------------------------------------------------------------

@dataclass
class IntentDecision(Event):
    intent: str = "fast"  # "fast" | "slow"
    text: str = ""


# ---------------------------------------------------------------------------
# Reasoning events (System 2 output)
# ---------------------------------------------------------------------------

@dataclass
class ThinkingStarted(Event):
    query: str = ""


@dataclass
class ToolExecuting(Event):
    tool: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class ReasoningDone(Event):
    text: str = ""
    tools_used: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Output events
# ---------------------------------------------------------------------------

@dataclass
class SpeakRequest(Event):
    text: str = ""
    via: str = "browser"  # "browser" | "system1"


# ---------------------------------------------------------------------------
# Channel events
# ---------------------------------------------------------------------------

@dataclass
class ChannelMessage(Event):
    channel: str = ""
    sender: str = ""
    text: str = ""
    media: bytes | None = None


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------

@dataclass
class MemoryUpdated(Event):
    fact: str = ""


@dataclass
class HealthCheck(Event):
    system1_ok: bool = True
    system2_ok: bool = True


# ---------------------------------------------------------------------------
# Second-brain capture events
# ---------------------------------------------------------------------------

@dataclass
class CaptureEvent(Event):
    """System 1 completed a screen/env capture cycle."""
    scene: str = ""
    source: str = "screen"  # "screen" | "camera" | "audio"


@dataclass
class DigestRequest(Event):
    buffer_size: int = 0


@dataclass
class ProactiveHint(Event):
    hint: str = ""
    related_memories: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Lightweight async pub/sub."""

    def __init__(self) -> None:
        self._handlers: dict[Type[Event], list[Callable]] = defaultdict(list)

    def on(self, event_type: Type[Event], handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def off(self, event_type: Type[Event], handler: Callable) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        for handler in self._handlers[type(event)]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                log.exception("EventBus handler error for %s", type(event).__name__)
