"""Channel abstractions for the edge-agent framework."""

from __future__ import annotations

from typing import Protocol, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..events import EventBus


class Channel(Protocol):
    name: str

    async def start(self, bus: "EventBus") -> None: ...
    async def send(self, text: str, media: Optional[bytes] = None) -> None: ...
