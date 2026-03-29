"""CLI channel for development and debugging."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..events import EventBus

from ..events import UserSpeech, ChannelMessage

log = logging.getLogger(__name__)


class CLIChannel:
    """stdin/stdout channel for debugging. Runs in a background thread."""

    name = "cli"

    def __init__(self) -> None:
        self._bus: EventBus | None = None
        self._running = False

    async def start(self, bus: "EventBus") -> None:
        self._bus = bus
        self._running = True
        asyncio.get_event_loop().run_in_executor(None, self._read_loop)
        log.info("CLI channel started. Type your message and press Enter.")

    def _read_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                line = input()
                if not line.strip():
                    continue
                if line.strip() in ("exit", "quit", "q"):
                    self._running = False
                    print("[CLI] Goodbye.")
                    break
                event = ChannelMessage(channel="cli", sender="user", text=line.strip())
                asyncio.run_coroutine_threadsafe(self._bus.emit(event), loop)
            except EOFError:
                break
            except Exception:
                log.exception("CLI read error")
                break

    async def send(self, text: str, media: bytes | None = None) -> None:
        print(f"\n[Agent] {text}\n")
        sys.stdout.flush()

    def stop(self) -> None:
        self._running = False
