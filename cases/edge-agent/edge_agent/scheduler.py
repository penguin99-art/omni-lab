"""GPUScheduler: ensures serial GPU usage between System 1 and System 2."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers import PerceptionProvider

log = logging.getLogger(__name__)


class GPUScheduler:
    """
    Both models stay in VRAM (128 GB is enough).
    Only manages compute timing: at most one model does forward pass at a time.
    """

    def __init__(self, perception: PerceptionProvider | None = None) -> None:
        self._perception = perception
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def use_reasoning(self):
        """Context manager: pause perception -> yield for reasoning -> resume perception."""
        async with self._lock:
            if self._perception is not None:
                try:
                    await self._perception.pause()
                except Exception:
                    log.warning("Failed to pause perception, continuing anyway")
            try:
                yield
            finally:
                if self._perception is not None:
                    try:
                        await self._perception.resume()
                    except Exception:
                        log.warning("Failed to resume perception")
