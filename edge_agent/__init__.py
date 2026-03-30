"""
Edge Agent: edge-native private AI agent framework.

Dual-system cognitive architecture:
  System 1 (MiniCPM-o) - real-time perception (voice + camera + screen)
  System 2 (Qwen3.5)   - deep reasoning + tool calling
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, TYPE_CHECKING

from .events import (
    EventBus,
    UserSpeech,
    ChannelMessage,
    VisualScene,
    ThinkingStarted,
    ReasoningDone,
)
from .state import StateMachine, State
from .scheduler import GPUScheduler
from .router import IntentRouter, KeywordRouter
from .memory import MemoryStore
from .tools import ToolRegistry
from .tools_builtin.system import set_memory_store

if TYPE_CHECKING:
    from .providers import PerceptionProvider, ReasoningProvider
    from .channels import Channel

log = logging.getLogger(__name__)

__version__ = "0.1.0"


class EdgeAgent:
    """
    Top-level orchestrator: composes all components and drives the main loop.

    Usage:
        agent = EdgeAgent(
            reasoning=OllamaProvider(model="qwen3.5:27b"),
            tools=[web_search, shell, read_file, memory_save],
            memory_dir="./memory",
        )
        await agent.run(port=8080)
    """

    def __init__(
        self,
        perception: PerceptionProvider | None = None,
        reasoning: ReasoningProvider | None = None,
        router: IntentRouter | None = None,
        tools: list[Callable] | None = None,
        memory_dir: str = "./memory",
        channels: list[Channel] | None = None,
    ) -> None:
        self.perception = perception
        self.reasoning = reasoning
        self.router = router or KeywordRouter()
        self.bus = EventBus()
        self.sm = StateMachine()
        self.scheduler = GPUScheduler(perception)
        self.memory = MemoryStore(memory_dir)
        self.tool_registry = ToolRegistry(tools)
        self.channels = channels or []

        self._last_visual = ""

        set_memory_store(self.memory)

        self.bus.on(UserSpeech, self._handle_user_speech)
        self.bus.on(ChannelMessage, self._handle_channel_message)
        self.bus.on(VisualScene, self._handle_visual_update)

    # -- Event handlers --------------------------------------------------------

    async def _handle_user_speech(self, event: UserSpeech) -> None:
        self._set_state(State.ROUTING)
        self.memory.append_turn("user", event.text)
        intent = self.router.classify(event.text, self._last_visual)
        log.info("Intent: %s for '%s'", intent, event.text[:60])

        if intent == "slow":
            await self._delegate_to_system2(event.text, reply_channel=None)
        else:
            self._set_state(State.LISTENING)

    async def _handle_channel_message(self, event: ChannelMessage) -> None:
        self._set_state(State.ROUTING)
        self.memory.append_turn("user", event.text)
        log.info("[%s/%s] %s", event.channel, event.sender, event.text[:60])

        target_channel = None
        for ch in self.channels:
            if ch.name == event.channel:
                target_channel = ch
                break

        await self._delegate_to_system2(event.text, reply_channel=target_channel)

    async def _handle_visual_update(self, event: VisualScene) -> None:
        self._last_visual = event.description

    # -- Core reasoning delegation ---------------------------------------------

    async def _delegate_to_system2(self, text: str, reply_channel=None) -> None:
        if self.reasoning is None:
            msg = "System 2 (reasoning) not configured."
            log.warning(msg)
            if reply_channel:
                await reply_channel.send(msg)
            return

        self._set_state(State.THINKING)
        await self.bus.emit(ThinkingStarted(query=text))

        async with self.scheduler.use_reasoning():
            context = self.memory.build_context(query=text)
            full_message = text
            if self._last_visual:
                full_message += f"\n\n[Current visual scene: {self._last_visual}]"

            result = await self.reasoning.reason(
                message=full_message,
                context=context,
                tools=self.tool_registry.definitions(),
                tool_executor=self.tool_registry.execute,
            )

        self.memory.append_turn("assistant", result.text)
        await self.bus.emit(ReasoningDone(text=result.text, tools_used=result.tools_used))
        self._set_state(State.SPEAKING)

        if reply_channel:
            await reply_channel.send(result.text)

        if self.perception is not None:
            try:
                summary = result.text[:200]
                await self.perception.inject_context(f"[Completed: {summary}]")
            except Exception:
                log.debug("inject_context not available")

        self._set_state(State.LISTENING)

    def _set_state(self, new_state: str) -> None:
        try:
            if self.sm.state != new_state:
                self.sm.transition(new_state)
        except Exception:
            log.debug("Ignored invalid state transition: %s -> %s", self.sm.state, new_state)

    # -- Run -------------------------------------------------------------------

    async def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        log.info("EdgeAgent v%s starting...", __version__)

        if self.perception is not None:
            self._set_state(State.LISTENING)
            system_prompt = self.memory.build_system_prompt()
            await self.perception.start(system_prompt)
            log.info("System 1 (perception) started.")

        if self.reasoning is not None:
            healthy = await self.reasoning.health()
            log.info("System 2 (reasoning) health: %s", healthy)

        for ch in self.channels:
            await ch.start(self.bus)
            log.info("Channel '%s' started.", ch.name)

        log.info("EdgeAgent ready. Tools: %s", self.tool_registry.names)
        log.info("Memory dir: %s", self.memory.base)

        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("EdgeAgent shutting down.")
