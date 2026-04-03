"""
Edge Agent: edge-native private AI agent framework.

Dual-system cognitive architecture:
  System 1 (MiniCPM-o) - real-time perception (voice + camera + screen)
  System 2 (Qwen3.5)   - deep reasoning + tool calling

Architecture inspired by claude-code:
  - ToolPool with rich Tool protocol, parallel orchestration, result limits
  - ConversationEngine: decoupled from transport, owns message history
  - query_loop: explicit state machine with QueryDeps injection
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Callable, TYPE_CHECKING

from .context import ContextBuilder, OllamaContextRenderer, TurnInput
from .events import (
    EventBus,
    UserSpeech,
    ChannelMessage,
    VisualScene,
    ThinkingStarted,
    ReasoningDone,
)
from .state import StateMachine, State
from .errors import InvalidTransition
from .scheduler import GPUScheduler
from .router import IntentRouter, KeywordRouter
from .memory import MemoryStore
from .tools import ToolPool
from .agent.conversation import ConversationEngine
from .tools_builtin.system import set_memory_store

if TYPE_CHECKING:
    from .providers import PerceptionProvider, ReasoningProvider
    from .channels import Channel

log = logging.getLogger(__name__)

__version__ = "0.3.0"


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
        max_iterations: int = 20,
        tool_deny: list[str] | None = None,
    ) -> None:
        self.perception = perception
        self.reasoning = reasoning
        self.router = router or KeywordRouter()
        self.bus = EventBus()
        self.sm = StateMachine()
        self.scheduler = GPUScheduler(perception)
        self.memory = MemoryStore(memory_dir)
        self.tool_pool = ToolPool(tools, deny=tool_deny)
        self.channels: list[Channel] = channels or []
        self.context_builder = ContextBuilder(memory=self.memory, tool_pool=self.tool_pool)
        self.context_renderer = OllamaContextRenderer()

        self._last_visual = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

        self._conversation: ConversationEngine | None = None
        self._max_iterations = max_iterations

        set_memory_store(self.memory)

        self.bus.on(UserSpeech, self._handle_user_speech)
        self.bus.on(ChannelMessage, self._handle_channel_message)
        self.bus.on(VisualScene, self._handle_visual_update)

    # -- backward compat alias -------------------------------------------------
    @property
    def tool_registry(self) -> ToolPool:
        return self.tool_pool

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("EdgeAgent.run() has not been called yet")
        return self._loop

    # -- Event handlers --------------------------------------------------------

    async def _handle_user_speech(self, event: UserSpeech) -> None:
        self._set_state(State.ROUTING)
        intent = self.router.classify(event.text, self._last_visual)
        log.info("Intent: %s for '%s'", intent, event.text[:60])

        if intent == "slow":
            await self._delegate_to_system2(event.text, reply_channel=None)
        else:
            self.memory.add_user(event.text)
            self._set_state(State.LISTENING)

    async def _handle_channel_message(self, event: ChannelMessage) -> None:
        self._set_state(State.ROUTING)
        log.info("[%s/%s] %s", event.channel, event.sender, event.text[:60])

        target_channel = None
        for ch in self.channels:
            if ch.name == event.channel:
                target_channel = ch
                break

        await self._delegate_to_system2(event.text, reply_channel=target_channel)

    async def _handle_visual_update(self, event: VisualScene) -> None:
        self._last_visual = event.description

    # -- Core reasoning delegation (now uses ConversationEngine) ----------------

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
            if self._conversation:
                turn = await self._conversation.submit_message(
                    text,
                    channel=getattr(reply_channel, "name", "internal"),
                    visual_context=self._last_visual,
                )
                result_text = turn.text
                tools_used = turn.tools_used
            else:
                snapshot = self.context_builder.build(
                    TurnInput(
                        text=text,
                        channel=getattr(reply_channel, "name", "internal"),
                        visual_context=self._last_visual,
                    )
                )
                context = self.context_renderer.render(snapshot)
                result = await self.reasoning.reason(
                    message=text,
                    context=context,
                    tools=self.tool_pool.definitions(),
                    tool_executor=self.tool_pool.execute,
                )
                self.memory.add_user(text)
                result_text = result.text
                tools_used = result.tools_used
                if result_text:
                    self.memory.add_assistant(result_text)

        await self.bus.emit(ReasoningDone(text=result_text, tools_used=tools_used))
        self._set_state(State.SPEAKING)

        if reply_channel:
            await reply_channel.send(result_text)

        if self.perception is not None:
            try:
                summary = result_text[:200]
                await self.perception.inject_context(f"[Completed: {summary}]")
            except Exception:
                log.debug("inject_context not available")

        self._set_state(State.LISTENING)

    def _set_state(self, new_state: str) -> None:
        try:
            if self.sm.state != new_state:
                self.sm.transition(new_state)
        except InvalidTransition as e:
            log.warning("Invalid state transition: %s", e)

    # -- Lifecycle -------------------------------------------------------------

    async def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        log.info("EdgeAgent v%s starting...", __version__)

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()

        for sig_name in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig_name, self._stop_event.set)

        if self.reasoning is not None:
            self._conversation = ConversationEngine(
                tool_pool=self.tool_pool,
                memory=self.memory,
                reason_fn=self.reasoning.reason,
                max_iterations=self._max_iterations,
                context_builder=self.context_builder,
                context_renderer=self.context_renderer,
            )

        if self.perception is not None:
            self._set_state(State.LISTENING)
            system_prompt = self.context_builder.bootstrap_prompt()
            await self.perception.start(system_prompt)
            log.info("System 1 (perception) started.")

        if self.reasoning is not None:
            healthy = await self.reasoning.health()
            log.info("System 2 (reasoning) health: %s", healthy)

        for ch in self.channels:
            await ch.start(self.bus)
            log.info("Channel '%s' started.", ch.name)

        log.info("EdgeAgent ready. Tools: %s", self.tool_pool.names)
        log.info("Memory dir: %s", self.memory.base)

        await self._stop_event.wait()
        await self._shutdown()

    async def _shutdown(self) -> None:
        log.info("EdgeAgent shutting down...")

        for ch in self.channels:
            try:
                await ch.stop()
            except Exception:
                log.debug("Channel '%s' stop error (ignored)", getattr(ch, 'name', '?'))

        if self.perception is not None:
            try:
                await self.perception.pause()
            except Exception:
                log.debug("Perception pause error (ignored)")

        self.memory.flush()
        log.info("EdgeAgent shutdown complete.")

    def request_stop(self) -> None:
        if self._stop_event and self._loop:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def abort_reasoning(self) -> None:
        """Abort the current reasoning turn (thread-safe)."""
        if self._conversation:
            self._conversation.abort()
        if self.reasoning and hasattr(self.reasoning, "abort"):
            self.reasoning.abort()
