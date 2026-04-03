"""Tests for EventBus and event dataclasses."""

from __future__ import annotations

import asyncio
import pytest
from edge_agent.events import (
    EventBus,
    Event,
    UserSpeech,
    VisualScene,
    ThinkingStarted,
    ReasoningDone,
    ChannelMessage,
    MemoryUpdated,
    HealthCheck,
)


class TestEventDataclasses:
    def test_event_has_timestamp(self):
        e = Event()
        assert e.ts > 0

    def test_user_speech(self):
        e = UserSpeech(text="你好")
        assert e.text == "你好"
        assert e.ts > 0

    def test_visual_scene(self):
        e = VisualScene(description="一个房间")
        assert e.description == "一个房间"

    def test_thinking_started(self):
        e = ThinkingStarted(query="搜索 Python")
        assert e.query == "搜索 Python"

    def test_reasoning_done(self):
        e = ReasoningDone(text="结果", tools_used=[{"tool": "search"}])
        assert e.text == "结果"
        assert len(e.tools_used) == 1

    def test_channel_message(self):
        e = ChannelMessage(channel="cli", sender="user", text="你好")
        assert e.channel == "cli"
        assert e.sender == "user"

    def test_health_check(self):
        e = HealthCheck(system1_ok=True, system2_ok=False)
        assert e.system1_ok is True
        assert e.system2_ok is False


class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_and_handle(self):
        bus = EventBus()
        received = []

        async def handler(event: UserSpeech):
            received.append(event.text)

        bus.on(UserSpeech, handler)
        await bus.emit(UserSpeech(text="测试消息"))
        assert received == ["测试消息"]

    @pytest.mark.asyncio
    async def test_sync_handler(self):
        bus = EventBus()
        received = []

        def handler(event: UserSpeech):
            received.append(event.text)

        bus.on(UserSpeech, handler)
        await bus.emit(UserSpeech(text="同步"))
        assert received == ["同步"]

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = EventBus()
        a, b = [], []

        async def handler_a(event: UserSpeech):
            a.append(event.text)

        async def handler_b(event: UserSpeech):
            b.append(event.text)

        bus.on(UserSpeech, handler_a)
        bus.on(UserSpeech, handler_b)
        await bus.emit(UserSpeech(text="hello"))
        assert a == ["hello"]
        assert b == ["hello"]

    @pytest.mark.asyncio
    async def test_different_event_types(self):
        bus = EventBus()
        speech, visual = [], []

        async def on_speech(event: UserSpeech):
            speech.append(event.text)

        async def on_visual(event: VisualScene):
            visual.append(event.description)

        bus.on(UserSpeech, on_speech)
        bus.on(VisualScene, on_visual)

        await bus.emit(UserSpeech(text="说话"))
        await bus.emit(VisualScene(description="场景"))

        assert speech == ["说话"]
        assert visual == ["场景"]

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        bus = EventBus()
        received = []

        async def handler(event: UserSpeech):
            received.append(event.text)

        bus.on(UserSpeech, handler)
        await bus.emit(UserSpeech(text="before"))
        bus.off(UserSpeech, handler)
        await bus.emit(UserSpeech(text="after"))

        assert received == ["before"]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        bus = EventBus()
        received = []

        async def bad_handler(event: UserSpeech):
            raise ValueError("oops")

        async def good_handler(event: UserSpeech):
            received.append(event.text)

        bus.on(UserSpeech, bad_handler)
        bus.on(UserSpeech, good_handler)
        await bus.emit(UserSpeech(text="still works"))
        assert received == ["still works"]

    @pytest.mark.asyncio
    async def test_emit_no_handlers(self):
        bus = EventBus()
        await bus.emit(UserSpeech(text="nobody listening"))
