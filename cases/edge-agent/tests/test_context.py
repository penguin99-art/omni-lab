"""Tests for context assembly and rendering."""

from __future__ import annotations

import pytest

from edge_agent.agent.conversation import ConversationEngine
from edge_agent.context import ContextBuilder, OllamaContextRenderer, TurnInput
from edge_agent.memory import MemoryStore
from edge_agent.providers import ReasoningResult
from edge_agent.tools import ToolPool, tool


@tool("Echo the current text", read_only=True)
def echo_text(text: str) -> str:
    return text


class TestContextBuilder:
    def test_build_snapshot(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store._embedder_checked = True
        store.save_fact("用户更偏爱 Python")
        store.append_turn("assistant", "你好，我在。")

        builder = ContextBuilder(memory=store, tool_pool=ToolPool([echo_text]))
        snapshot = builder.build(
            TurnInput(
                text="帮我回忆一下 Python 偏好",
                channel="websocket",
                sender="browser",
                visual_context="屏幕上打开着 VS Code",
            )
        )

        assert "测试用的 AI 助手" in snapshot.identity
        assert "开发者" in snapshot.user_profile
        assert snapshot.channel_context == "websocket"
        assert snapshot.visual_context == "屏幕上打开着 VS Code"
        assert "echo_text" in snapshot.tool_names
        assert snapshot.relevant_memories
        assert "Python" in snapshot.relevant_memories[0].text

    def test_bootstrap_prompt(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.save_fact("用户喜欢简洁回答")
        builder = ContextBuilder(memory=store, tool_pool=ToolPool())

        prompt = builder.bootstrap_prompt()

        assert "测试用的 AI 助手" in prompt
        assert "开发者" in prompt
        assert "用户喜欢简洁回答" in prompt


class TestOllamaRenderer:
    def test_render_messages(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.append_turn("assistant", "前一轮回复")
        builder = ContextBuilder(memory=store, tool_pool=ToolPool([echo_text]))
        renderer = OllamaContextRenderer()

        snapshot = builder.build(TurnInput(text="这轮问题", channel="cli"))
        rendered = renderer.render(snapshot)

        assert rendered.messages[0]["role"] == "system"
        assert "Enabled tools" in rendered.messages[0]["content"]
        assert rendered.messages[-1] == {"role": "user", "content": "这轮问题"}
        assert rendered.debug["channel"] == "cli"


class TestConversationEngine:
    @pytest.mark.asyncio
    async def test_submit_message_uses_context_pipeline(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.save_fact("用户今天要写重构方案")
        engine = ConversationEngine(
            tool_pool=ToolPool([echo_text]),
            memory=store,
            reason_fn=self._fake_reason,
        )

        result = await engine.submit_message(
            "帮我总结一下今天的重点",
            channel="cli",
            visual_context="桌面上有架构图",
        )

        assert result.text == "已收到"
        turns = store.recent_turns()
        assert turns[-2]["role"] == "user"
        assert turns[-1]["role"] == "assistant"

    @staticmethod
    async def _fake_reason(message, context, tools, tool_executor=None, max_iterations=20):
        assert message == "帮我总结一下今天的重点"
        assert context.messages[0]["role"] == "system"
        assert "桌面上有架构图" in context.system_prompt
        assert "用户今天要写重构方案" in context.system_prompt
        return ReasoningResult(text="已收到", tools_used=[])
