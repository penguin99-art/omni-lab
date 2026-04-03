"""Tests for OllamaProvider with mocked Ollama client."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from edge_agent.context import RenderedContext
from edge_agent.providers.ollama import OllamaProvider


def _make_response(content: str = "test response", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    resp = MagicMock()
    resp.message = msg
    return resp


class TestOllamaProvider:
    @pytest.mark.asyncio
    async def test_health_success(self):
        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.list = AsyncMock(return_value={"models": []})
        assert await provider.health() is True

    @pytest.mark.asyncio
    async def test_health_failure(self):
        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.list = AsyncMock(side_effect=ConnectionError("offline"))
        provider._ensure_client = AsyncMock(side_effect=ConnectionError("offline"))
        assert await provider.health() is False

    @pytest.mark.asyncio
    async def test_reason_simple(self):
        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.chat = AsyncMock(return_value=_make_response("你好！"))

        ctx = RenderedContext(
            system_prompt="你是AI",
            messages=[
                {"role": "system", "content": "你是AI"},
                {"role": "user", "content": "你好"},
            ],
        )
        result = await provider.reason(
            message="你好",
            context=ctx,
            tools=[],
        )
        assert result.text == "你好！"
        assert result.tools_used == []

    @pytest.mark.asyncio
    async def test_reason_with_tool_call(self):
        fn_obj = MagicMock()
        fn_obj.name = "web_search"
        fn_obj.arguments = {"query": "test"}
        tc = MagicMock()
        tc.function = fn_obj

        step1 = _make_response(content="", tool_calls=[tc])
        step2 = _make_response(content="搜索结果是...")

        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.chat = AsyncMock(side_effect=[step1, step2])

        async def mock_executor(name, args):
            return "mock search result"

        ctx = RenderedContext(
            system_prompt="你是AI",
            messages=[{"role": "system", "content": "你是AI"}],
        )
        result = await provider.reason(
            message="帮我搜索",
            context=ctx,
            tools=[{"type": "function", "function": {"name": "web_search"}}],
            tool_executor=mock_executor,
        )
        assert result.text == "搜索结果是..."
        assert len(result.tools_used) == 1
        assert result.tools_used[0]["tool"] == "web_search"

    @pytest.mark.asyncio
    async def test_reason_retry_on_failure(self):
        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()

        call_count = 0
        async def flaky_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("connection lost")
            return _make_response("recovered!")

        provider._client.chat = flaky_chat
        provider._client.list = AsyncMock(return_value={"models": []})

        ctx = RenderedContext(
            system_prompt="你是AI",
            messages=[{"role": "system", "content": "你是AI"}],
        )
        result = await provider.reason(message="test", context=ctx, tools=[])
        assert result.text == "recovered!"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_reason_all_retries_fail(self):
        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.chat = AsyncMock(side_effect=ConnectionError("offline"))
        provider._client.list = AsyncMock(side_effect=ConnectionError("offline"))

        ctx = RenderedContext(
            system_prompt="你是AI",
            messages=[{"role": "system", "content": "你是AI"}],
        )
        result = await provider.reason(message="test", context=ctx, tools=[])
        assert "retries" in result.text.lower()

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        fn_obj = MagicMock()
        fn_obj.name = "tool"
        fn_obj.arguments = {}
        tc = MagicMock()
        tc.function = fn_obj

        provider = OllamaProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.chat = AsyncMock(
            return_value=_make_response(content="", tool_calls=[tc])
        )

        async def mock_executor(name, args):
            return "result"

        ctx = RenderedContext(
            system_prompt="你是AI",
            messages=[{"role": "system", "content": "你是AI"}],
        )
        result = await provider.reason(
            message="test", context=ctx, tools=[{"type": "function", "function": {"name": "tool"}}],
            max_iterations=3, tool_executor=mock_executor,
        )
        assert "max iterations" in result.text.lower()
