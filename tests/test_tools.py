"""Tests for ToolRegistry: registration, schema generation, execution."""

from __future__ import annotations

import asyncio
import pytest
from edge_agent.tools import ToolRegistry, tool


@tool("A test tool that adds numbers")
def add(a: int, b: int) -> str:
    return str(a + b)


@tool("A test tool that greets")
def greet(name: str) -> str:
    return f"Hello, {name}!"


@tool("A test tool with optional param")
def optional_tool(text: str, uppercase: bool = False) -> str:
    return text.upper() if uppercase else text


class TestToolRegistry:
    def test_register(self):
        reg = ToolRegistry([add, greet])
        assert "add" in reg.names
        assert "greet" in reg.names
        assert len(reg.names) == 2

    def test_register_single(self):
        reg = ToolRegistry()
        reg.register(add)
        assert "add" in reg.names

    def test_definitions_structure(self):
        reg = ToolRegistry([add])
        defs = reg.definitions()
        assert len(defs) == 1
        d = defs[0]
        assert d["type"] == "function"
        assert d["function"]["name"] == "add"
        assert "a" in d["function"]["parameters"]["properties"]
        assert "b" in d["function"]["parameters"]["properties"]
        assert "a" in d["function"]["parameters"]["required"]

    def test_definitions_types(self):
        reg = ToolRegistry([add])
        defs = reg.definitions()
        props = defs[0]["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "integer"

    def test_definitions_optional_param(self):
        reg = ToolRegistry([optional_tool])
        defs = reg.definitions()
        required = defs[0]["function"]["parameters"]["required"]
        assert "text" in required
        assert "uppercase" not in required

    def test_definitions_description(self):
        reg = ToolRegistry([add])
        defs = reg.definitions()
        assert defs[0]["function"]["description"] == "A test tool that adds numbers"

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        reg = ToolRegistry([add])
        result = await reg.execute("add", {"a": 3, "b": 5})
        assert result == "8"

    @pytest.mark.asyncio
    async def test_execute_greet(self):
        reg = ToolRegistry([greet])
        result = await reg.execute("greet", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry([add])
        result = await reg.execute("nonexistent", {})
        assert "未知工具" in result

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        @tool("fails")
        def bad_tool() -> str:
            raise ValueError("intentional error")

        reg = ToolRegistry([bad_tool])
        result = await reg.execute("bad_tool", {})
        assert "工具执行失败" in result

    def test_empty_registry(self):
        reg = ToolRegistry()
        assert reg.names == []
        assert reg.definitions() == []
