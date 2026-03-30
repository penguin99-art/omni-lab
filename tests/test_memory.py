"""Tests for MemoryStore: save_fact, parse_facts, search_memory, build_context."""

from __future__ import annotations

import pytest
from edge_agent.memory import MemoryStore, Context


class TestMemoryStore:
    def test_init_creates_dir(self, tmp_path):
        d = tmp_path / "new_memory"
        store = MemoryStore(str(d))
        assert d.exists()

    def test_soul_and_profile(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        assert "测试" in store.soul()
        assert "开发者" in store.user_profile()

    def test_empty_long_term(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        assert store.long_term_memory() == ""

    def test_save_fact_creates_file(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.save_fact("Python 3.11 是首选版本")
        content = (tmp_memory_dir / "MEMORY.md").read_text(encoding="utf-8")
        assert "Python 3.11" in content
        assert "长期记忆" in content

    def test_save_multiple_facts(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.save_fact("事实 A")
        store.save_fact("事实 B")
        store.save_fact("事实 C")
        content = (tmp_memory_dir / "MEMORY.md").read_text(encoding="utf-8")
        assert "事实 A" in content
        assert "事实 B" in content
        assert "事实 C" in content

    def test_parse_facts(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.save_fact("Alpha")
        store.save_fact("Beta")
        facts = store._parse_facts()
        assert len(facts) == 2
        assert "Alpha" in facts[0]
        assert "Beta" in facts[1]

    def test_parse_facts_empty(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        facts = store._parse_facts()
        assert facts == []

    def test_keyword_search(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.save_fact("Python 是一门编程语言")
        store.save_fact("Rust 是系统编程语言")
        store.save_fact("下周三要开技术分享会")
        results = store._keyword_search("Python 编程", store._parse_facts(), top_k=3)
        assert len(results) > 0
        assert "Python" in results[0][0]

    def test_search_memory_without_embedder(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store._embedder_checked = True
        store.save_fact("GPU 显存占用 16GB")
        store.save_fact("下周三的会议")
        results = store.search_memory("GPU 显存")
        assert len(results) > 0

    def test_append_turn(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.append_turn("user", "你好")
        store.append_turn("assistant", "你好！有什么可以帮你的？")
        turns = store.recent_turns()
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_turn_overflow(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir), max_turns=10)
        for i in range(15):
            store.append_turn("user", f"消息 {i}")
        turns = store.recent_turns(n=100)
        assert len(turns) <= 10
        assert turns[0]["text"] == "消息 6"
        assert turns[-1]["text"] == "消息 14"

    def test_build_context(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.append_turn("user", "你好")
        ctx = store.build_context()
        assert isinstance(ctx, Context)
        assert "测试" in ctx.soul
        assert len(ctx.recent_turns) == 1

    def test_build_context_with_query(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store._embedder_checked = True
        store.save_fact("GPU 占用 16GB")
        store.save_fact("今天天气很好")
        ctx = store.build_context(query="GPU")
        assert "GPU" in ctx.long_term

    def test_build_system_prompt(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        prompt = store.build_system_prompt()
        assert "测试" in prompt

    def test_context_to_ollama_messages(self, tmp_memory_dir):
        store = MemoryStore(str(tmp_memory_dir))
        store.append_turn("user", "测试消息")
        ctx = store.build_context()
        msgs = ctx.to_ollama_messages()
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
