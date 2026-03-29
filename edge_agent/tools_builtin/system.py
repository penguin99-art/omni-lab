"""System tools: memory_save, cron_add, send_message."""

from __future__ import annotations

from edge_agent.tools import tool

_memory_store = None


def set_memory_store(store) -> None:
    """Called by EdgeAgent to inject the shared MemoryStore instance."""
    global _memory_store
    _memory_store = store


@tool("保存一条重要事实到长期记忆")
def memory_save(fact: str) -> str:
    if _memory_store is None:
        return "MemoryStore 未初始化"
    _memory_store.save_fact(fact)
    return f"已保存到长期记忆: {fact}"
