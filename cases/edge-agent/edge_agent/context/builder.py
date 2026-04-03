"""Turn-time context assembly.

Builds a provider-agnostic snapshot from memory, channel metadata, and
the current user input before any provider-specific rendering happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import ContextSnapshot, MemoryHit, TurnInput
from ..memory import MemoryStore
from ..tools import ToolPool


@dataclass
class ContextBuilder:
    """Collect and normalize all context relevant to a single turn."""

    memory: MemoryStore
    tool_pool: ToolPool
    recent_turns: int = 12
    memory_top_k: int = 8

    def build(self, turn: TurnInput | None = None) -> ContextSnapshot:
        memory_hits = self._select_memories(turn.text if turn else "")
        tool_names = list(self.tool_pool.names)
        runtime_notes: list[str] = []

        if turn and turn.channel:
            runtime_notes.append(f"Current channel: {turn.channel}")
        if turn and turn.sender:
            runtime_notes.append(f"Current sender: {turn.sender}")
        if turn and turn.visual_context:
            runtime_notes.append("Visual context is available for this turn.")
        if tool_names:
            runtime_notes.append(f"Enabled tools: {', '.join(tool_names)}")

        return ContextSnapshot(
            identity=self.memory.soul(),
            user_profile=self.memory.user_profile(),
            recent_turns=self.memory.recent_turns(self.recent_turns),
            relevant_memories=memory_hits,
            visual_context=turn.visual_context if turn else "",
            channel_context=turn.channel if turn else "",
            tool_names=tool_names,
            runtime_notes=runtime_notes,
            turn=turn,
        )

    def bootstrap_prompt(self) -> str:
        """Render a compact prompt for always-on perception startup."""
        snapshot = self.build()
        parts: list[str] = []
        if snapshot.identity:
            parts.append(snapshot.identity)
        if snapshot.user_profile:
            parts.append(f"关于用户:\n{snapshot.user_profile}")
        if snapshot.relevant_memories:
            parts.append(
                "你记住的事实:\n"
                + "\n".join(f"- {hit.text}" for hit in snapshot.relevant_memories[:10])
            )
        return "\n\n".join(parts) if parts else "你是一个友好的中文助手。"

    def _select_memories(self, query: str) -> list[MemoryHit]:
        if query:
            relevant = self.memory.search_memory(query, top_k=self.memory_top_k)
            if relevant:
                return [MemoryHit(text=fact, score=score) for fact, score in relevant]

        facts = self.memory.parse_facts()
        if not facts:
            return []

        recent = facts[-self.memory_top_k :]
        return [MemoryHit(text=fact, score=0.0, source="recent_memory") for fact in recent]
