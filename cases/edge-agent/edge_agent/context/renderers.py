"""Provider-specific rendering for context snapshots."""

from __future__ import annotations

from .types import ContextSnapshot, RenderedContext


class OllamaContextRenderer:
    """Render a context snapshot into Ollama chat messages."""

    def render(self, snapshot: ContextSnapshot) -> RenderedContext:
        system_parts: list[str] = []
        if snapshot.identity:
            system_parts.append(snapshot.identity)
        if snapshot.user_profile:
            system_parts.append(f"关于用户:\n{snapshot.user_profile}")
        if snapshot.relevant_memories:
            system_parts.append(
                "你记住的事实:\n"
                + "\n".join(f"- {hit.text}" for hit in snapshot.relevant_memories)
            )
        if snapshot.visual_context:
            system_parts.append(f"当前视觉场景:\n{snapshot.visual_context}")
        if snapshot.runtime_notes:
            system_parts.append(
                "运行时上下文:\n"
                + "\n".join(f"- {note}" for note in snapshot.runtime_notes)
            )

        system_prompt = "\n\n".join(system_parts).strip()
        if not system_prompt:
            system_prompt = "你是一个友好的中文助手。"

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for turn in snapshot.recent_turns:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system", "tool"):
                role = "user"
            messages.append({"role": role, "content": turn.get("text", "")})

        if snapshot.turn is not None:
            messages.append({"role": "user", "content": snapshot.turn.text})

        return RenderedContext(
            system_prompt=system_prompt,
            messages=messages,
            debug={
                "channel": snapshot.channel_context,
                "memory_hits": [hit.text for hit in snapshot.relevant_memories],
                "tool_names": snapshot.tool_names,
                "has_visual_context": bool(snapshot.visual_context),
            },
        )

    def render_system_prompt(self, snapshot: ContextSnapshot) -> str:
        """Render only the bootstrap system prompt for always-on components."""
        return self.render(snapshot).system_prompt
