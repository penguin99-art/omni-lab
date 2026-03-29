"""Three-tier memory system: SOUL.md, USER.md, MEMORY.md + session history."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Context:
    """Immutable snapshot passed to ReasoningProvider."""
    soul: str = ""
    user_profile: str = ""
    long_term: str = ""
    recent_turns: list[dict] = field(default_factory=list)

    def to_ollama_messages(self) -> list[dict]:
        messages: list[dict] = []
        system_parts = []
        if self.soul:
            system_parts.append(self.soul)
        if self.user_profile:
            system_parts.append(f"关于用户:\n{self.user_profile}")
        if self.long_term:
            system_parts.append(f"你记住的事实:\n{self.long_term}")
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for turn in self.recent_turns:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system", "tool"):
                role = "user"
            messages.append({"role": role, "content": turn.get("text", "")})
        return messages


class MemoryStore:
    """
    File layout (convention over configuration):
        memory/
        ├── SOUL.md       identity definition (read-only)
        ├── USER.md       user profile (read-only)
        ├── MEMORY.md     long-term facts (AI writes)
        └── sessions/     session history (auto-managed)
    """

    def __init__(self, base_dir: str = "./memory", max_turns: int = 100) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._turns: list[dict] = []
        self._max_turns = max_turns

    # -- Read identity & memory -----------------------------------------------

    def soul(self) -> str:
        return self._read_md("SOUL.md")

    def user_profile(self) -> str:
        return self._read_md("USER.md")

    def long_term_memory(self) -> str:
        return self._read_md("MEMORY.md")

    # -- Short-term memory (conversation turns) --------------------------------

    def append_turn(self, role: str, text: str) -> None:
        self._turns.append({"role": role, "text": text, "ts": time.time()})
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns // 2:]

    def recent_turns(self, n: int = 20) -> list[dict]:
        return self._turns[-n:]

    # -- Long-term memory write ------------------------------------------------

    def save_fact(self, fact: str) -> None:
        mem_path = self.base / "MEMORY.md"
        header = ""
        if not mem_path.exists():
            header = "# 长期记忆\n"
        with open(mem_path, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(f"\n- [{time.strftime('%Y-%m-%d %H:%M')}] {fact}")

    # -- Build context ---------------------------------------------------------

    def build_context(self, recent_n: int = 20) -> Context:
        return Context(
            soul=self.soul(),
            user_profile=self.user_profile(),
            long_term=self.long_term_memory(),
            recent_turns=self.recent_turns(recent_n),
        )

    def build_system_prompt(self) -> str:
        parts: list[str] = []
        soul = self.soul()
        if soul:
            parts.append(soul)
        user = self.user_profile()
        if user:
            parts.append(f"\n关于用户:\n{user}")
        mem = self.long_term_memory()
        if mem:
            lines = mem.strip().split("\n")
            recent = lines[-20:] if len(lines) > 20 else lines
            parts.append(f"\n你记住的事实:\n" + "\n".join(recent))
        return "\n\n".join(parts) if parts else "你是一个友好的中文助手。"

    # -- Helpers ---------------------------------------------------------------

    def _read_md(self, filename: str) -> str:
        path = self.base / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
