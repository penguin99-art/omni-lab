#!/usr/bin/env python3
"""
ai — agent-native workstation assistant.

Usage:
    python -m ai              Interactive chat
    python -m ai morning      Generate today's plan
    python -m ai evening      Summarize the day

State lives in ~/.ai/ as plain Markdown files.
"""

import os
import sys
from pathlib import Path
from datetime import date

from .prompts import build_system_prompt
from .engine import run_turn

MODEL = os.environ.get("AI_MODEL", "qwen3.5:27b")
AI_DIR = Path(os.environ.get("AI_DIR", Path.home() / ".ai"))
GOALS_FILE = AI_DIR / "GOALS.md"
MEMORY_FILE = AI_DIR / "MEMORY.md"
TODAY_FILE = AI_DIR / f"{date.today().isoformat()}.md"


def _prompt(mode: str) -> str:
    return build_system_prompt(
        mode,
        goals_file=GOALS_FILE,
        memory_file=MEMORY_FILE,
        today_file=TODAY_FILE,
        state_dir=AI_DIR,
    )


def _turn(messages: list) -> str:
    return run_turn(
        messages, model=MODEL, memory_file=MEMORY_FILE, today_file=TODAY_FILE,
    )


def _run_routine(mode: str) -> None:
    messages = [
        {"role": "system", "content": _prompt(mode)},
        {"role": "user", "content": "开始吧。"},
    ]
    result = _turn(messages)
    print(f"\n{result}\n")


def chat_loop() -> None:
    messages = [{"role": "system", "content": _prompt("chat")}]

    print(f"[ai] Ready. Model: {MODEL} | State: {AI_DIR}")
    print(f"[ai] Type 'q' to quit, 'morning' / 'evening' for routines.\n")

    while True:
        try:
            user_input = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break
        if user_input.lower() in ("morning", "evening"):
            _run_routine(user_input.lower())
            continue

        messages.append({"role": "user", "content": user_input})
        result = _turn(messages)
        print(f"\n{result}\n")


def main() -> None:
    AI_DIR.mkdir(parents=True, exist_ok=True)
    if not GOALS_FILE.exists():
        GOALS_FILE.write_text("# 我的目标\n\n## 本周\n\n- \n", encoding="utf-8")
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("# 记忆\n", encoding="utf-8")
    if not TODAY_FILE.exists():
        TODAY_FILE.write_text(f"# {date.today().isoformat()}\n", encoding="utf-8")

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "chat"

    if mode in ("morning", "evening"):
        _run_routine(mode)
    else:
        chat_loop()


if __name__ == "__main__":
    main()
