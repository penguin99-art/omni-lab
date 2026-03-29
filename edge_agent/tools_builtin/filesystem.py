"""Filesystem tools: read, write, edit, list."""

from __future__ import annotations

import os
from pathlib import Path

from edge_agent.tools import tool


@tool("读取文件内容")
def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"文件不存在: {path}"
    if not p.is_file():
        return f"不是文件: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 50_000:
            return content[:50_000] + f"\n\n... (截断, 原文共 {len(content)} 字符)"
        return content
    except Exception as e:
        return f"读取失败: {e}"


@tool("写入内容到文件（会覆盖已有内容）")
def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字符到 {path}"
    except Exception as e:
        return f"写入失败: {e}"


@tool("编辑文件：将 old_text 替换为 new_text")
def edit_file(path: str, old_text: str, new_text: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"文件不存在: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            return f"未找到要替换的文本。"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"已完成替换。"
    except Exception as e:
        return f"编辑失败: {e}"


@tool("列出目录内容")
def list_dir(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"目录不存在: {path}"
    if not p.is_dir():
        return f"不是目录: {path}"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = []
        for e in entries[:100]:
            prefix = "📁 " if e.is_dir() else "📄 "
            size = ""
            if e.is_file():
                sz = e.stat().st_size
                if sz < 1024:
                    size = f" ({sz}B)"
                elif sz < 1024 * 1024:
                    size = f" ({sz // 1024}KB)"
                else:
                    size = f" ({sz // (1024*1024)}MB)"
            lines.append(f"{prefix}{e.name}{size}")
        result = "\n".join(lines)
        if len(entries) > 100:
            result += f"\n... 共 {len(entries)} 项"
        return result
    except Exception as e:
        return f"列出失败: {e}"
