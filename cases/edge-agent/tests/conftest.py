"""Shared fixtures for edge-agent tests."""

from __future__ import annotations

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Provide a temporary memory directory with sample files."""
    soul_path = tmp_path / "SOUL.md"
    soul_path.write_text("你是一个测试用的 AI 助手。", encoding="utf-8")

    user_path = tmp_path / "USER.md"
    user_path.write_text("用户是一个开发者。", encoding="utf-8")

    return tmp_path
