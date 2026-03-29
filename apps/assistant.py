#!/usr/bin/env python3
"""
Private AI Assistant - CLI mode.

Runs System 2 (Qwen3.5 via Ollama) with tool calling.
Talk to your AI via terminal. All data stays local.

Usage:
    cd /home/pineapi/gy
    source .venv/bin/activate
    python apps/assistant.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_agent import EdgeAgent
from edge_agent.providers.ollama import OllamaProvider
from edge_agent.channels.cli import CLIChannel
from edge_agent.tools_builtin.web import web_search, web_fetch
from edge_agent.tools_builtin.filesystem import read_file, write_file, edit_file, list_dir
from edge_agent.tools_builtin.shell import shell
from edge_agent.tools_builtin.system import memory_save


agent = EdgeAgent(
    reasoning=OllamaProvider(model="qwen3.5:27b"),
    tools=[web_search, web_fetch, read_file, write_file, edit_file, list_dir, shell, memory_save],
    memory_dir="./memory",
    channels=[CLIChannel()],
)

if __name__ == "__main__":
    asyncio.run(agent.run())
