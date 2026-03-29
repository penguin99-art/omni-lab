#!/usr/bin/env python3
"""
Private Second Brain - Full dual-system mode.

System 1 (MiniCPM-o): real-time voice + camera perception
System 2 (Qwen3.5):   deep reasoning + tool calling + memory

Usage:
    # Terminal 1: Start Ollama
    ollama serve

    # Terminal 2: Start llama-server (MiniCPM-o)
    ./scripts/start.sh

    # Terminal 3: Start Second Brain
    cd /home/pineapi/gy
    source .venv/bin/activate
    python apps/second_brain.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_agent import EdgeAgent
from edge_agent.providers.minicpm import MiniCPMProvider
from edge_agent.providers.ollama import OllamaProvider
from edge_agent.router import KeywordRouter
from edge_agent.tools_builtin.web import web_search, web_fetch
from edge_agent.tools_builtin.filesystem import read_file, write_file, edit_file, list_dir
from edge_agent.tools_builtin.shell import shell
from edge_agent.tools_builtin.computer import screenshot
from edge_agent.tools_builtin.system import memory_save


MEMORY_TRIGGERS = [
    "remember", "save", "note",
    "last time", "before", "what was",
    "search", "find", "look up",
    "help me", "help",
    "open", "run", "execute", "create",
    "summarize", "translate",
]


class SecondBrainRouter(KeywordRouter):
    TRIGGERS = MEMORY_TRIGGERS + KeywordRouter.TRIGGERS


agent = EdgeAgent(
    perception=MiniCPMProvider(
        server_url="http://localhost:9060",
        model_dir="./models/MiniCPM-o-4_5-gguf",
        ref_audio="./official_ref_audio.wav",
    ),
    reasoning=OllamaProvider(model="qwen3.5:27b"),
    router=SecondBrainRouter(),
    tools=[
        web_search, web_fetch,
        read_file, write_file, edit_file, list_dir,
        shell, screenshot, memory_save,
    ],
    memory_dir="./memory",
)


async def main():
    from edge_agent.channels.websocket import WebSocketChannel
    ws_channel = WebSocketChannel(agent, port=8080)
    agent.channels.append(ws_channel)
    await agent.run(port=8080)


if __name__ == "__main__":
    asyncio.run(main())
