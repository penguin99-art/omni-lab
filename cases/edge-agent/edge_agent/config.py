"""Centralized configuration for edge-agent.

All hardcoded URLs, ports, model names, and tuning knobs live here.
Values are read from environment variables (if set) with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EdgeConfig:
    """Single source of truth for every tuneable parameter."""

    # -- Ollama (System 2) -----------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:27b"

    # -- MiniCPM-o (System 1) --------------------------------------------------
    minicpm_url: str = "http://localhost:9060"
    minicpm_model_dir: str = "./models/MiniCPM-o-4_5-gguf"
    minicpm_ref_audio: str = "./official_ref_audio.wav"
    minicpm_max_chunks: int = 300

    # -- Networking ------------------------------------------------------------
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    api_port: int = 8000

    # -- Memory ----------------------------------------------------------------
    memory_dir: str = "./memory"
    memory_max_turns: int = 100

    # -- Reasoning tuning ------------------------------------------------------
    max_react_iterations: int = 20
    ollama_max_retries: int = 3
    ollama_retry_delay: float = 2.0

    @classmethod
    def from_env(cls) -> "EdgeConfig":
        """Build config by overlaying environment variables on defaults."""
        defaults = cls()
        return cls(
            ollama_url=os.getenv("EA_OLLAMA_URL", defaults.ollama_url),
            ollama_model=os.getenv("EA_OLLAMA_MODEL", defaults.ollama_model),
            minicpm_url=os.getenv("EA_MINICPM_URL", defaults.minicpm_url),
            minicpm_model_dir=os.getenv("EA_MINICPM_MODEL_DIR", defaults.minicpm_model_dir),
            minicpm_ref_audio=os.getenv("EA_MINICPM_REF_AUDIO", defaults.minicpm_ref_audio),
            minicpm_max_chunks=int(os.getenv("EA_MINICPM_MAX_CHUNKS", str(defaults.minicpm_max_chunks))),
            web_host=os.getenv("EA_WEB_HOST", defaults.web_host),
            web_port=int(os.getenv("EA_WEB_PORT", str(defaults.web_port))),
            api_port=int(os.getenv("EA_API_PORT", str(defaults.api_port))),
            memory_dir=os.getenv("EA_MEMORY_DIR", defaults.memory_dir),
            memory_max_turns=int(os.getenv("EA_MEMORY_MAX_TURNS", str(defaults.memory_max_turns))),
            max_react_iterations=int(os.getenv("EA_MAX_REACT_ITERATIONS", str(defaults.max_react_iterations))),
            ollama_max_retries=int(os.getenv("EA_OLLAMA_MAX_RETRIES", str(defaults.ollama_max_retries))),
            ollama_retry_delay=float(os.getenv("EA_OLLAMA_RETRY_DELAY", str(defaults.ollama_retry_delay))),
        )
