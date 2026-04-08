#!/usr/bin/env python3
"""
GPU Model Bench — Standardized LLM benchmark for any compute platform.

Auto-detects hardware (DGX Spark, Jetson Orin, Mac Studio, etc.),
filters compatible models by memory, and produces cross-platform
comparable results.

Usage:
    python bench/bench.py --list                        # list models for detected platform
    python bench/bench.py --platform auto --suite quick  # auto-detect & run
    python bench/bench.py --models qwen3.5:35b           # test specific model
    python bench/bench.py --engines ollama --suite standard
    python bench/bench.py --platform jetson-orin-64 --list
"""

import argparse
import ast
import csv
import json
import os
import platform as platform_mod
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

@dataclass
class Platform:
    name: str               # "dgx-spark", "jetson-orin-64", "custom-xyz"
    gpu: str                # "GB10", "Orin", "M4 Ultra"
    gpu_arch: str           # "sm_121", "sm_87", "apple_m4"
    memory_gb: int          # total usable memory (unified or VRAM+RAM)
    bandwidth_gbps: float   # memory bandwidth in GB/s
    cuda_version: str       # "13.0", "12.6", "metal", "unknown"
    cpu: str                # "Grace ARM64", "ARM Cortex-A78AE"
    os_info: str            # "Ubuntu 24.04 aarch64"

    def summary(self) -> str:
        return f"{self.name} ({self.gpu}, {self.memory_gb}GB, {self.bandwidth_gbps} GB/s, CUDA {self.cuda_version})"

    def max_model_gb(self) -> float:
        """Max model size allowing ~20% headroom for KV cache and runtime."""
        return self.memory_gb * 0.80

    @classmethod
    def from_name(cls, name: str) -> "Platform":
        if name not in KNOWN_PLATFORMS:
            avail = ", ".join(sorted(KNOWN_PLATFORMS.keys()))
            raise ValueError(f"Unknown platform '{name}'. Known: {avail}")
        return KNOWN_PLATFORMS[name]

    @classmethod
    def detect(cls) -> "Platform":
        """Auto-detect hardware and match to a known profile or build a custom one."""
        gpu_name = _detect_gpu_name()
        memory_gb = _detect_total_memory_gb()
        cuda_ver = _detect_cuda_version()
        gpu_arch = _detect_gpu_arch()
        cpu_info = _detect_cpu()
        os_info = f"{platform_mod.system()} {platform_mod.release()} {platform_mod.machine()}"

        for p in KNOWN_PLATFORMS.values():
            if p.gpu.lower() in gpu_name.lower():
                return cls(
                    name=p.name, gpu=p.gpu, gpu_arch=p.gpu_arch,
                    memory_gb=memory_gb or p.memory_gb,
                    bandwidth_gbps=p.bandwidth_gbps,
                    cuda_version=cuda_ver or p.cuda_version,
                    cpu=cpu_info or p.cpu, os_info=os_info,
                )

        return cls(
            name=f"custom-{gpu_name.lower().replace(' ', '-')[:20]}",
            gpu=gpu_name or "unknown",
            gpu_arch=gpu_arch or "unknown",
            memory_gb=memory_gb or 16,
            bandwidth_gbps=0,
            cuda_version=cuda_ver or "unknown",
            cpu=cpu_info or "unknown",
            os_info=os_info,
        )


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0]
        return out
    except Exception:
        if platform_mod.system() == "Darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, timeout=5,
                ).strip()
                return out
            except Exception:
                pass
        return "unknown"


def _detect_total_memory_gb() -> int:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2))
        except Exception:
            pass
    return 0


def _detect_cuda_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
        return out.split("\n")[0]
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        m = re.search(r"release (\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""


def _detect_gpu_arch() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0]
        return f"sm_{out.replace('.', '')}"
    except Exception:
        return ""


def _detect_cpu() -> str:
    machine = platform_mod.machine()
    try:
        if platform_mod.system() == "Linux":
            out = subprocess.check_output(
                ["lscpu"], text=True, timeout=5, stderr=subprocess.DEVNULL,
            )
            for line in out.split("\n"):
                if "Model name" in line:
                    return line.split(":", 1)[1].strip()
        elif platform_mod.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=5,
            ).strip()
            return out
    except Exception:
        pass
    return machine


KNOWN_PLATFORMS = {
    "dgx-spark": Platform(
        name="dgx-spark", gpu="GB10", gpu_arch="sm_121",
        memory_gb=128, bandwidth_gbps=273, cuda_version="13.0",
        cpu="Grace ARM64", os_info="Ubuntu 24.04 aarch64",
    ),
    "jetson-orin-nano-8": Platform(
        name="jetson-orin-nano-8", gpu="Orin Nano", gpu_arch="sm_87",
        memory_gb=8, bandwidth_gbps=68, cuda_version="12.6",
        cpu="ARM Cortex-A78AE", os_info="JetPack 6.x aarch64",
    ),
    "jetson-orin-nx-16": Platform(
        name="jetson-orin-nx-16", gpu="Orin NX", gpu_arch="sm_87",
        memory_gb=16, bandwidth_gbps=102, cuda_version="12.6",
        cpu="ARM Cortex-A78AE", os_info="JetPack 6.x aarch64",
    ),
    "jetson-agx-orin-32": Platform(
        name="jetson-agx-orin-32", gpu="Orin", gpu_arch="sm_87",
        memory_gb=32, bandwidth_gbps=204, cuda_version="12.6",
        cpu="ARM Cortex-A78AE", os_info="JetPack 6.x aarch64",
    ),
    "jetson-agx-orin-64": Platform(
        name="jetson-agx-orin-64", gpu="Orin", gpu_arch="sm_87",
        memory_gb=64, bandwidth_gbps=204, cuda_version="12.6",
        cpu="ARM Cortex-A78AE", os_info="JetPack 6.x aarch64",
    ),
    "mac-studio-m4-max": Platform(
        name="mac-studio-m4-max", gpu="M4 Max", gpu_arch="apple_m4",
        memory_gb=128, bandwidth_gbps=546, cuda_version="metal",
        cpu="Apple M4 Max", os_info="macOS arm64",
    ),
    "mac-studio-m4-ultra": Platform(
        name="mac-studio-m4-ultra", gpu="M4 Ultra", gpu_arch="apple_m4",
        memory_gb=192, bandwidth_gbps=800, cuda_version="metal",
        cpu="Apple M4 Ultra", os_info="macOS arm64",
    ),
    "rtx-4090": Platform(
        name="rtx-4090", gpu="RTX 4090", gpu_arch="sm_89",
        memory_gb=24, bandwidth_gbps=1008, cuda_version="12.x",
        cpu="x86_64", os_info="Linux x86_64",
    ),
    "rtx-5090": Platform(
        name="rtx-5090", gpu="RTX 5090", gpu_arch="sm_100",
        memory_gb=32, bandwidth_gbps=1792, cuda_version="12.8",
        cpu="x86_64", os_info="Linux x86_64",
    ),
}


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

SUITES = {
    "quick": [
        {
            "id": "hello",
            "type": "chat",
            "messages": [{"role": "user", "content": "Say hello in 3 languages."}],
            "expected_min_tokens": 20,
        },
    ],
    "standard": [
        {
            "id": "hello",
            "type": "chat",
            "messages": [{"role": "user", "content": "Say hello in 3 languages."}],
            "expected_min_tokens": 20,
        },
        {
            "id": "reasoning",
            "type": "reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
                }
            ],
            "expected_min_tokens": 50,
        },
        {
            "id": "code",
            "type": "code",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function that finds the longest palindromic substring in a given string. Include type hints and a brief docstring.",
                }
            ],
            "expected_min_tokens": 100,
        },
        {
            "id": "long_output",
            "type": "generation",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a 500-word essay about the future of edge AI computing.",
                }
            ],
            "expected_min_tokens": 300,
        },
        {
            "id": "chinese",
            "type": "chat",
            "messages": [
                {
                    "role": "user",
                    "content": "用中文详细解释什么是混合专家模型（MoE），它相比 Dense 模型有什么优势和劣势？",
                }
            ],
            "expected_min_tokens": 150,
        },
    ],
    "toolcall": [
        {
            "id": "tool_weather",
            "type": "tool_call",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Beijing today?",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"},
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "expected_min_tokens": 5,
        },
        {
            "id": "tool_multi",
            "type": "tool_call",
            "messages": [
                {
                    "role": "user",
                    "content": "Read the file /tmp/notes.txt and then search for 'TODO' items in it.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"}
                            },
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_text",
                        "description": "Search for a pattern in text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "pattern": {"type": "string"},
                            },
                            "required": ["text", "pattern"],
                        },
                    },
                },
            ],
            "expected_min_tokens": 5,
        },
    ],
}

TTFT_SHORT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

TTFT_LONG_SYSTEM_PROMPT = """You are a highly capable AI assistant specialized in software engineering.
You have deep expertise in Python, TypeScript, Rust, and Go. You follow best practices
including clean code, proper error handling, comprehensive testing, and security awareness.

When writing code:
- Always include type hints and docstrings
- Prefer maintainable, testable implementations
- Explain important tradeoffs briefly

When debugging:
- Start by reproducing the issue
- Use logs and runtime evidence
- Think step by step before proposing a fix

Current project context: you are helping benchmark LLMs across different compute platforms.
Important metrics are tok/s, TTFT, quality score, memory efficiency, and tool calling accuracy."""

TTFT_CASES = [
    {
        "id": "short_cold",
        "prompt_size": "short",
        "cache_state": "cold",
        "system_prompt": TTFT_SHORT_SYSTEM_PROMPT,
        "user_message": "Summarize prefix caching in one sentence.",
        "warm": False,
    },
    {
        "id": "short_warm",
        "prompt_size": "short",
        "cache_state": "warm",
        "system_prompt": TTFT_SHORT_SYSTEM_PROMPT,
        "user_message": "Summarize prefix caching in one sentence.",
        "warm": True,
    },
    {
        "id": "long_cold",
        "prompt_size": "long",
        "cache_state": "cold",
        "system_prompt": TTFT_LONG_SYSTEM_PROMPT,
        "user_message": "Summarize prefix caching in one sentence.",
        "warm": False,
    },
    {
        "id": "long_warm",
        "prompt_size": "long",
        "cache_state": "warm",
        "system_prompt": TTFT_LONG_SYSTEM_PROMPT,
        "user_message": "Summarize prefix caching in one sentence.",
        "warm": True,
    },
]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    engine: str  # ollama | vllm | sglang
    model_id: str
    size_gb: float
    arch: str  # dense | moe
    active_params: Optional[str] = None
    base_url: str = ""
    notes: str = ""
    setup_cmd: Optional[str] = None
    tool_call_support: str = "unknown"  # yes | partial | no | unknown
    community_toks: Optional[float] = None  # expected tok/s from community data
    compare_group: str = ""


MODELS = [
    # --- Ollama: Qwen 3.5 ---
    ModelConfig("qwen3.5:9b", "ollama", "qwen3.5:9b", 6.6, "dense",
                notes="Lightweight, fast", community_toks=35),
    ModelConfig("qwen3.5:35b", "ollama", "qwen3.5:35b", 23, "moe", "3B",
                notes="MoE 35B/3B active", community_toks=57, tool_call_support="partial",
                compare_group="qwen3.5-35b"),
    ModelConfig("qwen3.5:27b", "ollama", "qwen3.5:27b", 17, "dense",
                notes="Dense 27B, slow but capable", community_toks=13,
                compare_group="qwen3.5-27b"),
    ModelConfig("qwen3.5:122b-a10b", "ollama", "qwen3.5:122b-a10b", 81, "moe", "10B",
                notes="Largest Qwen MoE", community_toks=23, tool_call_support="partial"),
    ModelConfig("qwen3:32b", "ollama", "qwen3:32b", 20, "dense",
                notes="Dense 32B", community_toks=10),

    # --- Ollama: NVIDIA models ---
    ModelConfig("gpt-oss:120b", "ollama", "gpt-oss:120b", 65, "moe",
                notes="NVIDIA GPT-OSS 120B MoE", community_toks=41, tool_call_support="yes",
                compare_group="gpt-oss-120b"),
    ModelConfig("gpt-oss:20b", "ollama", "gpt-oss:20b", 13, "moe",
                notes="NVIDIA GPT-OSS 20B", community_toks=58, tool_call_support="no",
                compare_group="gpt-oss-20b"),
    ModelConfig("nemotron-3-nano", "ollama", "nemotron-3-nano", 24, "moe", "3B",
                notes="30B MoE, fastest on Spark", community_toks=69, tool_call_support="no"),
    ModelConfig("nemotron-3-super", "ollama", "nemotron-3-super", 86, "moe", "12B",
                notes="120B MoE, quality focus", community_toks=20, tool_call_support="partial"),
    ModelConfig("nemotron-cascade-2", "ollama", "nemotron-cascade-2", 24, "moe", "3B",
                notes="30B MoE, needs custom Modelfile. Best speed+quality ratio",
                community_toks=72, tool_call_support="partial",
                setup_cmd="ollama pull hf.co/mradermacher/Nemotron-Cascade-2-30B-A3B-GGUF:Q4_K_M"),

    # --- Ollama: Gemma 4 E2B (2B Effective, Dense PLE) ---
    ModelConfig("gemma4:e2b", "ollama", "gemma4:e2b", 7.2, "dense", "2B",
                notes="Gemma4 E2B Q4_K_M default, 128K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:e2b-q8", "ollama", "gemma4:e2b-it-q8_0", 8.1, "dense", "2B",
                notes="Gemma4 E2B Q8_0 high precision, 128K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:e2b-bf16", "ollama", "gemma4:e2b-it-bf16", 10, "dense", "2B",
                notes="Gemma4 E2B BF16 full precision, 128K ctx", tool_call_support="unknown"),

    # --- Ollama: Gemma 4 E4B (4B Effective, Dense PLE) ---
    ModelConfig("gemma4:e4b", "ollama", "gemma4:e4b", 9.6, "dense", "4B",
                notes="Gemma4 E4B Q4_K_M default, 128K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:e4b-q8", "ollama", "gemma4:e4b-it-q8_0", 12, "dense", "4B",
                notes="Gemma4 E4B Q8_0 high precision, 128K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:e4b-bf16", "ollama", "gemma4:e4b-it-bf16", 16, "dense", "4B",
                notes="Gemma4 E4B BF16 full precision, 128K ctx", tool_call_support="unknown"),

    # --- Ollama: Gemma 4 26B (4B Active MoE) ---
    ModelConfig("gemma4:26b", "ollama", "gemma4:26b", 18, "moe", "4B",
                notes="Gemma4 26B MoE Q4_K_M, 256K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:26b-q8", "ollama", "gemma4:26b-a4b-it-q8_0", 28, "moe", "4B",
                notes="Gemma4 26B MoE Q8_0 high precision, 256K ctx", tool_call_support="unknown"),

    # --- Ollama: Gemma 4 31B (Dense) ---
    ModelConfig("gemma4:31b", "ollama", "gemma4:31b", 20, "dense",
                notes="Gemma4 31B Dense Q4_K_M, 256K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:31b-q8", "ollama", "gemma4:31b-it-q8_0", 34, "dense",
                notes="Gemma4 31B Dense Q8_0, 256K ctx", tool_call_support="unknown"),
    ModelConfig("gemma4:31b-bf16", "ollama", "gemma4:31b-it-bf16", 63, "dense",
                notes="Gemma4 31B Dense BF16 full precision, 256K ctx", tool_call_support="unknown"),

    # --- vLLM Docker (Qwen3.5 official image) ---
    ModelConfig("qwen3.5-27b-fp8-vllm", "vllm", "Qwen/Qwen3.5-27B-FP8", 29, "dense",
                base_url="http://localhost:18080/v1",
                notes="Best tool calling accuracy, slow (~6 tok/s)",
                community_toks=6, tool_call_support="yes",
                compare_group="qwen3.5-27b"),

    # --- vLLM + namake-taro patch (best performance) ---
    ModelConfig("qwen3.5-35b-mxfp4-vllm", "vllm", "Qwen/Qwen3.5-35B-A3B", 35, "moe", "3B",
                base_url="http://localhost:18080/v1",
                notes="MXFP4 quantized, requires namake-taro patch",
                community_toks=60, tool_call_support="partial",
                compare_group="qwen3.5-35b"),
    ModelConfig("gpt-oss-120b-mxfp4-vllm", "vllm", "openai/gpt-oss-120b", 65, "moe",
                base_url="http://localhost:18080/v1",
                notes="MXFP4 quantized, requires namake-taro patch",
                community_toks=81,
                compare_group="gpt-oss-120b"),

    # --- SGLang Docker (sglang:spark) ---
    ModelConfig("gpt-oss-20b-sglang", "sglang", "openai/gpt-oss-20b", 13, "moe",
                base_url="http://localhost:18080/v1",
                notes="SGLang optimized for Spark",
                community_toks=61, tool_call_support="no",
                compare_group="gpt-oss-20b"),
]

MODEL_BY_NAME = {m.name: m for m in MODELS}


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------

def get_ollama_base_url():
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def check_ollama_model(model_id: str, base_url: str = "") -> bool:
    """Check if model is available in Ollama."""
    url = base_url or get_ollama_base_url()
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model_id in m for m in models)
    except Exception:
        return False


def check_openai_endpoint(base_url: str, model_id: str) -> bool:
    """Check if vLLM/SGLang endpoint is serving the model."""
    try:
        r = httpx.get(f"{base_url}/models", timeout=5)
        models = [m["id"] for m in r.json().get("data", [])]
        return model_id in models
    except Exception:
        return False


def call_ollama(model_id: str, messages: list, tools: list | None = None,
                timeout: float = 600, base_url: str = "") -> dict:
    """Call Ollama chat API and return timing + response info."""
    url = base_url or get_ollama_base_url()
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    r = httpx.post(
        f"{url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    wall_time = time.perf_counter() - t0
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"ollama: {data['error']}")

    content = data.get("message", {}).get("content", "")
    tool_calls = data.get("message", {}).get("tool_calls", [])

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)

    gen_toks = eval_duration_ns / 1e9 if eval_duration_ns else wall_time
    ttft = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else None

    return {
        "content": content,
        "tool_calls": tool_calls,
        "output_tokens": eval_count,
        "input_tokens": prompt_eval_count,
        "tok_per_sec": eval_count / gen_toks if gen_toks > 0 and eval_count > 0 else 0,
        "ttft_sec": ttft,
        "wall_sec": wall_time,
        "raw_eval_duration_sec": eval_duration_ns / 1e9,
    }


def call_openai_compat(base_url: str, model_id: str, messages: list,
                       tools: list | None = None, timeout: float = 600) -> dict:
    """Call OpenAI-compatible API (vLLM / SGLang)."""
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "max_tokens": 2048,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    r = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    wall_time = time.perf_counter() - t0
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"api: {data['error']}")

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    usage = data.get("usage", {})

    output_tokens = usage.get("completion_tokens", 0)

    return {
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls", []),
        "output_tokens": output_tokens,
        "input_tokens": usage.get("prompt_tokens", 0),
        "tok_per_sec": output_tokens / wall_time if wall_time > 0 and output_tokens > 0 else 0,
        "ttft_sec": None,
        "wall_sec": wall_time,
        "raw_eval_duration_sec": wall_time,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_output_length(output_tokens: int, expected_min_tokens: int) -> float:
    if expected_min_tokens <= 0:
        return 1.0
    return _clamp01(output_tokens / expected_min_tokens)


def _extract_python_code(text: str) -> str:
    """Extract the most likely Python snippet from a model response."""
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return max((block.strip() for block in fenced), key=len, default="")

    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "from ", "import ")):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()

    return text.strip()


def _score_reasoning_response(text: str, output_tokens: int,
                              expected_min_tokens: int) -> tuple[float, str]:
    text_lower = text.lower()
    has_nine = bool(re.search(r"\b9\b", text))
    has_wrong_eight = bool(re.search(r"\b8\b", text))
    mentions_key_logic = sum(
        1 for marker in ["all but 9", "run away", "left", "remain", "still has"]
        if marker in text_lower
    )
    has_final_answer = any(
        marker in text_lower for marker in ["answer is 9", "there are 9", "has 9", "left with 9"]
    )

    score = 0.0
    if has_nine:
        score += 0.65
    if has_final_answer:
        score += 0.15
    score += 0.15 * _clamp01(mentions_key_logic / 2)
    score += 0.05 * _score_output_length(output_tokens, expected_min_tokens)

    if has_wrong_eight and not has_nine:
        score = min(score, 0.15)

    note = "correct answer 9" if has_nine else "answer unclear"
    if has_wrong_eight and not has_nine:
        note = "likely answered 8"
    return round(_clamp01(score), 2), note


def _score_code_response(text: str, output_tokens: int,
                         expected_min_tokens: int) -> tuple[float, str]:
    code = _extract_python_code(text)
    parsed_ok = False
    has_def = False
    has_type_hints = False
    has_docstring = False
    has_return = False
    target_like_name = False
    note_parts = []

    try:
        tree = ast.parse(code)
        parsed_ok = True
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        has_def = bool(funcs)
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(tree))
        if funcs:
            has_docstring = bool(ast.get_docstring(funcs[0]))
            for fn in funcs:
                if fn.returns is not None or any(arg.annotation is not None for arg in fn.args.args):
                    has_type_hints = True
                if "pal" in fn.name.lower():
                    target_like_name = True
    except SyntaxError:
        parsed_ok = False

    text_lower = text.lower()
    mentions_palindrome = "palindrome" in text_lower or "palindromic" in text_lower
    score = (
        0.35 * float(parsed_ok) +
        0.2 * float(has_def) +
        0.15 * float(has_type_hints) +
        0.1 * float(has_docstring) +
        0.1 * float(has_return) +
        0.05 * float(target_like_name or mentions_palindrome) +
        0.05 * _score_output_length(output_tokens, expected_min_tokens)
    )

    if parsed_ok:
        note_parts.append("ast ok")
    if has_def:
        note_parts.append("function")
    if has_type_hints:
        note_parts.append("type hints")
    if has_docstring:
        note_parts.append("docstring")
    if has_return:
        note_parts.append("return")
    if not note_parts:
        note_parts.append("syntax invalid")
    return round(_clamp01(score), 2), ", ".join(note_parts)


def _score_long_output_response(text: str, output_tokens: int,
                                expected_min_tokens: int) -> tuple[float, str]:
    words = len(re.findall(r"\b[\w-]+\b", text))
    length_score = _clamp01(words / 500) if words else _score_output_length(output_tokens, expected_min_tokens)
    topic_hits = sum(
        1 for term in ["edge", "ai", "latency", "device", "inference", "privacy"]
        if term in text.lower()
    )
    score = 0.75 * length_score + 0.25 * _clamp01(topic_hits / 3)
    return round(_clamp01(score), 2), f"{words} words, {topic_hits} topic hits"


def _score_chinese_response(text: str) -> tuple[float, str]:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    chinese_ratio = cjk_chars / max(cjk_chars + latin_chars, 1)
    has_moe = "moe" in text.lower() or "混合专家" in text
    has_dense = "dense" in text.lower() or "稠密" in text
    has_pros = "优势" in text or "优点" in text
    has_cons = "劣势" in text or "缺点" in text
    has_compare = "相比" in text or "相比于" in text or "对比" in text
    score = (
        0.25 * _clamp01(cjk_chars / 150) +
        0.2 * _clamp01(chinese_ratio / 0.6) +
        0.15 * float(has_moe) +
        0.15 * float(has_dense) +
        0.1 * float(has_pros) +
        0.1 * float(has_cons) +
        0.05 * float(has_compare)
    )
    note = f"{cjk_chars} CJK chars, zh ratio {chinese_ratio:.2f}"
    return round(_clamp01(score), 2), note


def _score_quality(prompt: dict, content: str, output_tokens: int,
                   tool_call_correct: bool) -> tuple[float, str]:
    """Heuristic quality scoring for benchmark prompts."""
    prompt_id = prompt["id"]
    expected_min_tokens = prompt.get("expected_min_tokens", 0)
    text = (content or "").strip()
    text_lower = text.lower()

    if not text and not tool_call_correct:
        return 0.0, "empty output"

    if prompt["type"] == "tool_call":
        return (1.0, "tool call emitted") if tool_call_correct else (0.0, "missing tool call")

    if prompt_id == "hello":
        greeting_hits = {
            name for name, pattern in {
                "english": r"\bhello\b|\bhi\b",
                "spanish": r"\bhola\b",
                "french": r"\bbonjour\b|\bsalut\b",
                "japanese": r"こんにちは|konnichiwa",
                "chinese": r"你好|您好",
                "german": r"\bhallo\b",
            }.items()
            if re.search(pattern, text_lower)
        }
        score = 0.7 * _clamp01(len(greeting_hits) / 3) + 0.3 * _score_output_length(output_tokens, expected_min_tokens)
        return round(score, 2), f"{len(greeting_hits)} greetings recognized"

    if prompt_id == "reasoning":
        return _score_reasoning_response(text, output_tokens, expected_min_tokens)

    if prompt_id == "code":
        return _score_code_response(text, output_tokens, expected_min_tokens)

    if prompt_id == "long_output":
        return _score_long_output_response(text, output_tokens, expected_min_tokens)

    if prompt_id == "chinese":
        return _score_chinese_response(text)

    score = _score_output_length(output_tokens, expected_min_tokens)
    return round(score, 2), f"{output_tokens}/{expected_min_tokens} tokens"


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    timestamp: str
    platform_name: str
    platform_gpu: str
    platform_memory_gb: int
    model_name: str
    engine: str
    arch: str
    active_params: str
    size_gb: float
    prompt_id: str
    prompt_type: str
    output_tokens: int
    input_tokens: int
    tok_per_sec: float
    ttft_sec: Optional[float]
    wall_sec: float
    tool_calls_count: int
    tool_call_correct: bool
    mem_used_before_gb: Optional[float] = None
    mem_peak_gb: Optional[float] = None
    mem_used_after_gb: Optional[float] = None
    swap_used_before_gb: Optional[float] = None
    swap_peak_gb: Optional[float] = None
    swap_used_after_gb: Optional[float] = None
    memory_pressure_note: str = ""
    quality_score: float = 0.0
    quality_notes: str = ""
    content_preview: str = ""
    community_toks: Optional[float] = None
    error: str = ""


@dataclass
class TTFTResult:
    timestamp: str
    platform_name: str
    platform_gpu: str
    platform_memory_gb: int
    model_name: str
    engine: str
    case_id: str
    prompt_size: str
    cache_state: str
    trial: int
    system_prompt_chars: int
    user_prompt_chars: int
    ttft_sec: Optional[float]
    wall_sec: float
    output_tokens: int
    mem_used_before_gb: Optional[float] = None
    mem_peak_gb: Optional[float] = None
    mem_used_after_gb: Optional[float] = None
    swap_used_before_gb: Optional[float] = None
    swap_peak_gb: Optional[float] = None
    swap_used_after_gb: Optional[float] = None
    memory_pressure_note: str = ""
    content_preview: str = ""
    error: str = ""


def _read_memory_snapshot() -> dict:
    """Read current system memory and swap usage in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "mem_used_gb": round(mem.used / 1e9, 2),
            "swap_used_gb": round(swap.used / 1e9, 2),
        }
    except Exception:
        return {
            "mem_used_gb": None,
            "swap_used_gb": None,
        }


def _summarize_memory_pressure(before: dict, peak: dict) -> str:
    notes = []
    mem_before = before.get("mem_used_gb")
    mem_peak = peak.get("mem_used_gb")
    swap_before = before.get("swap_used_gb")
    swap_peak = peak.get("swap_used_gb")
    if mem_before is not None and mem_peak is not None:
        mem_delta = mem_peak - mem_before
        if mem_delta >= 5:
            notes.append(f"peak +{mem_delta:.1f}GB")
    if swap_before is not None and swap_peak is not None:
        swap_delta = swap_peak - swap_before
        if swap_delta > 0.1:
            notes.append(f"swap +{swap_delta:.1f}GB")
    return ", ".join(notes) if notes else "stable"


class MemoryMonitor:
    """Background sampler for peak memory and swap usage."""

    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = interval_sec
        self.before = _read_memory_snapshot()
        self.peak = dict(self.before)
        self.after = dict(self.before)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self):
        while not self._stop.is_set():
            snap = _read_memory_snapshot()
            for key in ("mem_used_gb", "swap_used_gb"):
                cur = snap.get(key)
                peak = self.peak.get(key)
                if cur is not None and (peak is None or cur > peak):
                    self.peak[key] = cur
            self._stop.wait(self.interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.after = _read_memory_snapshot()
        return {
            "mem_used_before_gb": self.before.get("mem_used_gb"),
            "mem_peak_gb": self.peak.get("mem_used_gb"),
            "mem_used_after_gb": self.after.get("mem_used_gb"),
            "swap_used_before_gb": self.before.get("swap_used_gb"),
            "swap_peak_gb": self.peak.get("swap_used_gb"),
            "swap_used_after_gb": self.after.get("swap_used_gb"),
            "memory_pressure_note": _summarize_memory_pressure(self.before, self.peak),
        }


def _measure_ollama_ttft(model_id: str, messages: list, base_url: str = "",
                         timeout: float = 120, max_tokens: int = 128) -> dict:
    url = base_url or get_ollama_base_url()
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": max_tokens},
    }

    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    content = ""

    with httpx.stream("POST", f"{url}/api/chat", json=payload, timeout=timeout) as r:
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(f"ollama: {data['error']}")
            chunk = data.get("message", {}).get("content", "")
            if first_token_time is None and chunk:
                first_token_time = time.perf_counter()
            content += chunk
            if data.get("done"):
                total_tokens = data.get("eval_count", 0)
                break

    wall_time = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else wall_time
    return {
        "ttft_sec": round(ttft, 4) if ttft is not None else None,
        "wall_sec": round(wall_time, 2),
        "output_tokens": total_tokens,
        "content_preview": content[:120].replace("\n", " "),
    }


def _measure_openai_ttft(base_url: str, model_id: str, messages: list,
                         timeout: float = 120, max_tokens: int = 128) -> dict:
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    if "v1" in base_url:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    content = ""

    with httpx.stream("POST", f"{base_url}/chat/completions", json=payload, timeout=timeout) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            data = json.loads(data_str)
            if "error" in data:
                raise RuntimeError(f"api: {data['error']}")
            choice = data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            chunk = delta.get("content", "")
            if first_token_time is None and chunk:
                first_token_time = time.perf_counter()
            content += chunk
            usage = data.get("usage", {})
            if usage:
                total_tokens = usage.get("completion_tokens", total_tokens)

    wall_time = time.perf_counter() - t0
    if total_tokens == 0 and content:
        total_tokens = len(content.split())
    ttft = (first_token_time - t0) if first_token_time else wall_time
    return {
        "ttft_sec": round(ttft, 4) if ttft is not None else None,
        "wall_sec": round(wall_time, 2),
        "output_tokens": total_tokens,
        "content_preview": content[:120].replace("\n", " "),
    }


def _build_ttft_messages(case: dict, trial: int) -> list[dict]:
    system_prompt = case["system_prompt"]
    if case["cache_state"] == "cold":
        system_prompt = f"{system_prompt}\n\nCache-bust nonce: {trial}-{time.time_ns()}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case["user_message"]},
    ]


def run_ttft_single(model: ModelConfig, case: dict, plat: Platform,
                    trial: int, max_tokens: int = 128) -> TTFTResult:
    ts = datetime.now().isoformat()
    plat_fields = dict(
        platform_name=plat.name,
        platform_gpu=plat.gpu,
        platform_memory_gb=plat.memory_gb,
    )
    try:
        monitor = MemoryMonitor().start()
        messages = _build_ttft_messages(case, trial)
        if case["warm"]:
            prime_messages = [m.copy() for m in messages]
            if model.engine == "ollama":
                _measure_ollama_ttft(
                    model.model_id, prime_messages, base_url=model.base_url,
                    max_tokens=max_tokens,
                )
            else:
                _measure_openai_ttft(
                    model.base_url, model.model_id, prime_messages, max_tokens=max_tokens,
                )

        if model.engine == "ollama":
            result = _measure_ollama_ttft(
                model.model_id, messages, base_url=model.base_url, max_tokens=max_tokens,
            )
        else:
            result = _measure_openai_ttft(
                model.base_url, model.model_id, messages, max_tokens=max_tokens,
            )
        mem_stats = monitor.stop()

        return TTFTResult(
            timestamp=ts,
            **plat_fields,
            model_name=model.name,
            engine=model.engine,
            case_id=case["id"],
            prompt_size=case["prompt_size"],
            cache_state=case["cache_state"],
            trial=trial,
            system_prompt_chars=len(messages[0]["content"]),
            user_prompt_chars=len(messages[1]["content"]),
            ttft_sec=result["ttft_sec"],
            wall_sec=result["wall_sec"],
            output_tokens=result["output_tokens"],
            **mem_stats,
            content_preview=result["content_preview"],
        )
    except Exception as e:
        mem_stats = monitor.stop() if "monitor" in locals() else {}
        return TTFTResult(
            timestamp=ts,
            **plat_fields,
            model_name=model.name,
            engine=model.engine,
            case_id=case["id"],
            prompt_size=case["prompt_size"],
            cache_state=case["cache_state"],
            trial=trial,
            system_prompt_chars=len(case["system_prompt"]),
            user_prompt_chars=len(case["user_message"]),
            ttft_sec=None,
            wall_sec=0,
            output_tokens=0,
            **mem_stats,
            content_preview="",
            error=str(e)[:200],
        )


def run_single(model: ModelConfig, prompt: dict, plat: Platform) -> BenchResult:
    """Run a single model+prompt benchmark."""
    ts = datetime.now().isoformat()
    plat_fields = dict(
        platform_name=plat.name, platform_gpu=plat.gpu,
        platform_memory_gb=plat.memory_gb,
    )
    try:
        monitor = MemoryMonitor().start()
        if model.engine == "ollama":
            result = call_ollama(
                model.model_id,
                prompt["messages"],
                tools=prompt.get("tools"),
                base_url=model.base_url,
            )
        else:
            result = call_openai_compat(
                model.base_url,
                model.model_id,
                prompt["messages"],
                tools=prompt.get("tools"),
            )
        mem_stats = monitor.stop()

        tc = result.get("tool_calls") or []
        tc_correct = False
        if prompt["type"] == "tool_call" and tc:
            tc_correct = True
        quality_score, quality_notes = _score_quality(
            prompt,
            result.get("content", ""),
            result["output_tokens"],
            tc_correct,
        )

        return BenchResult(
            timestamp=ts,
            **plat_fields,
            model_name=model.name,
            engine=model.engine,
            arch=model.arch,
            active_params=model.active_params or "all",
            size_gb=model.size_gb,
            prompt_id=prompt["id"],
            prompt_type=prompt["type"],
            output_tokens=result["output_tokens"],
            input_tokens=result["input_tokens"],
            tok_per_sec=round(result["tok_per_sec"], 2),
            ttft_sec=round(result["ttft_sec"], 3) if result["ttft_sec"] else None,
            wall_sec=round(result["wall_sec"], 2),
            tool_calls_count=len(tc),
            tool_call_correct=tc_correct,
            **mem_stats,
            quality_score=quality_score,
            quality_notes=quality_notes,
            content_preview=result["content"][:200].replace("\n", " "),
            community_toks=model.community_toks,
        )
    except Exception as e:
        mem_stats = monitor.stop() if "monitor" in locals() else {}
        return BenchResult(
            timestamp=ts,
            **plat_fields,
            model_name=model.name,
            engine=model.engine,
            arch=model.arch,
            active_params=model.active_params or "all",
            size_gb=model.size_gb,
            prompt_id=prompt["id"],
            prompt_type=prompt["type"],
            output_tokens=0,
            input_tokens=0,
            tok_per_sec=0,
            ttft_sec=None,
            wall_sec=0,
            tool_calls_count=0,
            tool_call_correct=False,
            **mem_stats,
            quality_score=0.0,
            quality_notes="request failed",
            content_preview="",
            community_toks=model.community_toks,
            error=str(e)[:200],
        )


def get_memory_usage() -> dict:
    """Get current memory usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {"total_gb": round(mem.total / 1e9, 1), "used_gb": round(mem.used / 1e9, 1)}
    except ImportError:
        return {}


def is_model_available(model: ModelConfig) -> bool:
    if model.engine == "ollama":
        return check_ollama_model(model.model_id, model.base_url)
    else:
        return check_openai_endpoint(model.base_url, model.model_id)


def _model_compare_group(model_name: str) -> str:
    model = MODEL_BY_NAME.get(model_name)
    if model:
        return model.compare_group or model.name
    return model_name


def save_matrix_report(results: list[BenchResult], plat: Platform, suite_name: str, tag: str = ""):
    """Generate an engine comparison matrix report from benchmark results."""
    plat_dir = RESULTS_DIR / plat.name
    plat_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"matrix_{ts}" + (f"_{tag}" if tag else "")
    report_path = plat_dir / f"{stem}.md"

    ok = [r for r in results if r.output_tokens > 0 and not r.error]
    grouped: dict[str, dict[str, list[BenchResult]]] = {}
    for r in ok:
        grouped.setdefault(_model_compare_group(r.model_name), {}).setdefault(r.model_name, []).append(r)

    lines = [
        f"# Engine Matrix Report — {plat.name}",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> Platform: **{plat.gpu}** | {plat.memory_gb}GB | {plat.bandwidth_gbps} GB/s | CUDA {plat.cuda_version}  ",
        f"> Suite: `{suite_name}`  ",
        f"> Compare groups: {len(grouped)}",
        "",
    ]

    if not grouped:
        lines.append("No successful benchmark results available for matrix generation.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nMatrix report saved to:\n  {report_path}")
        return

    for compare_group, model_map in sorted(grouped.items()):
        lines.append(f"## Group: `{compare_group}`")
        lines.append("")
        lines.append("| Engine | Model | Prompts | Avg quality | Avg tok/s | Avg TTFT | Avg mem peak | Avg swap peak |")
        lines.append("|--------|-------|-------:|------------:|----------:|---------:|-------------:|--------------:|")

        rows = []
        for model_name, model_results in model_map.items():
            sample = model_results[0]
            avg_quality = sum(r.quality_score for r in model_results) / len(model_results)
            avg_tok = sum(r.tok_per_sec for r in model_results) / len(model_results)
            ttfts = [r.ttft_sec for r in model_results if r.ttft_sec is not None]
            avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
            mem_peaks = [r.mem_peak_gb for r in model_results if r.mem_peak_gb is not None]
            swap_peaks = [r.swap_peak_gb for r in model_results if r.swap_peak_gb is not None]
            avg_mem = (sum(mem_peaks) / len(mem_peaks)) if mem_peaks else None
            avg_swap = (sum(swap_peaks) / len(swap_peaks)) if swap_peaks else None
            rows.append((sample.engine, model_name, len(model_results), avg_quality, avg_tok, avg_ttft, avg_mem, avg_swap))

        rows.sort(key=lambda row: (row[3], row[4]), reverse=True)
        for engine, model_name, prompt_count, avg_quality, avg_tok, avg_ttft, avg_mem, avg_swap in rows:
            ttft_str = f"{avg_ttft:.3f}s" if avg_ttft is not None else "-"
            mem_str = f"{avg_mem:.1f}GB" if avg_mem is not None else "-"
            swap_str = f"{avg_swap:.1f}GB" if avg_swap is not None else "-"
            lines.append(
                f"| {engine} | {model_name} | {prompt_count} | **{avg_quality:.2f}** | "
                f"{avg_tok:.1f} | {ttft_str} | {mem_str} | {swap_str} |"
            )
        lines.append("")

        prompt_ids = sorted({r.prompt_id for rows in model_map.values() for r in rows})
        lines.append("### Prompt Breakdown")
        lines.append("")
        lines.append("| Prompt | Engine | Model | tok/s | Quality | TTFT | Mem peak |")
        lines.append("|--------|--------|-------|------:|--------:|-----:|---------:|")
        for prompt_id in prompt_ids:
            prompt_rows = [
                r for rows in model_map.values() for r in rows
                if r.prompt_id == prompt_id
            ]
            prompt_rows.sort(key=lambda r: (r.quality_score, r.tok_per_sec), reverse=True)
            for r in prompt_rows:
                ttft_str = f"{r.ttft_sec:.3f}s" if r.ttft_sec is not None else "-"
                mem_str = f"{r.mem_peak_gb:.1f}GB" if r.mem_peak_gb is not None else "-"
                lines.append(
                    f"| {prompt_id} | {r.engine} | {r.model_name} | {r.tok_per_sec:.1f} | "
                    f"{r.quality_score:.2f} | {ttft_str} | {mem_str} |"
                )
        lines.append("")

        engines_present = sorted({items[0].engine for items in model_map.values() if items})
        if len(engines_present) < 2:
            lines.append(f"> Only one engine available in this group right now: `{', '.join(engines_present)}`.")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMatrix report saved to:\n  {report_path}")


def print_table(results: list[BenchResult]):
    """Print results as a formatted table."""
    if not results:
        return

    print("\n" + "=" * 136)
    print(f"{'Model':<30} {'Engine':<8} {'Prompt':<14} {'tok/s':>7} {'TTFT':>7} "
          f"{'Wall':>7} {'OutTok':>7} {'Qual':>5} {'MemPk':>6} {'SwapPk':>7} {'Cmty':>6} {'TC':>3}")
    print("-" * 136)
    for r in results:
        ttft = f"{r.ttft_sec:.2f}s" if r.ttft_sec else "  -"
        cmty = f"{r.community_toks:.0f}" if r.community_toks else " -"
        tc = "✓" if r.tool_call_correct else ("✗" if r.prompt_type == "tool_call" else "-")
        err = " ⚠" if r.error else ""
        quality = f"{r.quality_score:.2f}"
        mem_peak = f"{r.mem_peak_gb:.1f}" if r.mem_peak_gb is not None else "-"
        swap_peak = f"{r.swap_peak_gb:.1f}" if r.swap_peak_gb is not None else "-"
        print(f"{r.model_name:<30} {r.engine:<8} {r.prompt_id:<14} "
              f"{r.tok_per_sec:>6.1f} {ttft:>7} {r.wall_sec:>6.1f}s "
              f"{r.output_tokens:>6} {quality:>5} {mem_peak:>6} {swap_peak:>7} {cmty:>6} {tc:>3}{err}")
    print("=" * 136)


def print_ttft_table(results: list[TTFTResult]):
    """Print TTFT results as a formatted table."""
    if not results:
        return

    print("\n" + "=" * 128)
    print(f"{'Model':<30} {'Engine':<8} {'Case':<12} {'Trial':>5} {'TTFT':>8} {'Wall':>7} {'OutTok':>7} {'MemPk':>6} {'SwapPk':>7} {'Err':>4}")
    print("-" * 128)
    for r in results:
        ttft = f"{r.ttft_sec:.3f}s" if r.ttft_sec is not None else "-"
        err = "⚠" if r.error else ""
        mem_peak = f"{r.mem_peak_gb:.1f}" if r.mem_peak_gb is not None else "-"
        swap_peak = f"{r.swap_peak_gb:.1f}" if r.swap_peak_gb is not None else "-"
        print(f"{r.model_name:<30} {r.engine:<8} {r.case_id:<12} {r.trial:>5} "
              f"{ttft:>8} {r.wall_sec:>6.1f}s {r.output_tokens:>7} {mem_peak:>6} {swap_peak:>7} {err:>4}")
    print("=" * 128)


def save_results(results: list[BenchResult], plat: Platform, tag: str = ""):
    """Save results to CSV, JSON, and Markdown report under results/{platform}/."""
    plat_dir = RESULTS_DIR / plat.name
    plat_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"bench_{ts}" + (f"_{tag}" if tag else "")

    csv_path = plat_dir / f"{stem}.csv"
    json_path = plat_dir / f"{stem}.json"
    report_path = plat_dir / f"{stem}.md"

    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))

        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

        _generate_report(results, plat, report_path, stem)

        print(f"\nResults saved to:\n  {csv_path}\n  {json_path}\n  {report_path}")


def save_ttft_results(results: list[TTFTResult], plat: Platform, tag: str = ""):
    """Save TTFT results to CSV, JSON, and Markdown report under results/{platform}/."""
    plat_dir = RESULTS_DIR / plat.name
    plat_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ttft_{ts}" + (f"_{tag}" if tag else "")

    csv_path = plat_dir / f"{stem}.csv"
    json_path = plat_dir / f"{stem}.json"
    report_path = plat_dir / f"{stem}.md"

    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))

        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

        _generate_ttft_report(results, plat, report_path, stem)

        print(f"\nResults saved to:\n  {csv_path}\n  {json_path}\n  {report_path}")


def _aggregate_numeric(values: list[float]) -> tuple[float, float, float, float]:
    ordered = sorted(values)
    avg = sum(ordered) / len(ordered)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        p50 = ordered[mid]
    else:
        p50 = (ordered[mid - 1] + ordered[mid]) / 2
    return avg, p50, ordered[0], ordered[-1]


def _generate_report(results: list[BenchResult], plat: Platform,
                     path: Path, stem: str):
    """Generate a Markdown benchmark report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ok = [r for r in results if r.output_tokens > 0 and not r.error]
    failed = [r for r in results if r.output_tokens == 0 or r.error]

    prompt_ids = list(dict.fromkeys(r.prompt_id for r in results))
    suites_used = ", ".join(prompt_ids)

    lines = [
        f"# Benchmark Report — {plat.name}",
        f"",
        f"> Generated: {now}  ",
        f"> Platform: **{plat.gpu}** | {plat.memory_gb}GB | {plat.bandwidth_gbps} GB/s | CUDA {plat.cuda_version}  ",
        f"> Test suite: {suites_used} ({len(prompt_ids)} prompt{'s' if len(prompt_ids)>1 else ''})  ",
        f"> Models tested: {len(set(r.model_name for r in results))} | "
        f"Succeeded: {len(ok)} | Failed: {len(failed)}",
        f"",
    ]

    if ok:
        by_model: dict[str, list[BenchResult]] = {}
        for r in ok:
            by_model.setdefault(r.model_name, []).append(r)

        model_rows = []
        max_speed = max(r.tok_per_sec for r in ok) if ok else 1.0
        for model_name, model_results in by_model.items():
            avg_quality = sum(r.quality_score for r in model_results) / len(model_results)
            avg_speed = sum(r.tok_per_sec for r in model_results) / len(model_results)
            avg_ttft_vals = [r.ttft_sec for r in model_results if r.ttft_sec is not None]
            avg_ttft = sum(avg_ttft_vals) / len(avg_ttft_vals) if avg_ttft_vals else None
            composite = 0.7 * avg_quality + 0.3 * (avg_speed / max_speed)
            sample = model_results[0]
            model_rows.append((model_name, sample, avg_quality, avg_speed, avg_ttft, composite))

        model_rows.sort(key=lambda row: (row[5], row[2], row[3]), reverse=True)

        lines.append("## Overall Ranking")
        lines.append("")
        lines.append("| # | Model | Engine | Arch | Avg quality | Avg tok/s | Avg TTFT | Composite |")
        lines.append("|---|-------|--------|------|------------:|----------:|---------:|----------:|")
        for i, (model_name, sample, avg_quality, avg_speed, avg_ttft, composite) in enumerate(model_rows, 1):
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}"))
            ttft = f"{avg_ttft:.2f}s" if avg_ttft is not None else "-"
            lines.append(
                f"| {medal} | {model_name} | {sample.engine} | {sample.arch} "
                f"| **{avg_quality:.2f}** | {avg_speed:.1f} | {ttft} | {composite:.2f} |"
            )
        lines.append("")

    for pid in prompt_ids:
        pid_results = [r for r in ok if r.prompt_id == pid]
        if not pid_results:
            continue

        ranked = sorted(pid_results, key=lambda r: r.tok_per_sec, reverse=True)

        lines.append(f"## Prompt: `{pid}`")
        lines.append("")
        lines.append("| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | Mem peak | Swap peak | tok/s per GB |")
        lines.append("|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|---------:|----------:|-------------:|")

        for i, r in enumerate(ranked, 1):
            ttft = f"{r.ttft_sec:.2f}s" if r.ttft_sec else "-"
            tpg = r.tok_per_sec / r.size_gb if r.size_gb > 0 else 0
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}"))
            mem_peak = f"{r.mem_peak_gb:.1f}GB" if r.mem_peak_gb is not None else "-"
            swap_peak = f"{r.swap_peak_gb:.1f}GB" if r.swap_peak_gb is not None else "-"
            lines.append(
                f"| {medal} | {r.model_name} | {r.engine} | {r.arch} "
                f"| {r.size_gb:.0f}GB | **{r.tok_per_sec:.1f}** | {ttft} "
                f"| {r.wall_sec:.1f}s | {r.output_tokens} | {r.quality_score:.2f} "
                f"| {mem_peak} | {swap_peak} | {tpg:.1f} |"
            )
        lines.append("")

    # Key findings
    if ok:
        lines.append("## Key Findings")
        lines.append("")

        fastest = max(ok, key=lambda r: r.tok_per_sec)
        lines.append(f"- **Fastest**: {fastest.model_name} @ **{fastest.tok_per_sec:.1f} tok/s**")

        ttft_results = [r for r in ok if r.ttft_sec is not None and r.ttft_sec > 0]
        if ttft_results:
            best_ttft = min(ttft_results, key=lambda r: r.ttft_sec)
            lines.append(f"- **Best TTFT**: {best_ttft.model_name} @ **{best_ttft.ttft_sec:.3f}s**")

        best_eff = max(ok, key=lambda r: r.tok_per_sec / r.size_gb if r.size_gb > 0 else 0)
        eff_val = best_eff.tok_per_sec / best_eff.size_gb if best_eff.size_gb > 0 else 0
        lines.append(f"- **Best efficiency** (tok/s per GB): {best_eff.model_name} "
                      f"@ **{eff_val:.1f} tok/s/GB**")

        best_quality = max(ok, key=lambda r: r.quality_score)
        lines.append(f"- **Best single-prompt quality**: {best_quality.model_name} "
                      f"on `{best_quality.prompt_id}` @ **{best_quality.quality_score:.2f}** "
                      f"({best_quality.quality_notes})")

        mem_tracked = [r for r in ok if r.mem_peak_gb is not None]
        if mem_tracked:
            highest_mem = max(mem_tracked, key=lambda r: r.mem_peak_gb or 0)
            lines.append(f"- **Highest memory peak**: {highest_mem.model_name} "
                          f"on `{highest_mem.prompt_id}` @ **{highest_mem.mem_peak_gb:.1f}GB** "
                          f"({highest_mem.memory_pressure_note})")

        moe = [r for r in ok if r.arch == "moe"]
        dense = [r for r in ok if r.arch == "dense"]
        if moe and dense:
            avg_moe = sum(r.tok_per_sec for r in moe) / len(moe)
            avg_dense = sum(r.tok_per_sec for r in dense) / len(dense)
            lines.append(f"- **MoE avg**: {avg_moe:.1f} tok/s vs **Dense avg**: {avg_dense:.1f} tok/s "
                          f"({'MoE' if avg_moe > avg_dense else 'Dense'} wins by "
                          f"{abs(avg_moe - avg_dense) / min(avg_moe, avg_dense) * 100:.0f}%)")

        lines.append("")

    # MoE vs Dense breakdown
    moe_ok = [r for r in ok if r.arch == "moe"]
    dense_ok = [r for r in ok if r.arch == "dense"]
    if moe_ok and dense_ok:
        lines.append("## Architecture Comparison")
        lines.append("")
        lines.append("| Metric | MoE | Dense |")
        lines.append("|--------|----:|------:|")
        avg_moe_tok = sum(r.tok_per_sec for r in moe_ok) / len(moe_ok)
        avg_dense_tok = sum(r.tok_per_sec for r in dense_ok) / len(dense_ok)
        lines.append(f"| Avg tok/s | {avg_moe_tok:.1f} | {avg_dense_tok:.1f} |")
        moe_ttft = [r for r in moe_ok if r.ttft_sec]
        dense_ttft = [r for r in dense_ok if r.ttft_sec]
        if moe_ttft and dense_ttft:
            avg_moe_t = sum(r.ttft_sec for r in moe_ttft) / len(moe_ttft)
            avg_dense_t = sum(r.ttft_sec for r in dense_ttft) / len(dense_ttft)
            lines.append(f"| Avg TTFT | {avg_moe_t:.3f}s | {avg_dense_t:.3f}s |")
        avg_moe_eff = sum(r.tok_per_sec / r.size_gb for r in moe_ok if r.size_gb > 0) / len(moe_ok)
        avg_dense_eff = sum(r.tok_per_sec / r.size_gb for r in dense_ok if r.size_gb > 0) / len(dense_ok)
        lines.append(f"| Avg tok/s/GB | {avg_moe_eff:.1f} | {avg_dense_eff:.1f} |")
        avg_moe_quality = sum(r.quality_score for r in moe_ok) / len(moe_ok)
        avg_dense_quality = sum(r.quality_score for r in dense_ok) / len(dense_ok)
        lines.append(f"| Avg quality | {avg_moe_quality:.2f} | {avg_dense_quality:.2f} |")
        moe_mem = [r.mem_peak_gb for r in moe_ok if r.mem_peak_gb is not None]
        dense_mem = [r.mem_peak_gb for r in dense_ok if r.mem_peak_gb is not None]
        if moe_mem and dense_mem:
            lines.append(f"| Avg mem peak | {sum(moe_mem) / len(moe_mem):.1f}GB | {sum(dense_mem) / len(dense_mem):.1f}GB |")
        lines.append(f"| Count | {len(moe_ok)} | {len(dense_ok)} |")
        lines.append("")

    # Failed models
    if failed:
        lines.append("## Failed Models")
        lines.append("")
        for r in failed:
            reason = r.error if r.error else "0 tokens returned"
            lines.append(f"- **{r.model_name}** ({r.engine}): {reason}")
        lines.append("")

    # Raw data reference
    lines.append("## Raw Data")
    lines.append("")
    lines.append(f"- CSV: `{stem}.csv`")
    lines.append(f"- JSON: `{stem}.json`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _generate_ttft_report(results: list[TTFTResult], plat: Platform,
                          path: Path, stem: str):
    """Generate a Markdown TTFT report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ok = [r for r in results if r.ttft_sec is not None and not r.error]
    failed = [r for r in results if r.ttft_sec is None or r.error]

    lines = [
        f"# TTFT Report — {plat.name}",
        "",
        f"> Generated: {now}  ",
        f"> Platform: **{plat.gpu}** | {plat.memory_gb}GB | {plat.bandwidth_gbps} GB/s | CUDA {plat.cuda_version}  ",
        f"> Models tested: {len(set(r.model_name for r in results))} | Trials: {len(results)}  ",
        f"> Cases: short/long x cold/warm",
        "",
    ]

    grouped: dict[tuple[str, str], list[TTFTResult]] = {}
    for r in ok:
        grouped.setdefault((r.model_name, r.case_id), []).append(r)

    if grouped:
        lines.append("## Case Summary")
        lines.append("")
        lines.append("| Model | Engine | Case | Prompt | Cache | Avg TTFT | P50 | Min | Max | Avg Wall | Avg Mem Peak | Avg Swap Peak |")
        lines.append("|-------|--------|------|--------|-------|---------:|----:|----:|----:|---------:|-------------:|--------------:|")
        for (model_name, case_id), items in sorted(grouped.items()):
            ttfts = [r.ttft_sec for r in items if r.ttft_sec is not None]
            walls = [r.wall_sec for r in items]
            avg_ttft, p50_ttft, min_ttft, max_ttft = _aggregate_numeric(ttfts)
            avg_wall = sum(walls) / len(walls)
            mem_peaks = [r.mem_peak_gb for r in items if r.mem_peak_gb is not None]
            swap_peaks = [r.swap_peak_gb for r in items if r.swap_peak_gb is not None]
            avg_mem_peak = (sum(mem_peaks) / len(mem_peaks)) if mem_peaks else None
            avg_swap_peak = (sum(swap_peaks) / len(swap_peaks)) if swap_peaks else None
            sample = items[0]
            lines.append(
                f"| {model_name} | {sample.engine} | {case_id} | {sample.prompt_size} | {sample.cache_state} "
                f"| **{avg_ttft:.3f}s** | {p50_ttft:.3f}s | {min_ttft:.3f}s | {max_ttft:.3f}s | {avg_wall:.2f}s "
                f"| {f'{avg_mem_peak:.1f}GB' if avg_mem_peak is not None else '-'} "
                f"| {f'{avg_swap_peak:.1f}GB' if avg_swap_peak is not None else '-'} |"
            )
        lines.append("")

        by_model: dict[str, dict[str, list[TTFTResult]]] = {}
        for (model_name, case_id), items in grouped.items():
            by_model.setdefault(model_name, {})[case_id] = items

        lines.append("## Cache Impact")
        lines.append("")
        for model_name, case_map in sorted(by_model.items()):
            short_cold = case_map.get("short_cold", [])
            short_warm = case_map.get("short_warm", [])
            long_cold = case_map.get("long_cold", [])
            long_warm = case_map.get("long_warm", [])
            lines.append(f"### {model_name}")
            lines.append("")
            if short_cold and short_warm:
                cold_avg = _aggregate_numeric([r.ttft_sec for r in short_cold if r.ttft_sec is not None])[0]
                warm_avg = _aggregate_numeric([r.ttft_sec for r in short_warm if r.ttft_sec is not None])[0]
                gain = cold_avg / warm_avg if warm_avg > 0 else 0
                lines.append(f"- Short prompt: cold **{cold_avg:.3f}s** vs warm **{warm_avg:.3f}s** ({gain:.2f}x)")
            if long_cold and long_warm:
                cold_avg = _aggregate_numeric([r.ttft_sec for r in long_cold if r.ttft_sec is not None])[0]
                warm_avg = _aggregate_numeric([r.ttft_sec for r in long_warm if r.ttft_sec is not None])[0]
                gain = cold_avg / warm_avg if warm_avg > 0 else 0
                lines.append(f"- Long prompt: cold **{cold_avg:.3f}s** vs warm **{warm_avg:.3f}s** ({gain:.2f}x)")
            lines.append("")

    lines.append("## Raw Trials")
    lines.append("")
    lines.append("| Model | Engine | Case | Trial | TTFT | Wall | Tokens | Mem Peak | Swap Peak |")
    lines.append("|-------|--------|------|------:|-----:|-----:|-------:|---------:|----------:|")
    for r in results:
        ttft = f"{r.ttft_sec:.3f}s" if r.ttft_sec is not None else "-"
        mem_peak = f"{r.mem_peak_gb:.1f}GB" if r.mem_peak_gb is not None else "-"
        swap_peak = f"{r.swap_peak_gb:.1f}GB" if r.swap_peak_gb is not None else "-"
        lines.append(
            f"| {r.model_name} | {r.engine} | {r.case_id} | {r.trial} | {ttft} | {r.wall_sec:.2f}s | {r.output_tokens} | {mem_peak} | {swap_peak} |"
        )
    lines.append("")

    if failed:
        lines.append("## Failed Trials")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.model_name}** `{r.case_id}` trial {r.trial}: {r.error or 'no TTFT measured'}")
        lines.append("")

    lines.append("## Raw Data")
    lines.append("")
    lines.append(f"- CSV: `{stem}.csv`")
    lines.append(f"- JSON: `{stem}.json`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_platform(args) -> Platform:
    """Resolve platform from CLI args."""
    name = getattr(args, "platform", "auto")
    if name == "auto":
        return Platform.detect()
    return Platform.from_name(name)


def main():
    known_names = list(KNOWN_PLATFORMS.keys())
    suite_choices = list(SUITES.keys()) + ["ttft"]
    parser = argparse.ArgumentParser(
        description="GPU Model Bench — Standardized LLM benchmark for any compute platform",
    )
    parser.add_argument("--platform", default="auto",
                        help=f"Platform (auto|{'|'.join(known_names)})")
    parser.add_argument("--models", nargs="+",
                        help="Model names to test (default: all compatible & available)")
    parser.add_argument("--engines", nargs="+", choices=["ollama", "vllm", "sglang"],
                        help="Only test these engines")
    parser.add_argument("--suite", default="standard", choices=suite_choices,
                        help="Prompt suite (default: standard)")
    parser.add_argument("--list", action="store_true", help="List all models with compatibility info")
    parser.add_argument("--tag", default="", help="Tag for result files")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup query before benchmarking each model")
    parser.add_argument("--no-filter", action="store_true",
                        help="Don't filter models by platform memory (show/test all)")
    parser.add_argument("--matrix", action="store_true",
                        help="Generate engine comparison matrix report after benchmark")
    parser.add_argument("--matrix-group",
                        help="Only benchmark models in this compare group (e.g. qwen3.5-35b)")
    parser.add_argument("--ttft-runs", type=int, default=3,
                        help="Trials per TTFT case (default: 3)")
    parser.add_argument("--ttft-max-tokens", type=int, default=128,
                        help="Max tokens to generate for TTFT suite (default: 128)")
    args = parser.parse_args()

    plat = _resolve_platform(args)

    if args.list:
        max_gb = plat.max_model_gb()
        print(f"\nPlatform: {plat.summary()}")
        print(f"Max model size: {max_gb:.0f}GB (80% of {plat.memory_gb}GB)")
        print(f"\n{'':>2}{'Name':<33} {'Engine':<8} {'Size':>6} {'Arch':<6} {'Active':<8} "
              f"{'Cmty':>6} {'TC':>3} {'Fit':>4}")
        print("-" * 95)
        for m in MODELS:
            cmty = f"{m.community_toks:.0f}" if m.community_toks else "-"
            tc = m.tool_call_support[0].upper()
            fits = "✓" if m.size_gb <= max_gb else "✗"
            avail = "✓" if is_model_available(m) else " "
            print(f"{avail} {m.name:<33} {m.engine:<8} {m.size_gb:>5.1f}G {m.arch:<6} "
                  f"{m.active_params or 'all':<8} {cmty:>6} {tc:>3} {fits:>4}")
        print(f"\n  ✓ (col 1) = installed/running   Fit = fits in {plat.memory_gb}GB memory")
        return

    # Filter models by engine / name
    candidates = MODELS
    if args.engines:
        candidates = [m for m in candidates if m.engine in args.engines]
    if args.models:
        candidates = [m for m in candidates if m.name in args.models]
    if args.matrix_group:
        args.matrix = True
        candidates = [m for m in candidates if (m.compare_group or m.name) == args.matrix_group]

    # Filter by platform memory
    if not args.no_filter:
        max_gb = plat.max_model_gb()
        too_large = [m for m in candidates if m.size_gb > max_gb]
        candidates = [m for m in candidates if m.size_gb <= max_gb]
        if too_large:
            print(f"\n⚠ Skipping {len(too_large)} model(s) too large for {plat.name} "
                  f"({plat.memory_gb}GB):")
            for m in too_large:
                print(f"  - {m.name} ({m.size_gb:.0f}GB)")

    # Check availability
    available = []
    unavailable = []
    for m in candidates:
        if is_model_available(m):
            available.append(m)
        else:
            unavailable.append(m)

    if unavailable:
        print(f"\n⚠ Unavailable models (skipping):")
        for m in unavailable:
            print(f"  - {m.name} ({m.engine})")

    if not available:
        print("\nNo available models to benchmark. Use --list to see all models.")
        sys.exit(1)

    if args.suite == "ttft":
        print(f"\n{'='*60}")
        print("  GPU Model Bench")
        print(f"  Platform: {plat.summary()}")
        print(f"  Models:   {len(available)} available ({len(candidates)} compatible)")
        print(f"  Suite:    ttft ({len(TTFT_CASES)} cases x {args.ttft_runs} trials)")
        print(f"  Memory:   {get_memory_usage()}")
        print(f"{'='*60}")

        ttft_results = []
        for i, model in enumerate(available, 1):
            print(f"\n[{i}/{len(available)}] {model.name} ({model.engine}, {model.size_gb}GB {model.arch})...")
            for case in TTFT_CASES:
                print(f"  [{case['id']}]")
                case_results = []
                for trial in range(1, args.ttft_runs + 1):
                    result = run_ttft_single(model, case, plat, trial, max_tokens=args.ttft_max_tokens)
                    ttft_results.append(result)
                    case_results.append(result)
                    if result.error:
                        print(f"    trial {trial}: ERROR: {result.error[:80]}")
                    else:
                        ttft = result.ttft_sec if result.ttft_sec is not None else result.wall_sec
                        print(f"    trial {trial}: TTFT {ttft:.3f}s, wall {result.wall_sec:.2f}s")
                ok_case = [r for r in case_results if r.ttft_sec is not None and not r.error]
                if ok_case:
                    avg_ttft = sum(r.ttft_sec for r in ok_case if r.ttft_sec is not None) / len(ok_case)
                    print(f"    avg: {avg_ttft:.3f}s")

        print_ttft_table(ttft_results)
        save_ttft_results(ttft_results, plat, args.tag)
        return

    prompts = SUITES[args.suite]
    print(f"\n{'='*60}")
    print(f"  GPU Model Bench")
    print(f"  Platform: {plat.summary()}")
    print(f"  Models:   {len(available)} available ({len(candidates)} compatible)")
    print(f"  Suite:    {args.suite} ({len(prompts)} prompts)")
    print(f"  Memory:   {get_memory_usage()}")
    print(f"{'='*60}")

    results = []
    for i, model in enumerate(available, 1):
        print(f"\n[{i}/{len(available)}] {model.name} ({model.engine}, {model.size_gb}GB {model.arch})...")

        if args.warmup:
            print("  warming up...", end=" ", flush=True)
            try:
                run_single(model, {"id": "warmup", "type": "chat",
                                    "messages": [{"role": "user", "content": "hi"}]}, plat)
                print("done")
            except Exception:
                print("failed")

        for prompt in prompts:
            print(f"  [{prompt['id']}] ", end="", flush=True)
            result = run_single(model, prompt, plat)
            results.append(result)
            if result.error:
                print(f"ERROR: {result.error[:80]}")
            elif result.output_tokens == 0:
                print(f"⚠ 0 tokens ({result.wall_sec:.1f}s) — model may have failed silently")
            else:
                print(f"{result.tok_per_sec:.1f} tok/s, {result.wall_sec:.1f}s, "
                      f"{result.output_tokens} tokens")

    print_table(results)
    save_results(results, plat, args.tag)
    if args.matrix:
        save_matrix_report(results, plat, args.suite, args.tag)


if __name__ == "__main__":
    main()
