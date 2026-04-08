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
import csv
import json
import os
import platform as platform_mod
import re
import subprocess
import sys
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


MODELS = [
    # --- Ollama: Qwen 3.5 ---
    ModelConfig("qwen3.5:9b", "ollama", "qwen3.5:9b", 6.6, "dense",
                notes="Lightweight, fast", community_toks=35),
    ModelConfig("qwen3.5:35b", "ollama", "qwen3.5:35b", 23, "moe", "3B",
                notes="MoE 35B/3B active", community_toks=57, tool_call_support="partial"),
    ModelConfig("qwen3.5:27b", "ollama", "qwen3.5:27b", 17, "dense",
                notes="Dense 27B, slow but capable", community_toks=13),
    ModelConfig("qwen3.5:122b-a10b", "ollama", "qwen3.5:122b-a10b", 81, "moe", "10B",
                notes="Largest Qwen MoE", community_toks=23, tool_call_support="partial"),
    ModelConfig("qwen3:32b", "ollama", "qwen3:32b", 20, "dense",
                notes="Dense 32B", community_toks=10),

    # --- Ollama: NVIDIA models ---
    ModelConfig("gpt-oss:120b", "ollama", "gpt-oss:120b", 65, "moe",
                notes="NVIDIA GPT-OSS 120B MoE", community_toks=41, tool_call_support="yes"),
    ModelConfig("gpt-oss:20b", "ollama", "gpt-oss:20b", 13, "moe",
                notes="NVIDIA GPT-OSS 20B", community_toks=58, tool_call_support="no"),
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
                community_toks=6, tool_call_support="yes"),

    # --- vLLM + namake-taro patch (best performance) ---
    ModelConfig("qwen3.5-35b-mxfp4-vllm", "vllm", "Qwen/Qwen3.5-35B-A3B", 35, "moe", "3B",
                base_url="http://localhost:18080/v1",
                notes="MXFP4 quantized, requires namake-taro patch",
                community_toks=60, tool_call_support="partial"),
    ModelConfig("gpt-oss-120b-mxfp4-vllm", "vllm", "openai/gpt-oss-120b", 65, "moe",
                base_url="http://localhost:18080/v1",
                notes="MXFP4 quantized, requires namake-taro patch",
                community_toks=81),

    # --- SGLang Docker (sglang:spark) ---
    ModelConfig("gpt-oss-20b-sglang", "sglang", "openai/gpt-oss-20b", 13, "moe",
                base_url="http://localhost:18080/v1",
                notes="SGLang optimized for Spark",
                community_toks=61, tool_call_support="no"),
]


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
        has_nine = bool(re.search(r"\b9\b", text))
        mentions_logic = any(x in text_lower for x in ["all but 9", "run away", "left", "remain"])
        wrong_shortcut = bool(re.search(r"\b8\b", text)) and not has_nine
        score = 0.0
        if has_nine:
            score += 0.75
        if mentions_logic:
            score += 0.2
        if output_tokens >= expected_min_tokens:
            score += 0.05
        if wrong_shortcut:
            score = min(score, 0.2)
        note = "correct answer 9" if has_nine else "answer unclear"
        return round(_clamp01(score), 2), note

    if prompt_id == "code":
        has_def = "def " in text
        has_type_hints = "->" in text
        has_docstring = '"""' in text or "'''" in text
        mentions_palindrome = "palindrome" in text_lower or "palindromic" in text_lower
        score = (
            0.3 * has_def +
            0.2 * has_type_hints +
            0.2 * has_docstring +
            0.1 * mentions_palindrome +
            0.2 * _score_output_length(output_tokens, expected_min_tokens)
        )
        parts = []
        if has_def:
            parts.append("function")
        if has_type_hints:
            parts.append("type hints")
        if has_docstring:
            parts.append("docstring")
        return round(_clamp01(score), 2), ", ".join(parts) if parts else "structure incomplete"

    if prompt_id == "long_output":
        length_score = _score_output_length(output_tokens, expected_min_tokens)
        mentions_topic = all(term in text_lower for term in ["edge", "ai"])
        score = 0.8 * length_score + 0.2 * float(mentions_topic)
        note = f"{output_tokens}/{expected_min_tokens} tokens"
        return round(_clamp01(score), 2), note

    if prompt_id == "chinese":
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        has_moe = "moe" in text_lower or "混合专家" in text
        has_dense = "dense" in text_lower or "稠密" in text
        has_pros = "优势" in text or "优点" in text
        has_cons = "劣势" in text or "缺点" in text
        score = (
            0.3 * _clamp01(cjk_chars / 80) +
            0.2 * has_moe +
            0.2 * has_dense +
            0.15 * has_pros +
            0.15 * has_cons
        )
        note = f"{cjk_chars} CJK chars"
        return round(_clamp01(score), 2), note

    score = _score_output_length(output_tokens, expected_min_tokens)
    return round(score, 2), f"{output_tokens}/{expected_min_tokens} tokens"


# ---------------------------------------------------------------------------
# Benchmark runner
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
    quality_score: float = 0.0
    quality_notes: str = ""
    content_preview: str = ""
    community_toks: Optional[float] = None
    error: str = ""


def run_single(model: ModelConfig, prompt: dict, plat: Platform) -> BenchResult:
    """Run a single model+prompt benchmark."""
    ts = datetime.now().isoformat()
    plat_fields = dict(
        platform_name=plat.name, platform_gpu=plat.gpu,
        platform_memory_gb=plat.memory_gb,
    )
    try:
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
            quality_score=quality_score,
            quality_notes=quality_notes,
            content_preview=result["content"][:200].replace("\n", " "),
            community_toks=model.community_toks,
        )
    except Exception as e:
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


def print_table(results: list[BenchResult]):
    """Print results as a formatted table."""
    if not results:
        return

    print("\n" + "=" * 118)
    print(f"{'Model':<30} {'Engine':<8} {'Prompt':<14} {'tok/s':>7} {'TTFT':>7} "
          f"{'Wall':>7} {'OutTok':>7} {'Qual':>5} {'Cmty':>6} {'TC':>3}")
    print("-" * 118)
    for r in results:
        ttft = f"{r.ttft_sec:.2f}s" if r.ttft_sec else "  -"
        cmty = f"{r.community_toks:.0f}" if r.community_toks else " -"
        tc = "✓" if r.tool_call_correct else ("✗" if r.prompt_type == "tool_call" else "-")
        err = " ⚠" if r.error else ""
        quality = f"{r.quality_score:.2f}"
        print(f"{r.model_name:<30} {r.engine:<8} {r.prompt_id:<14} "
              f"{r.tok_per_sec:>6.1f} {ttft:>7} {r.wall_sec:>6.1f}s "
              f"{r.output_tokens:>6} {quality:>5} {cmty:>6} {tc:>3}{err}")
    print("=" * 118)


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
        lines.append("| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |")
        lines.append("|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|")

        for i, r in enumerate(ranked, 1):
            ttft = f"{r.ttft_sec:.2f}s" if r.ttft_sec else "-"
            tpg = r.tok_per_sec / r.size_gb if r.size_gb > 0 else 0
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}"))
            lines.append(
                f"| {medal} | {r.model_name} | {r.engine} | {r.arch} "
                f"| {r.size_gb:.0f}GB | **{r.tok_per_sec:.1f}** | {ttft} "
                f"| {r.wall_sec:.1f}s | {r.output_tokens} | {r.quality_score:.2f} | {tpg:.1f} |"
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
    parser = argparse.ArgumentParser(
        description="GPU Model Bench — Standardized LLM benchmark for any compute platform",
    )
    parser.add_argument("--platform", default="auto",
                        help=f"Platform (auto|{'|'.join(known_names)})")
    parser.add_argument("--models", nargs="+",
                        help="Model names to test (default: all compatible & available)")
    parser.add_argument("--engines", nargs="+", choices=["ollama", "vllm", "sglang"],
                        help="Only test these engines")
    parser.add_argument("--suite", default="standard", choices=list(SUITES.keys()),
                        help="Prompt suite (default: standard)")
    parser.add_argument("--list", action="store_true", help="List all models with compatibility info")
    parser.add_argument("--tag", default="", help="Tag for result files")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup query before benchmarking each model")
    parser.add_argument("--no-filter", action="store_true",
                        help="Don't filter models by platform memory (show/test all)")
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


if __name__ == "__main__":
    main()
