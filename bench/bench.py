#!/usr/bin/env python3
"""
DGX Spark LLM Benchmark Suite

Systematic benchmark for local LLMs on DGX Spark (GB10, 128GB).
Tests models across Ollama / vLLM / SGLang with standardized prompts.

Usage:
    python bench/bench.py                    # run all available models
    python bench/bench.py --models qwen3.5:9b nemotron-3-nano
    python bench/bench.py --engines ollama   # only test Ollama models
    python bench/bench.py --suite quick      # short prompts only
    python bench/bench.py --list             # list all configured models
"""

import argparse
import csv
import json
import os
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
    # --- Ollama models (easiest) ---
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


def check_ollama_model(model_id: str) -> bool:
    """Check if model is available in Ollama."""
    try:
        r = httpx.get(f"{get_ollama_base_url()}/api/tags", timeout=5)
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
                timeout: float = 600) -> dict:
    """Call Ollama chat API and return timing + response info."""
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    r = httpx.post(
        f"{get_ollama_base_url()}/api/chat",
        json=payload,
        timeout=timeout,
    )
    wall_time = time.perf_counter() - t0
    data = r.json()

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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    timestamp: str
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
    content_preview: str
    community_toks: Optional[float]
    error: str = ""


def run_single(model: ModelConfig, prompt: dict) -> BenchResult:
    """Run a single model+prompt benchmark."""
    ts = datetime.now().isoformat()
    try:
        if model.engine == "ollama":
            result = call_ollama(
                model.model_id,
                prompt["messages"],
                tools=prompt.get("tools"),
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

        return BenchResult(
            timestamp=ts,
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
            content_preview=result["content"][:200].replace("\n", " "),
            community_toks=model.community_toks,
        )
    except Exception as e:
        return BenchResult(
            timestamp=ts,
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
        return check_ollama_model(model.model_id)
    else:
        return check_openai_endpoint(model.base_url, model.model_id)


def print_table(results: list[BenchResult]):
    """Print results as a formatted table."""
    if not results:
        return

    print("\n" + "=" * 110)
    print(f"{'Model':<30} {'Engine':<8} {'Prompt':<14} {'tok/s':>7} {'TTFT':>7} "
          f"{'Wall':>7} {'OutTok':>7} {'Cmty':>6} {'TC':>3}")
    print("-" * 110)
    for r in results:
        ttft = f"{r.ttft_sec:.2f}s" if r.ttft_sec else "  -"
        cmty = f"{r.community_toks:.0f}" if r.community_toks else " -"
        tc = "✓" if r.tool_call_correct else ("✗" if r.prompt_type == "tool_call" else "-")
        err = " ⚠" if r.error else ""
        print(f"{r.model_name:<30} {r.engine:<8} {r.prompt_id:<14} "
              f"{r.tok_per_sec:>6.1f} {ttft:>7} {r.wall_sec:>6.1f}s "
              f"{r.output_tokens:>6} {cmty:>6} {tc:>3}{err}")
    print("=" * 110)


def save_results(results: list[BenchResult], tag: str = ""):
    """Save results to CSV and JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"bench_{ts}" + (f"_{tag}" if tag else "")

    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path = RESULTS_DIR / f"{stem}.json"

    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))

        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to:\n  {csv_path}\n  {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DGX Spark LLM Benchmark")
    parser.add_argument("--models", nargs="+", help="Model names to test (default: all available)")
    parser.add_argument("--engines", nargs="+", choices=["ollama", "vllm", "sglang"],
                        help="Only test these engines")
    parser.add_argument("--suite", default="standard", choices=list(SUITES.keys()),
                        help="Prompt suite (default: standard)")
    parser.add_argument("--list", action="store_true", help="List all configured models")
    parser.add_argument("--tag", default="", help="Tag for result files")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup query before benchmarking each model")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Name':<35} {'Engine':<8} {'Size':>6} {'Arch':<6} {'Active':<8} "
              f"{'Cmty tok/s':>10} {'TC':>4}")
        print("-" * 90)
        for m in MODELS:
            cmty = f"{m.community_toks:.0f}" if m.community_toks else "-"
            tc = m.tool_call_support[0].upper()
            avail = "✓" if is_model_available(m) else "✗"
            print(f"{avail} {m.name:<33} {m.engine:<8} {m.size_gb:>5.1f}G {m.arch:<6} "
                  f"{m.active_params or 'all':<8} {cmty:>10} {tc:>4}")
        return

    # Filter models
    candidates = MODELS
    if args.engines:
        candidates = [m for m in candidates if m.engine in args.engines]
    if args.models:
        candidates = [m for m in candidates if m.name in args.models]

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
    print(f"\n🏁 DGX Spark Benchmark")
    print(f"   Models: {len(available)}")
    print(f"   Suite:  {args.suite} ({len(prompts)} prompts)")
    print(f"   Memory: {get_memory_usage()}")

    results = []
    for i, model in enumerate(available, 1):
        print(f"\n[{i}/{len(available)}] {model.name} ({model.engine}, {model.size_gb}GB {model.arch})...")

        if args.warmup:
            print("  warming up...", end=" ", flush=True)
            try:
                run_single(model, {"id": "warmup", "type": "chat",
                                    "messages": [{"role": "user", "content": "hi"}]})
                print("done")
            except Exception:
                print("failed")

        for prompt in prompts:
            print(f"  [{prompt['id']}] ", end="", flush=True)
            result = run_single(model, prompt)
            results.append(result)
            if result.error:
                print(f"ERROR: {result.error[:80]}")
            else:
                print(f"{result.tok_per_sec:.1f} tok/s, {result.wall_sec:.1f}s, "
                      f"{result.output_tokens} tokens")

    print_table(results)
    save_results(results, args.tag)


if __name__ == "__main__":
    main()
