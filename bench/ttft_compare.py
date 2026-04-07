#!/usr/bin/env python3
"""
TTFT (Time To First Token) comparison: Ollama vs vLLM prefix caching.

Tests the hypothesis that vLLM's prefix caching reduces TTFT from 2-4s to 0.12s
for repeated system prompts, compared to Ollama which recomputes every time.

Usage:
    python3 bench/ttft_compare.py --engine ollama   # test Ollama only
    python3 bench/ttft_compare.py --engine vllm     # test vLLM only
    python3 bench/ttft_compare.py                   # test both
"""

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LONG_SYSTEM_PROMPT = """You are a highly capable AI assistant specialized in software engineering.
You have deep expertise in Python, TypeScript, Rust, and Go. You follow best practices
including clean code, proper error handling, comprehensive testing, and security awareness.

When writing code:
- Always include type hints and docstrings
- Follow the principle of least surprise
- Prefer composition over inheritance
- Use dependency injection where appropriate
- Write code that is testable and maintainable

When debugging:
- Start with reproducing the issue
- Check logs and error messages carefully
- Form hypotheses and test them systematically
- Consider edge cases and race conditions

You have access to the following tools:
1. read_file(path: str) -> str: Read a file from the filesystem
2. write_file(path: str, content: str) -> bool: Write content to a file
3. search_code(pattern: str, directory: str) -> list[str]: Search for code patterns
4. run_command(cmd: str) -> dict: Execute a shell command
5. get_git_status() -> dict: Get current git repository status
6. create_pull_request(title: str, body: str, branch: str) -> str: Create a PR
7. list_directory(path: str) -> list[str]: List directory contents
8. get_file_history(path: str) -> list[dict]: Get git history for a file

Always think step by step before acting. Consider the broader context of the codebase.
When making changes, consider backward compatibility and potential side effects.
Provide clear explanations of your reasoning and any tradeoffs involved.

Current project context: You are working on a DGX Spark model testing lab.
The project benchmarks various LLMs on NVIDIA GB10 hardware with 128GB unified memory.
Key metrics are tok/s, TTFT, and tool calling accuracy."""

USER_MESSAGES = [
    "Hello, who are you?",
    "Write a Python function to find the longest palindromic substring.",
    "Explain the difference between MoE and Dense model architectures.",
    "What is the current git status of this project?",
    "Hello, who are you?",  # repeat to test cache
]


def measure_ollama_ttft(model: str, system_prompt: str, user_msg: str,
                        base_url: str = "http://localhost:11434") -> dict:
    """Measure TTFT using Ollama streaming API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
    }

    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    content = ""

    with httpx.stream("POST", f"{base_url}/api/chat", json=payload, timeout=120) as r:
        for line in r.iter_lines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if first_token_time is None and data.get("message", {}).get("content"):
                first_token_time = time.perf_counter()

            if data.get("done"):
                total_tokens = data.get("eval_count", 0)
                break
            content += data.get("message", {}).get("content", "")

    wall_time = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else wall_time

    return {
        "ttft_sec": round(ttft, 4),
        "wall_sec": round(wall_time, 2),
        "output_tokens": total_tokens,
        "content_preview": content[:100].replace("\n", " "),
    }


def measure_vllm_ttft(model: str, system_prompt: str, user_msg: str,
                       base_url: str = "http://localhost:8000/v1") -> dict:
    """Measure TTFT using vLLM OpenAI-compatible streaming API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    content = ""

    with httpx.stream("POST", f"{base_url}/chat/completions", json=payload,
                       timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = data.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content", "")
            if first_token_time is None and chunk:
                first_token_time = time.perf_counter()
            content += chunk
            if data.get("choices", [{}])[0].get("finish_reason"):
                usage = data.get("usage", {})
                total_tokens = usage.get("completion_tokens", len(content.split()))

    wall_time = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else wall_time

    return {
        "ttft_sec": round(ttft, 4),
        "wall_sec": round(wall_time, 2),
        "output_tokens": total_tokens,
        "content_preview": content[:100].replace("\n", " "),
    }


def run_ttft_series(engine: str, measure_fn, model: str, base_url: str,
                    system_prompt: str, rounds: int = 5) -> list[dict]:
    """Run a series of TTFT measurements."""
    results = []

    print(f"\n{'='*70}")
    print(f"  TTFT Test: {engine} | {model}")
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  Rounds: {rounds}")
    print(f"{'='*70}")

    for i, msg in enumerate(USER_MESSAGES[:rounds]):
        label = f"round_{i+1}"
        is_repeat = (i == 4 and msg == USER_MESSAGES[0])
        tag = " (REPEAT - cache test)" if is_repeat else ""

        print(f"  [{label}] \"{msg[:40]}...\" {tag}", end="", flush=True)

        result = measure_fn(model, system_prompt, msg, base_url)
        result["engine"] = engine
        result["round"] = i + 1
        result["user_msg"] = msg[:50]
        result["is_repeat"] = is_repeat
        result["system_prompt_len"] = len(system_prompt)
        results.append(result)

        print(f" → TTFT={result['ttft_sec']:.3f}s, wall={result['wall_sec']:.1f}s")

    avg_ttft = sum(r["ttft_sec"] for r in results) / len(results)
    print(f"\n  Average TTFT: {avg_ttft:.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="TTFT comparison: Ollama vs vLLM")
    parser.add_argument("--engine", choices=["ollama", "vllm", "both"], default="both")
    parser.add_argument("--ollama-model", default="qwen3.5:35b")
    parser.add_argument("--ollama-url", default="http://localhost:11436")
    parser.add_argument("--vllm-model", default="qwen3.5-35b-fp8")
    parser.add_argument("--vllm-url", default="http://localhost:18080/v1")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--short-prompt", action="store_true",
                        help="Use a short system prompt (control group)")
    args = parser.parse_args()

    system_prompt = "You are a helpful assistant." if args.short_prompt else LONG_SYSTEM_PROMPT

    all_results = []

    if args.engine in ("ollama", "both"):
        try:
            results = run_ttft_series(
                "ollama", measure_ollama_ttft,
                args.ollama_model, args.ollama_url,
                system_prompt, args.rounds,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n  ❌ Ollama test failed: {e}")

    if args.engine in ("vllm", "both"):
        try:
            results = run_ttft_series(
                "vllm", measure_vllm_ttft,
                args.vllm_model, args.vllm_url,
                system_prompt, args.rounds,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n  ❌ vLLM test failed: {e}")

    if all_results:
        # Summary
        print(f"\n{'='*70}")
        print("  TTFT COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Engine':<10} {'Round':<10} {'TTFT':>8} {'Wall':>8} {'Repeat':>8}")
        print(f"  {'-'*50}")
        for r in all_results:
            repeat = "✓" if r["is_repeat"] else ""
            print(f"  {r['engine']:<10} {r['round']:<10} {r['ttft_sec']:>7.3f}s {r['wall_sec']:>7.1f}s {repeat:>8}")

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"ttft_{ts}.json"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {path}")


if __name__ == "__main__":
    main()
