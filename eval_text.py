#!/usr/bin/env python3
"""
GSM8K Math Reasoning Benchmark for text models (GGUF via llama-server)

Sends math word problems to llama-server's /v1/chat/completions,
extracts numeric answers, compares to ground truth. Supports thinking mode.

Usage:
    # Standard eval (200 samples, Q4_K_M)
    nohup python3 eval_text.py --samples 200 --tag Q4_K_M > eval_text.log 2>&1 &

    # With thinking enabled (for reasoning-distilled models)
    nohup python3 eval_text.py --samples 200 --tag Q4_K_M --thinking > eval_text.log 2>&1 &

    # Monitor progress
    tail -f eval_results_text/progress.log
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

RESULTS_DIR = Path("eval_results_text")

_progress_log = None


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _progress_log:
        with open(_progress_log, "a") as f:
            f.write(line + "\n")


def _strip_think(text):
    """Remove <think>...</think> wrapper if present, keep only the final answer part."""
    if not text:
        return ""
    s = text.strip()
    close = "</think>"
    if close in s:
        tail = s.split(close)[-1].strip()
        if tail:
            return tail
    return s


def query_model(base_url, question, enable_thinking=True, timeout=180, model=None):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math problem solver. "
                "Solve the problem step by step, then give your final numeric answer "
                "on the last line in the format: #### <number>"
            ),
        },
        {"role": "user", "content": question},
    ]

    if model:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": enable_thinking,
            "options": {"num_ctx": 4096, "temperature": 0.1},
        }
        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()
        thinking = (msg.get("thinking") or "").strip()
        if not content and thinking:
            content = _strip_think(thinking)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
    else:
        payload = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        content = (choice.get("content") or "").strip()
        thinking = (choice.get("reasoning_content") or "").strip()
        if not content and thinking:
            content = _strip_think(thinking)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

    return content, thinking, prompt_tokens, completion_tokens


def extract_gsm8k_gt(answer_str):
    """Extract numeric answer from GSM8K ground truth (after ####)."""
    m = re.search(r"####\s*(.+)", answer_str)
    if m:
        num_str = m.group(1).strip().replace(",", "")
        try:
            return float(num_str) if "." in num_str else int(num_str)
        except ValueError:
            return num_str
    return answer_str.strip()


def extract_answer(response):
    """Extract numeric answer from model response."""
    m = re.search(r"####\s*(.+?)(?:\s*$|\n)", response)
    if m:
        num_str = m.group(1).strip().replace(",", "").replace("$", "")
        try:
            return float(num_str) if "." in num_str else int(num_str)
        except ValueError:
            pass

    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", line.replace(",", ""))
        if nums:
            try:
                return float(nums[-1]) if "." in nums[-1] else int(nums[-1])
            except ValueError:
                continue

    return None


def answers_match(pred, gt):
    """Compare predicted and ground truth answers (numeric tolerance)."""
    if pred is None:
        return False
    try:
        p = float(pred)
        g = float(gt)
        if g == 0:
            return abs(p) < 1e-6
        return abs(p - g) / max(abs(g), 1e-9) < 0.01
    except (TypeError, ValueError):
        return str(pred).strip() == str(gt).strip()


def compute_stats(results_file, tag):
    total = correct = 0
    latencies = []
    tok_speeds = []
    if not results_file.exists():
        return {}
    with open(results_file) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if r.get("correct"):
                correct += 1
            if "elapsed_s" in r:
                latencies.append(r["elapsed_s"])
            ct = r.get("completion_tokens", 0)
            el = r.get("elapsed_s", 0)
            if ct > 0 and el > 0:
                tok_speeds.append(ct / el)

    latency_stats = {}
    if latencies:
        latencies.sort()
        latency_stats = {
            "avg_s": round(sum(latencies) / len(latencies), 2),
            "p50_s": round(latencies[len(latencies) // 2], 2),
            "p95_s": round(latencies[int(len(latencies) * 0.95)], 2),
            "max_s": round(max(latencies), 2),
        }

    speed_stats = {}
    if tok_speeds:
        tok_speeds.sort()
        speed_stats = {
            "avg_tok_s": round(sum(tok_speeds) / len(tok_speeds), 1),
            "p50_tok_s": round(tok_speeds[len(tok_speeds) // 2], 1),
        }

    return {
        "model": tag,
        "tag": tag,
        "benchmark": "GSM8K",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0,
        "latency": latency_stats,
        "speed": speed_stats,
        "updated_at": datetime.now().isoformat(),
    }


def main():
    global _progress_log

    parser = argparse.ArgumentParser(description="GSM8K reasoning benchmark")
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=9062)
    parser.add_argument("--samples", type=int, default=0, help="0 = all ~1319 test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tag", default="Q4_K_M")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (for reasoning-distilled models)")
    parser.add_argument("--model", default=None,
                        help="Ollama model name (e.g. qwen3.5:27b). If set, uses Ollama API instead of llama-server.")
    args = parser.parse_args()

    base_url = f"http://{args.llama_host}:{args.llama_port}"
    RESULTS_DIR.mkdir(exist_ok=True)

    results_file = RESULTS_DIR / f"gsm8k_results_{args.tag}.jsonl"
    summary_file = RESULTS_DIR / f"gsm8k_summary_{args.tag}.json"
    _progress_log = RESULTS_DIR / f"progress_{args.tag}.log"

    log(f"=== GSM8K eval for {args.tag} (thinking={'ON' if args.thinking else 'OFF'}) ===")
    if args.model:
        log(f"Using Ollama model: {args.model}")
        log("Checking Ollama health...")
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=10)
            r.raise_for_status()
            log("Ollama is healthy")
        except Exception as e:
            log(f"ERROR: Ollama not reachable at {base_url}: {e}")
            sys.exit(1)
    else:
        log("Checking llama-server health...")
        try:
            r = requests.get(f"{base_url}/health", timeout=10)
            r.raise_for_status()
            log("llama-server is healthy")
        except Exception as e:
            log(f"ERROR: llama-server not reachable at {base_url}: {e}")
            sys.exit(1)

    log("Loading GSM8K dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    log(f"Loaded {len(ds)} test samples")

    indices = list(range(len(ds)))
    if args.samples and args.samples < len(ds):
        random.seed(args.seed)
        indices = sorted(random.sample(indices, args.samples))
        log(f"Sampled {args.samples} questions (seed={args.seed})")

    done_ids = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r["index"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            log(f"Resuming: {len(done_ids)} already completed, skipping them")

    log(f"Target: {len(indices)} questions → {base_url}")
    log("=" * 60)

    done_count = len(done_ids)
    correct_count = sum(
        1 for line in (open(results_file).readlines() if results_file.exists() else [])
        if line.strip() and json.loads(line).get("correct")
    )

    for i, idx in enumerate(indices):
        if idx in done_ids:
            continue

        sample = ds[idx]
        question = sample["question"]
        gt_answer_raw = sample["answer"]
        gt_answer = extract_gsm8k_gt(gt_answer_raw)

        raw_response = None
        thinking_content = ""
        prompt_tok = comp_tok = 0
        elapsed = 0

        for attempt in range(3):
            t0 = time.time()
            try:
                raw_response, thinking_content, prompt_tok, comp_tok = query_model(
                    base_url, question, enable_thinking=args.thinking,
                    timeout=args.timeout, model=args.model
                )
                elapsed = time.time() - t0
                break
            except Exception as e:
                elapsed = time.time() - t0
                log(f"[{done_count+1}/{len(indices)}] idx={idx} attempt {attempt+1} ERROR: {e}")
                time.sleep(10 * (attempt + 1))

        if raw_response is None:
            log(f"[{done_count+1}/{len(indices)}] idx={idx} SKIPPED after 3 retries")
            continue

        pred = extract_answer(raw_response)
        is_correct = answers_match(pred, gt_answer)
        done_count += 1
        if is_correct:
            correct_count += 1
        acc = correct_count / done_count * 100

        result = {
            "index": idx,
            "question": question[:200],
            "gt_answer": str(gt_answer),
            "pred_answer": str(pred),
            "correct": is_correct,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "thinking_len": len(thinking_content),
            "raw_response": raw_response[:500],
            "timestamp": datetime.now().isoformat(),
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        mark = "✓" if is_correct else "✗"
        log(
            f"[{done_count}/{len(indices)}] idx={idx} "
            f"{mark} pred={pred} gt={gt_answer} "
            f"acc={acc:.1f}% ({elapsed:.1f}s, {comp_tok}tok)"
        )

        if done_count % 10 == 0:
            summary = compute_stats(results_file, args.tag)
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    summary = compute_stats(results_file, args.tag)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("=" * 60)
    if not summary or summary["total"] == 0:
        log("No results collected.")
        return
    log(f"DONE! [{args.tag}] GSM8K Accuracy: {summary['accuracy']}% ({summary['correct']}/{summary['total']})")
    log(f"Reference: Qwen3.5-27B base ≈ 90%+ on GSM8K")
    if summary.get("latency"):
        lat = summary["latency"]
        log(f"Latency: avg={lat['avg_s']}s p50={lat['p50_s']}s p95={lat['p95_s']}s max={lat['max_s']}s")
    if summary.get("speed"):
        spd = summary["speed"]
        log(f"Speed: avg={spd['avg_tok_s']} tok/s p50={spd['p50_tok_s']} tok/s")
    log(f"\nResults: {results_file}")
    log(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
