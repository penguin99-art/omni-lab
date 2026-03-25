#!/usr/bin/env python3
"""
MMStar Vision Benchmark for MiniCPM-o 4.5 (GGUF Q4_K_M)

Sends images + MCQ questions to llama-server's /v1/chat/completions,
compares model answers to ground truth, logs results incrementally.

Usage:
    # Quick run (300 samples, ~1-2h on GB10)
    nohup python3 eval_vision.py --samples 300 > eval_vision.log 2>&1 &

    # Full run (all 1500)
    nohup python3 eval_vision.py > eval_vision.log 2>&1 &

    # Monitor progress
    tail -f eval_results/progress.log
"""

import argparse
import base64
import io
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

RESULTS_DIR = Path("eval_results")
RESULTS_FILE = RESULTS_DIR / "mmstar_results.jsonl"
SUMMARY_FILE = RESULTS_DIR / "mmstar_summary.json"
PROGRESS_LOG = RESULTS_DIR / "progress.log"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def query_model(base_url, image_b64, question, timeout=120):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]["message"]
    content = choice.get("content", "") or ""
    return content.strip()


def extract_answer(response):
    """Extract single letter answer (A/B/C/D) from model response."""
    response = response.strip()
    if not response:
        return "?"

    # Best case: response starts with the letter
    if response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
        return response[0]

    # Look for patterns like "The answer is B" or "Answer: A"
    m = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-D])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Look for standalone letter at the end: "... so B"
    m = re.search(r"\b([A-D])\s*\.?\s*$", response)
    if m:
        return m.group(1)

    # Single standalone letter anywhere (word boundary on both sides)
    matches = re.findall(r"\b([A-D])\b", response)
    if len(matches) == 1:
        return matches[0]

    # If multiple letters found, prefer the last one (usually the conclusion)
    if matches:
        return matches[-1]

    return "?"


def build_prompt(question):
    """Wrap question in a concise MCQ prompt."""
    return (
        f"{question}\n\n"
        "Please answer with only the letter (A, B, C, or D) of the correct option."
    )


def compute_stats(results_file):
    total = correct = 0
    by_cat = {}
    if not results_file.exists():
        return {}
    with open(results_file) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            ok = r.get("correct", False)
            if ok:
                correct += 1
            cat = r.get("category", "unknown")
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "correct": 0}
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["correct"] += 1

    return {
        "model": "MiniCPM-o-4_5-Q4_K_M.gguf",
        "benchmark": "MMStar",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0,
        "official_bf16": 73.1,
        "by_category": {
            k: {
                **v,
                "accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
            }
            for k, v in sorted(by_cat.items())
        },
        "updated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="MMStar vision benchmark")
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=9061)
    parser.add_argument("--samples", type=int, default=0, help="0 = all 1500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    base_url = f"http://{args.llama_host}:{args.llama_port}"
    RESULTS_DIR.mkdir(exist_ok=True)

    # Check server health
    log("Checking llama-server health...")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        log("llama-server is healthy")
    except Exception as e:
        log(f"ERROR: llama-server not reachable: {e}")
        sys.exit(1)

    # Load dataset
    log("Loading MMStar dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    log(f"Loaded {len(ds)} samples")

    # Sample if requested
    indices = list(range(len(ds)))
    if args.samples and args.samples < len(ds):
        random.seed(args.seed)
        indices = sorted(random.sample(indices, args.samples))
        log(f"Sampled {args.samples} questions (seed={args.seed})")

    log(f"Target: {len(indices)} questions → {base_url}")
    log("=" * 60)

    done_count = 0
    correct_count = 0

    for i, idx in enumerate(indices):
        sample = ds[idx]
        sample_idx = sample["index"]
        question = sample["question"]
        gt_answer = sample["answer"].strip().upper()
        category = sample["category"]
        image = sample["image"]

        prompt = build_prompt(question)
        image_b64 = image_to_base64(image)

        raw_response = None
        elapsed = 0
        for attempt in range(3):
            t0 = time.time()
            try:
                raw_response = query_model(base_url, image_b64, prompt, args.timeout)
                elapsed = time.time() - t0
                break
            except Exception as e:
                elapsed = time.time() - t0
                log(f"[{done_count+1}/{len(indices)}] #{sample_idx} attempt {attempt+1} ERROR: {e}")
                time.sleep(10 * (attempt + 1))
        if raw_response is None:
            log(f"[{done_count+1}/{len(indices)}] #{sample_idx} SKIPPED after 3 retries")
            continue

        pred = extract_answer(raw_response)
        is_correct = pred == gt_answer
        done_count += 1
        if is_correct:
            correct_count += 1
        acc = correct_count / done_count * 100

        result = {
            "index": sample_idx,
            "category": category,
            "l2_category": sample["l2_category"],
            "gt_answer": gt_answer,
            "pred_answer": pred,
            "raw_response": raw_response,
            "correct": is_correct,
            "elapsed_s": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        mark = "✓" if is_correct else "✗"
        log(
            f"[{done_count}/{len(indices)}] #{sample_idx} "
            f"{mark} pred={pred} gt={gt_answer} "
            f"acc={acc:.1f}% ({elapsed:.1f}s) [{category}]"
        )

        # Update summary every 10 questions
        if done_count % 10 == 0:
            summary = compute_stats(RESULTS_FILE)
            with open(SUMMARY_FILE, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    # Final summary
    summary = compute_stats(RESULTS_FILE)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("=" * 60)
    if not summary or summary["total"] == 0:
        log("No results collected.")
        return
    log(f"DONE! Accuracy: {summary['accuracy']}% ({summary['correct']}/{summary['total']})")
    log(f"Official MiniCPM-o 4.5 (bf16): 73.1%")
    gap = 73.1 - summary['accuracy']
    log(f"Quantization gap: {gap:+.1f} percentage points")
    log("")
    log("By category:")
    for cat, stats in summary.get("by_category", {}).items():
        log(f"  {cat}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
    log(f"\nResults: {RESULTS_FILE}")
    log(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
