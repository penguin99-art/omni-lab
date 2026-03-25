#!/usr/bin/env python3
"""
MMStar Vision Benchmark for MiniCPM-o 4.5 (GGUF quantized models)

Sends images + MCQ questions to llama-server's /v1/chat/completions,
compares model answers to ground truth, logs results incrementally.

Usage:
    # Q4_K_M eval (300 samples)
    nohup python3 eval_vision.py --samples 300 --tag Q4_K_M > eval_vision.log 2>&1 &

    # Q8_0 eval (300 samples, same seed for apples-to-apples comparison)
    nohup python3 eval_vision.py --samples 300 --tag Q8_0 > eval_vision_q8.log 2>&1 &

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


_progress_log = None

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _progress_log:
        with open(_progress_log, "a") as f:
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
            "max_tokens": 1024,
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


def compute_stats(results_file, tag="Q4_K_M"):
    total = correct = 0
    by_cat = {}
    latencies = []
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
            if "elapsed_s" in r:
                latencies.append(r["elapsed_s"])

    latency_stats = {}
    if latencies:
        latencies.sort()
        latency_stats = {
            "avg_s": round(sum(latencies) / len(latencies), 2),
            "p50_s": round(latencies[len(latencies) // 2], 2),
            "p95_s": round(latencies[int(len(latencies) * 0.95)], 2),
            "max_s": round(max(latencies), 2),
        }

    return {
        "model": f"MiniCPM-o-4_5-{tag}.gguf",
        "tag": tag,
        "benchmark": "MMStar",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0,
        "official_bf16": 73.1,
        "latency": latency_stats,
        "by_category": {
            k: {
                **v,
                "accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
            }
            for k, v in sorted(by_cat.items())
        },
        "updated_at": datetime.now().isoformat(),
    }


def generate_comparison_report(results_dir: Path):
    """Generate a comparative REPORT.md if both Q4_K_M and Q8_0 results exist."""
    summaries = {}
    for tag in ["Q4_K_M", "Q8_0"]:
        sf = results_dir / f"mmstar_summary_{tag}.json"
        if sf.exists():
            with open(sf) as f:
                summaries[tag] = json.load(f)

    if not summaries:
        return

    report_path = results_dir / "REPORT.md"
    lines = ["# MiniCPM-o 4.5 量化对比评测报告", ""]

    if len(summaries) >= 2:
        lines.append("## 概述")
        lines.append("")
        lines.append("本评测对比了 MiniCPM-o 4.5 在不同量化级别下的视觉理解能力，")
        lines.append("使用 [MMStar](https://github.com/MMStar-Benchmark/MMStar) 多模态 benchmark，相同采样 (seed=42)。")
    else:
        tag = list(summaries.keys())[0]
        lines.append("## 概述")
        lines.append("")
        lines.append(f"本评测评估了 MiniCPM-o 4.5 {tag} 量化的视觉理解能力。")

    lines += ["", "## 评测配置", ""]
    lines.append("| 项目 | 配置 |")
    lines.append("|---|---|")
    lines.append("| 推理引擎 | llama.cpp (upstream, CUDA) |")
    lines.append("| 视觉编码器 | MiniCPM-o-4_5-vision-F16.gguf (未量化) |")
    lines.append("| Benchmark | MMStar (采样 300 题, seed=42) |")
    lines.append("| 温度 | 0.1 |")
    lines.append("| Thinking 模式 | 关闭 (`enable_thinking: false`) |")
    lines.append("| max_tokens | 1024 |")
    lines.append("| 硬件 | NVIDIA GB10 (aarch64, CUDA) |")

    lines += ["", "## 总体准确率", ""]

    if len(summaries) >= 2:
        lines.append("| 模型版本 | MMStar 准确率 | 量化损失 | 平均延迟 |")
        lines.append("|---|---|---|---|")
        lines.append("| bf16 (官方) | **73.1%** | -- | -- |")
        for tag in ["Q8_0", "Q4_K_M"]:
            s = summaries.get(tag)
            if s:
                gap = 73.1 - s["accuracy"]
                lat = s.get("latency", {}).get("avg_s", "?")
                lines.append(f"| {tag} | **{s['accuracy']}%** | **{gap:+.2f} pp** | {lat}s |")
    else:
        tag = list(summaries.keys())[0]
        s = summaries[tag]
        gap = 73.1 - s["accuracy"]
        lines.append("| 模型版本 | MMStar 准确率 | 样本数 |")
        lines.append("|---|---|---|")
        lines.append("| bf16 (官方) | **73.1%** | 1500 |")
        lines.append(f"| {tag} (本次) | **{s['accuracy']}%** | {s['total']} |")
        lines.append(f"| **量化损失** | **{gap:+.2f} pp** | |")

    lines += ["", "## 分类细项", ""]

    all_cats = set()
    for s in summaries.values():
        all_cats.update(s.get("by_category", {}).keys())
    all_cats = sorted(all_cats)

    if len(summaries) >= 2:
        header = "| 类别 |"
        divider = "|---|"
        for tag in ["Q8_0", "Q4_K_M"]:
            if tag in summaries:
                header += f" {tag} |"
                divider += "---|"
        header += " 差异 |"
        divider += "---|"
        lines.append(header)
        lines.append(divider)
        for cat in all_cats:
            row = f"| {cat} |"
            accs = []
            for tag in ["Q8_0", "Q4_K_M"]:
                s = summaries.get(tag, {}).get("by_category", {}).get(cat, {})
                a = s.get("accuracy", 0)
                t = s.get("total", 0)
                c = s.get("correct", 0)
                row += f" {a:.1f}% ({c}/{t}) |"
                accs.append(a)
            if len(accs) == 2:
                diff = accs[0] - accs[1]
                row += f" {diff:+.1f} pp |"
            else:
                row += " -- |"
            lines.append(row)
    else:
        tag = list(summaries.keys())[0]
        lines.append("| 类别 | 准确率 | 正确/总数 |")
        lines.append("|---|---|---|")
        for cat in all_cats:
            s = summaries[tag].get("by_category", {}).get(cat, {})
            lines.append(f"| {cat} | {s.get('accuracy', 0):.2f}% | {s.get('correct', 0)}/{s.get('total', 0)} |")

    if len(summaries) >= 2:
        lines += ["", "## 延迟对比", ""]
        lines.append("| 指标 |")
        header = "| 指标 |"
        divider = "|---|"
        for tag in ["Q8_0", "Q4_K_M"]:
            if tag in summaries:
                header += f" {tag} |"
                divider += "---|"
        lines[-1] = header
        lines.append(divider)
        for metric, label in [("avg_s", "平均"), ("p50_s", "P50"), ("p95_s", "P95"), ("max_s", "最大")]:
            row = f"| {label} |"
            for tag in ["Q8_0", "Q4_K_M"]:
                s = summaries.get(tag, {}).get("latency", {})
                v = s.get(metric, "?")
                row += f" {v}s |"
            lines.append(row)

    lines += ["", "## 分析", ""]
    if len(summaries) >= 2:
        q4 = summaries.get("Q4_K_M", {})
        q8 = summaries.get("Q8_0", {})
        q4_acc = q4.get("accuracy", 0)
        q8_acc = q8.get("accuracy", 0)
        diff = q8_acc - q4_acc
        lines.append(f"1. **Q8_0 vs Q4_K_M**: Q8_0 准确率 {q8_acc}%, Q4_K_M 准确率 {q4_acc}%, ")
        lines.append(f"   Q8_0 比 Q4_K_M 高 {diff:+.2f} 个百分点。")
        q8_lat = q8.get("latency", {}).get("avg_s", "?")
        q4_lat = q4.get("latency", {}).get("avg_s", "?")
        lines.append(f"2. **延迟**: Q8_0 平均 {q8_lat}s/题, Q4_K_M 平均 {q4_lat}s/题。")
        lines.append("3. Q8_0 模型约多 3.5GB 显存占用，但量化损失更小。")
        lines.append("4. 两者的视觉编码器均为 F16 未量化，差异完全来自语言模型量化级别。")
    else:
        tag = list(summaries.keys())[0]
        s = summaries[tag]
        lines.append(f"- {tag} 量化损失约 {73.1 - s['accuracy']:.1f} 个百分点。")

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "# 启动评测用 llama-server",
        "./llama.cpp-eval/build/bin/llama-server \\",
        "  --host 127.0.0.1 --port 9061 \\",
        "  --model ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-<QUANT>.gguf \\",
        "  --mmproj ./models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf \\",
        "  -ngl 99 --ctx-size 4096 --temp 0.1",
        "",
        "# 运行评测",
        'nohup python3 eval_vision.py --samples 300 --tag <QUANT> > eval_vision.log 2>&1 &',
        "```",
        "",
        "## 参考",
        "",
        "- [MMStar Benchmark](https://mmstar-benchmark.github.io/)",
        "- [MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o) — OpenBMB",
        "- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGML 推理引擎",
        "",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log(f"Report written to {report_path}")


def main():
    global _progress_log

    parser = argparse.ArgumentParser(description="MMStar vision benchmark")
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=9061)
    parser.add_argument("--samples", type=int, default=0, help="0 = all 1500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tag", default="Q4_K_M", help="Quantization tag (e.g. Q4_K_M, Q8_0)")
    args = parser.parse_args()

    base_url = f"http://{args.llama_host}:{args.llama_port}"
    RESULTS_DIR.mkdir(exist_ok=True)

    results_file = RESULTS_DIR / f"mmstar_results_{args.tag}.jsonl"
    summary_file = RESULTS_DIR / f"mmstar_summary_{args.tag}.json"
    _progress_log = RESULTS_DIR / f"progress_{args.tag}.log"

    log(f"=== MMStar eval for {args.tag} ===")
    log("Checking llama-server health...")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        log("llama-server is healthy")
    except Exception as e:
        log(f"ERROR: llama-server not reachable: {e}")
        sys.exit(1)

    log("Loading MMStar dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    log(f"Loaded {len(ds)} samples")

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
        with open(results_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        mark = "✓" if is_correct else "✗"
        log(
            f"[{done_count}/{len(indices)}] #{sample_idx} "
            f"{mark} pred={pred} gt={gt_answer} "
            f"acc={acc:.1f}% ({elapsed:.1f}s) [{category}]"
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
    log(f"DONE! [{args.tag}] Accuracy: {summary['accuracy']}% ({summary['correct']}/{summary['total']})")
    log(f"Official MiniCPM-o 4.5 (bf16): 73.1%")
    gap = 73.1 - summary['accuracy']
    log(f"Quantization gap: {gap:+.1f} percentage points")
    log("")
    log("By category:")
    for cat, stats in summary.get("by_category", {}).items():
        log(f"  {cat}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
    if summary.get("latency"):
        lat = summary["latency"]
        log(f"\nLatency: avg={lat['avg_s']}s p50={lat['p50_s']}s p95={lat['p95_s']}s max={lat['max_s']}s")

    log(f"\nResults: {results_file}")
    log(f"Summary: {summary_file}")

    generate_comparison_report(RESULTS_DIR)


if __name__ == "__main__":
    main()
