# Benchmark Report — dgx-spark

> Generated: 2026-04-08 13:43  
> Platform: **GB10** | 122GB | 273 GB/s | CUDA 580.126.09  
> Test suite: hello (1 prompt)  
> Models tested: 1 | Succeeded: 1 | Failed: 0

## Overall Ranking

| # | Model | Engine | Arch | Avg quality | Avg tok/s | Avg TTFT | Composite |
|---|-------|--------|------|------------:|----------:|---------:|----------:|
| 🥇 | qwen3.5:35b | ollama | moe | **1.00** | 57.3 | 0.08s | 1.00 |

## Prompt: `hello`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | Mem peak | Swap peak | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|---------:|----------:|-------------:|
| 🥇 | qwen3.5:35b | ollama | moe | 23GB | **57.3** | 0.08s | 25.8s | 1012 | 1.00 | 48.4GB | 7.9GB | 2.5 |

## Key Findings

- **Fastest**: qwen3.5:35b @ **57.3 tok/s**
- **Best TTFT**: qwen3.5:35b @ **0.083s**
- **Best efficiency** (tok/s per GB): qwen3.5:35b @ **2.5 tok/s/GB**
- **Best single-prompt quality**: qwen3.5:35b on `hello` @ **1.00** (3 greetings recognized)
- **Highest memory peak**: qwen3.5:35b on `hello` @ **48.4GB** (peak +42.6GB)

## Raw Data

- CSV: `bench_20260408_134326_matrixcheck.csv`
- JSON: `bench_20260408_134326_matrixcheck.json`
