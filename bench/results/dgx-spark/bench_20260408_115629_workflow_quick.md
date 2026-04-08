# Benchmark Report — dgx-spark

> Generated: 2026-04-08 11:56  
> Platform: **GB10** | 122GB | 273 GB/s | CUDA 580.126.09  
> Test suite: hello (1 prompt)  
> Models tested: 3 | Succeeded: 3 | Failed: 0

## Overall Ranking

| # | Model | Engine | Arch | Avg quality | Avg tok/s | Avg TTFT | Composite |
|---|-------|--------|------|------------:|----------:|---------:|----------:|
| 🥇 | gemma4:e2b | ollama | dense | **1.00** | 107.1 | 0.02s | 1.00 |
| 🥈 | gemma4:26b | ollama | moe | **1.00** | 59.2 | 0.07s | 0.87 |
| 🥉 | qwen3.5:35b | ollama | moe | **1.00** | 58.0 | 0.08s | 0.86 |

## Prompt: `hello`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|
| 🥇 | gemma4:e2b | ollama | dense | 7GB | **107.1** | 0.02s | 6.3s | 276 | 1.00 | 14.9 |
| 🥈 | gemma4:26b | ollama | moe | 18GB | **59.2** | 0.07s | 10.2s | 200 | 1.00 | 3.3 |
| 🥉 | qwen3.5:35b | ollama | moe | 23GB | **58.0** | 0.08s | 21.9s | 740 | 1.00 | 2.5 |

## Key Findings

- **Fastest**: gemma4:e2b @ **107.1 tok/s**
- **Best TTFT**: gemma4:e2b @ **0.018s**
- **Best efficiency** (tok/s per GB): gemma4:e2b @ **14.9 tok/s/GB**
- **Best single-prompt quality**: qwen3.5:35b on `hello` @ **1.00** (3 greetings recognized)
- **MoE avg**: 58.6 tok/s vs **Dense avg**: 107.1 tok/s (Dense wins by 83%)

## Architecture Comparison

| Metric | MoE | Dense |
|--------|----:|------:|
| Avg tok/s | 58.6 | 107.1 |
| Avg TTFT | 0.077s | 0.018s |
| Avg tok/s/GB | 2.9 | 14.9 |
| Avg quality | 1.00 | 1.00 |
| Count | 2 | 1 |

## Raw Data

- CSV: `bench_20260408_115629_workflow_quick.csv`
- JSON: `bench_20260408_115629_workflow_quick.json`
