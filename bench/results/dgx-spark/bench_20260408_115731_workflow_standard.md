# Benchmark Report — dgx-spark

> Generated: 2026-04-08 11:57  
> Platform: **GB10** | 122GB | 273 GB/s | CUDA 580.126.09  
> Test suite: hello, reasoning, code, long_output, chinese (5 prompts)  
> Models tested: 1 | Succeeded: 5 | Failed: 0

## Overall Ranking

| # | Model | Engine | Arch | Avg quality | Avg tok/s | Avg TTFT | Composite |
|---|-------|--------|------|------------:|----------:|---------:|----------:|
| 🥇 | gemma4:e2b | ollama | dense | **1.00** | 108.0 | 0.02s | 0.99 |

## Prompt: `hello`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|
| 🥇 | gemma4:e2b | ollama | dense | 7GB | **111.4** | 0.01s | 2.4s | 243 | 1.00 | 15.5 |

## Prompt: `reasoning`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|
| 🥇 | gemma4:e2b | ollama | dense | 7GB | **108.8** | 0.02s | 3.9s | 392 | 1.00 | 15.1 |

## Prompt: `code`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|
| 🥇 | gemma4:e2b | ollama | dense | 7GB | **106.0** | 0.02s | 18.1s | 1827 | 1.00 | 14.7 |

## Prompt: `long_output`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|
| 🥇 | gemma4:e2b | ollama | dense | 7GB | **107.2** | 0.01s | 10.5s | 1062 | 1.00 | 14.9 |

## Prompt: `chinese`

| # | Model | Engine | Arch | Size | tok/s | TTFT | Wall | Tokens | Quality | tok/s per GB |
|---|-------|--------|------|------|------:|-----:|-----:|-------:|--------:|-------------:|
| 🥇 | gemma4:e2b | ollama | dense | 7GB | **106.3** | 0.02s | 20.1s | 2046 | 1.00 | 14.8 |

## Key Findings

- **Fastest**: gemma4:e2b @ **111.4 tok/s**
- **Best TTFT**: gemma4:e2b @ **0.012s**
- **Best efficiency** (tok/s per GB): gemma4:e2b @ **15.5 tok/s/GB**
- **Best single-prompt quality**: gemma4:e2b on `hello` @ **1.00** (3 greetings recognized)

## Raw Data

- CSV: `bench_20260408_115731_workflow_standard.csv`
- JSON: `bench_20260408_115731_workflow_standard.json`
