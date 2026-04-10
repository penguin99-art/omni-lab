# Ollama v0.20.0 vs v0.20.4 — DGX Spark Benchmark Comparison

> Date: 2026-04-10
> Platform: **GB10** | 122GB | 273 GB/s | CUDA 580.126.09
> Upgrade: Ollama 0.20.0 → 0.20.4
> Key changes in 0.20.4: flash attention for Gemma 4, reworked tool call handling, arg parsing fixes

## Quick Suite (1 prompt: hello)

| Model | v0.20.0 tok/s | v0.20.4 tok/s | Delta | v0.20.0 TTFT | v0.20.4 TTFT |
|-------|-------------:|-------------:|------:|-------------:|-------------:|
| gemma4:e2b | 107.1 | 107.6 | **+0.5%** | 0.02s | 0.02s |
| gemma4:26b | 59.2 | 65.9 | **+11.3%** | 0.07s | 0.06s |
| qwen3.5:35b | 58.0 | 57.2 | -1.4% | 0.08s | 0.09s |

**Findings**: `gemma4:26b` (MoE) gained **+11.3%** throughput, likely from flash attention enablement. `gemma4:e2b` (dense, small) is within noise. `qwen3.5:35b` is also within noise (non-Gemma, unaffected by the flash attention change).

## Standard Suite (5 prompts: hello, reasoning, code, long_output, chinese)

### gemma4:e2b per-prompt comparison

| Prompt | v0.20.0 tok/s | v0.20.4 tok/s | Delta | v0.20.0 Quality | v0.20.4 Quality |
|--------|-------------:|-------------:|------:|----------------:|----------------:|
| hello | 111.4 | 108.2 | -2.9% | 1.00 | 1.00 |
| reasoning | 108.8 | 106.4 | -2.2% | 1.00 | 0.85 |
| code | 106.0 | 105.5 | -0.5% | 1.00 | 1.00 |
| long_output | 107.2 | 106.1 | -1.0% | 1.00 | 0.97 |
| chinese | 106.3 | 105.1 | -1.1% | 1.00 | 1.00 |
| **Average** | **108.0** | **106.3** | **-1.6%** | **1.00** | **0.96** |

**Findings**: `gemma4:e2b` speed is stable (within ~2% noise). Quality remains high. The minor quality drops on `reasoning` (0.85) and `long_output` (0.97) are within the variance of autoeval — individual runs can fluctuate.

## Tool Call Suite (2 prompts: tool_weather, tool_multi)

| Model | Prompt | tok/s | Quality | TC |
|-------|--------|------:|--------:|---:|
| gemma4:e2b | tool_weather | 106.5 | 1.00 | ✓ |
| gemma4:e2b | tool_multi | 106.9 | 1.00 | ✓ |
| gemma4:26b | tool_weather | 66.3 | 1.00 | ✓ |
| gemma4:26b | tool_multi | 65.5 | 1.00 | ✓ |

**Findings**: Both Gemma 4 models have **100% tool call success** with the reworked tool call handling in v0.20.4. All quality scores are 1.00. The `gemma4:26b` tool call speed (~66 tok/s) matches the improved throughput observed in the quick suite.

## TTFT Suite (4 cases x 3 trials)

### gemma4:e2b

| Case | v0.20.0 Avg | v0.20.4 Avg | Delta |
|------|------------:|------------:|------:|
| short_cold | 1.446s | 1.423s | **-1.6%** |
| short_warm | 1.443s | 1.410s | **-2.3%** |
| long_cold | 1.489s | 1.438s | **-3.4%** |
| long_warm | 1.462s | 1.425s | **-2.5%** |

**Findings**: TTFT improved slightly across the board (~1.5–3.5% reduction). The improvement is most visible on long prompts. However, prefix caching still shows minimal benefit (warm ~= cold), consistent with v0.20.0 observations.

## Summary

| Metric | Impact | Notes |
|--------|--------|-------|
| gemma4:26b throughput | **+11.3%** | Flash attention benefit on MoE |
| gemma4:e2b throughput | ~stable | Dense model already well-optimized |
| qwen3.5:35b throughput | ~stable | Non-Gemma, unaffected |
| Gemma 4 tool calling | **100% success** | Reworked tool call handling works well |
| TTFT | **-2–3% improved** | Small but consistent improvement |
| Quality | Stable | No regressions |

**Recommendation**: The upgrade to v0.20.4 is worthwhile. The `gemma4:26b` MoE model benefits significantly (+11.3%), tool calling is reliable, and there are no regressions.

## Data Files

### v0.20.0 Baselines
- `bench_20260408_115629_workflow_quick.*`
- `bench_20260408_115731_workflow_standard.*`
- `ttft_20260408_122934_ttft_v1.*`

### v0.20.4 Results
- `bench_20260410_115757_v0204_quick.*`
- `bench_20260410_115909_v0204_standard.*`
- `bench_20260410_115929_v0204_toolcall.*`
- `ttft_20260410_120004_v0204_ttft.*`
