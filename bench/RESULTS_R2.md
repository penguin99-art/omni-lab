# Round 2: Gemma 4 全系列 vs Qwen 3.5:35b 对比测试结果

> 测试时间: 2026-04-03
> 硬件: DGX Spark (GB10, 128GB LPDDR5x, 273 GB/s, CUDA 13.0)
> 引擎: Ollama 0.20.0
> 量化: 所有模型使用 Q4_K_M 默认量化

## 汇总排行榜

### 速度排行 (Standard Suite 平均 tok/s)

| 排名 | 模型 | 架构 | 激活参数 | 大小 | 平均 tok/s | TTFT | 工具调用 |
|------|------|------|----------|------|-----------|------|---------|
| **1** | **qwen3.5:35b** | MoE | 3B | 23GB | **55.8** | 0.09s | ✓✓ |
| 2 | gemma4:26b | MoE | 4B | 18GB | 35.1 | 0.07s | ✓✓ |
| 3 | gemma4:e2b | Dense PLE | 2B | 7.2GB | 33.9 | 0.03s | ✓✓ |
| 4 | gemma4:e4b | Dense PLE | 4B | 9.6GB | 25.9 | 0.05s | ✓✓ |
| 5 | gemma4:31b | Dense | 31B | 20GB | 8.1 | 0.17s | ✓✓ |

### 效率排行 (tok/s per GB)

| 排名 | 模型 | tok/s | 大小 | tok/s/GB |
|------|------|-------|------|----------|
| **1** | **gemma4:e2b** | 33.9 | 7.2GB | **4.71** |
| 2 | gemma4:e4b | 25.9 | 9.6GB | 2.70 |
| 3 | qwen3.5:35b | 55.8 | 23GB | 2.43 |
| 4 | gemma4:26b | 35.1 | 18GB | 1.95 |
| 5 | gemma4:31b | 8.1 | 20GB | 0.41 |

## 详细测试数据

### qwen3.5:35b (MoE, 23GB, 3B active) — 速度冠军

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 54.3 | 0.10s | 20.2s | 449 |
| reasoning | 56.0 | 0.09s | 39.2s | 2,147 |
| code | 56.5 | 0.09s | 10.8s | 587 |
| long_output | 55.9 | 0.09s | 37.8s | 2,068 |
| chinese | 56.1 | 0.09s | 53.8s | 2,957 |
| **tool_weather** | 56.7 | 0.27s | 2.1s | 93 | ✓ |
| **tool_multi** | 57.0 | 0.32s | 2.0s | 86 | ✓ |

### gemma4:26b (MoE, 18GB, 4B active) — Gemma 最快

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 37.9 | 0.03s | 3.9s | 138 |
| reasoning | 37.5 | 0.10s | 11.5s | 414 |
| code | 33.7 | 0.08s | 53.4s | 1,774 |
| long_output | 34.3 | 0.06s | 38.9s | 1,312 |
| chinese | 32.0 | 0.07s | 66.3s | 2,093 |
| **tool_weather** | 36.4 | 0.11s | 2.1s | 64 | ✓ |
| **tool_multi** | 36.0 | 0.11s | 2.4s | 74 | ✓ |

### gemma4:e2b (Dense PLE, 7.2GB, 2B effective) — 最轻量

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 37.0 | 0.02s | 7.0s | 248 |
| reasoning | 35.8 | 0.04s | 14.0s | 488 |
| code | 32.9 | 0.04s | 51.9s | 1,681 |
| long_output | 33.1 | 0.02s | 35.9s | 1,168 |
| chinese | 30.9 | 0.05s | 66.7s | 2,032 |
| **tool_weather** | 35.3 | 0.04s | 8.9s | 301 | ✓ |
| **tool_multi** | 32.7 | 0.07s | 8.4s | 261 | ✓ |

### gemma4:e4b (Dense PLE, 9.6GB, 4B effective)

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 27.2 | 0.03s | 10.2s | 268 |
| reasoning | 27.2 | 0.05s | 19.3s | 515 |
| code | 24.9 | 0.06s | 49.8s | 1,224 |
| long_output | 26.5 | 0.04s | 37.0s | 967 |
| chinese | 23.9 | 0.08s | 106.5s | 2,518 |
| **tool_weather** | 26.8 | 0.08s | 7.7s | 197 | ✓ |
| **tool_multi** | 26.7 | 0.06s | 13.3s | 347 | ✓ |

### gemma4:31b (Dense, 20GB, 31B full) — 最慢但最大

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 8.5 | 0.16s | 14.2s | 117 |
| reasoning | 8.3 | 0.21s | 41.7s | 342 |
| code | 8.0 | 0.16s | 170.2s | 1,349 |
| long_output | 8.0 | 0.14s | 148.2s | 1,184 |
| chinese | 7.7 | 0.16s | 268.9s | 2,071 |
| **tool_weather** | 8.4 | 0.26s | 16.3s | 83 | ✓ |
| **tool_multi** | 8.4 | 0.31s | 11.9s | 95 | ✓ |

## 关键分析

### 1. MoE 在 Spark 上的绝对优势再次验证

```
qwen3.5:35b (MoE, 3B active):  55.8 tok/s  ← 6.9x faster than 31b dense
gemma4:26b  (MoE, 4B active):  35.1 tok/s  ← 4.3x faster than 31b dense
gemma4:31b  (Dense, 31B):       8.1 tok/s
```

MoE 架构在 Spark 的 273 GB/s 带宽下优势巨大。qwen3.5:35b 只激活 3B 参数，比 gemma4:26b 的 4B 激活参数更少，所以更快。

### 2. Gemma 4 PLE 架构的惊喜

gemma4:e2b（2B effective）跑出了 33.9 tok/s，和 gemma4:26b MoE（35.1 tok/s）几乎持平！这意味着 Google 的 Per-Layer Embedding 技术在小模型上非常高效。

反常的是 **e4b 比 e2b 更慢**（25.9 vs 33.9 tok/s），说明 PLE 架构不是简单的参数缩放——4B 的计算复杂度超过了 2B 的比例。

### 3. 所有模型都通过了工具调用测试

**5 个模型（4 个 Gemma 4 + qwen3.5:35b）全部成功通过工具调用测试**（单工具 + 多工具链式调用），包括最小的 gemma4:e2b。

注：社区数据中 Qwen 3.5 的工具调用被标记为 "partial"（△），意指未专门微调、复杂场景可能不稳定。但在我们的标准测试中 qwen3.5:35b 两项均通过。更复杂的 Agent 场景（多轮工具编排、错误恢复等）需要额外测试。

### 4. Gemma 4 输出更精炼

对比 hello prompt 的输出 token 数：
```
qwen3.5:35b:  449 tokens (大量 thinking tokens)
gemma4:26b:   138 tokens (简洁直接)
gemma4:e2b:   248 tokens
gemma4:31b:   117 tokens (最精炼)
```

Gemma 4 不使用 thinking 模式，输出更简洁。对于需要快速响应的场景这是优势。

### 5. TTFT (首 token 时间) 对比

```
gemma4:e2b:   0.02-0.05s  ← 最快
gemma4:e4b:   0.03-0.08s
gemma4:26b:   0.03-0.11s
qwen3.5:35b:  0.09-0.32s  ← thinking 初始化有额外延迟
gemma4:31b:   0.14-0.31s  ← 最慢（Dense 大模型）
```

## 量化建议

基于 128GB Spark 内存，每个模型的最佳量化版本选择：

| 模型 | 推荐量化 | 大小 | 原因 |
|------|----------|------|------|
| gemma4:e2b | **Q4_K_M (默认)** | 7.2GB | 已足够快(34tok/s)，BF16仅10GB增量不值得 |
| gemma4:e4b | Q4_K_M (默认) | 9.6GB | 性价比低于e2b，不建议优先使用 |
| gemma4:26b | **Q4_K_M (默认)** | 18GB | MoE架构已高效，Q8仅增至28GB但速度可能下降 |
| gemma4:31b | Q4_K_M (默认) | 20GB | 太慢(8tok/s)，不建议使用更大量化 |
| qwen3.5:35b | Q4_K_M (默认) | 23GB | 已是最快(56tok/s)，无需更高量化 |

> 注：Q8_0 和 BF16 版本可提升精度但会降低速度（参数更多 → 带宽瓶颈更严重）。
> 在 Spark 的 273 GB/s 带宽下，Q4_K_M 是最佳平衡点。

## 最终推荐

| 使用场景 | 推荐模型 | tok/s | 内存 | 原因 |
|----------|----------|-------|------|------|
| **日常助手（速度优先）** | qwen3.5:35b | 55.8 | 23GB | 绝对最快，质量一流 |
| **轻量嵌入/并发** | gemma4:e2b | 33.9 | 7.2GB | 仅7GB，可与大模型共存 |
| **Agent/工具调用** | gemma4:26b | 35.1 | 18GB | 原生工具调用+MoE速度 |
| **多模态（视觉+音频）** | gemma4:e4b 或 26b | 26-35 | 10-18GB | Gemma4 原生支持视觉输入 |
| **高质量推理（不急）** | gemma4:31b | 8.1 | 20GB | 最大参数量，质量最高 |

## 测试环境详情

```
Ollama:       0.20.0 (用户级安装 @ port 11436)
GPU:          NVIDIA GB10 (SM 12.1, CUDA 13.0)
统一内存:     128GB LPDDR5x (~273 GB/s)
Flash Attn:   enabled
OS:           Ubuntu 24.04.4 LTS (aarch64)
```

## 原始数据文件

```
bench/results/dgx-spark/
├── bench_20260403_100942_r2_baseline.{csv,json}              # qwen3.5:35b quick
├── bench_20260403_103024_r2_gemma4_e2b_quick.{csv,json}      # e2b quick
├── bench_20260403_103327_r2_gemma4_e2b_standard.{csv,json}   # e2b standard
├── bench_20260403_103349_r2_gemma4_e2b_toolcall.{csv,json}   # e2b toolcall
├── bench_20260403_120355_r2_gemma4_e4b_quick.{csv,json}      # e4b quick
├── bench_20260403_120738_r2_gemma4_e4b_standard.{csv,json}   # e4b standard
├── bench_20260403_120759_r2_gemma4_e4b_toolcall.{csv,json}   # e4b toolcall
├── bench_20260403_124226_r2_gemma4_26b_quick.{csv,json}      # 26b quick
├── bench_20260403_124520_r2_gemma4_26b_standard.{csv,json}   # 26b standard
├── bench_20260403_124524_r2_gemma4_26b_toolcall.{csv,json}   # 26b toolcall
├── bench_20260403_154655_r2_gemma4_31b_quick.{csv,json}      # 31b quick
├── bench_20260403_155741_r2_gemma4_31b_standard.{csv,json}   # 31b standard
├── bench_20260403_160503_r2_gemma4_31b_toolcall.{csv,json}   # 31b toolcall
├── bench_20260403_160748_r2_qwen35_35b_standard.{csv,json}   # qwen3.5 standard (control)
└── bench_20260403_160757_r2_qwen35_35b_toolcall.{csv,json}   # qwen3.5 toolcall (control)
```
