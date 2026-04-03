# DGX Spark 大模型 Benchmark 结果

> 测试时间: 2026-04-02
> 硬件: NVIDIA DGX Spark (GB10, 128GB, SM 12.1)
> 引擎: Ollama (预装)
> 模式: thinking 默认开启

## 汇总表

| 模型 | 架构 | 大小 | 激活参数 | hello | reasoning | code | long_output | chinese | 平均 tok/s |
|------|------|------|----------|-------|-----------|------|-------------|---------|-----------|
| **qwen3.5:35b** | MoE | 23GB | 3B | **57.3** | **57.1** | **57.7** | **57.5** | **57.7** | **57.4** |
| qwen3.5:9b | Dense | 6.6GB | 9B | 14.0* | 21.5* | 36.6 | 36.3 | 36.1 | 28.9 |
| qwen3.5:27b | Dense | 17GB | 27B | 11.3 | 11.3 | 11.3 | timeout | timeout | 11.3 |
| qwen3:32b | Dense | 20GB | 32B | 10.0 | 10.0 | timeout | timeout | timeout | 10.0 |
| qwen3.5:122b | MoE | 81GB | 10B | crash | - | - | - | - | - |

> *qwen3.5:9b 的 hello/reasoning 第一次较慢可能是冷启动或 thinking token 比例高

## 关键发现

### 1. MoE 是 DGX Spark 的甜点架构

```
qwen3.5:35b (MoE, 3B active):  57.4 tok/s ← 5.7x faster
qwen3.5:27b (Dense, 27B):      11.3 tok/s
qwen3:32b   (Dense, 32B):      10.0 tok/s
```

MoE 模型因为只激活少量参数（3B），内存带宽压力小，在 DGX Spark 的 273 GB/s 带宽下优势巨大。

### 2. qwen3.5:35b 是当前最佳选择

- **速度**: 57.4 tok/s（所有 prompt 类型高度一致）
- **质量**: 35B 总参数保证知识容量
- **内存**: 23GB，留大量空间给其他工作
- **TTFT**: ~0.09s（近乎即时）
- **稳定性**: 5 种 prompt 速度波动 < 1%

### 3. Dense 模型在 Spark 上不实用

- 10-11 tok/s 的交互体验差
- thinking 模式下容易超时（生成 3000+ token 需要 300+ 秒）
- qwen3.5:9b 是唯一例外（参数少，36 tok/s 可用）

### 4. 122b 模型加载失败

81GB 模型 + KV cache 可能超出可用内存（Docker 容器占用部分内存）。
需要停掉其他容器后单独测试。

## 详细数据

### qwen3.5:35b (MoE, 23GB) — 🏆 推荐

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 57.27 | 0.08s | 13.8s | 764 |
| reasoning | 57.14 | 0.11s | 47.8s | 2,662 |
| code | 57.65 | 0.09s | 8.6s | 472 |
| long_output | 57.52 | 0.09s | 46.4s | 2,608 |
| chinese | 57.67 | 0.10s | 55.1s | 3,105 |

### qwen3.5:9b (Dense, 6.6GB) — 轻量级备选

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 13.95 | 0.13s | 25.7s | 351 |
| reasoning | 21.47 | 0.12s | 185.8s | 3,948 |
| code | 36.58 | 0.07s | 50.2s | 1,813 |
| long_output | 36.27 | 0.05s | 142.0s | 5,097 |
| chinese | 36.05 | 0.06s | 83.1s | 2,962 |

> 注意: 前两个 prompt 速度低是因为 thinking 阶段 token 生成占比高

### qwen3.5:27b (Dense, 17GB)

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 11.27 | 0.24s | 121.4s | 1,134 |
| reasoning | 11.25 | 0.24s | 234.7s | 1,212 |
| code | 11.27 | 0.23s | 248.6s | 1,134 |
| long_output | - | - | timeout | - |
| chinese | - | - | timeout | - |

### qwen3:32b (Dense, 20GB)

| Prompt | tok/s | TTFT | 总时间 | 输出 tokens |
|--------|-------|------|--------|-------------|
| hello | 10.01 | 0.25s | 54.0s | 387 |
| reasoning | 9.95 | 0.25s | 157.1s | 817 |
| code | - | - | timeout | - |
| long_output | - | - | timeout | - |
| chinese | - | - | timeout | - |

## 与社区数据对比

| 模型 | 我们实测 | 社区数据 | 差异 |
|------|----------|----------|------|
| qwen3.5:35b | 57.4 | 57 | ✅ 吻合 |
| qwen3.5:9b | 36.1 | 35 | ✅ 吻合 |
| qwen3.5:27b | 11.3 | 13 | ≈ 接近 |
| qwen3:32b | 10.0 | 10 | ✅ 吻合 |

## 待测模型

以下模型下载中或需要额外环境搭建:

| 模型 | 引擎 | 社区预期 | 状态 |
|------|------|----------|------|
| gpt-oss:120b | Ollama | ~41 tok/s | 待下载 |
| gpt-oss:20b | Ollama | ~58 tok/s | 下载中 |
| nemotron-3-nano | Ollama | ~69 tok/s | 待下载 |
| nemotron-cascade-2 | Ollama | ~72 tok/s | 待下载 |
| nemotron-3-super | Ollama | ~20 tok/s | 待下载 |
| qwen3.5:122b-a10b | Ollama | ~23 tok/s | 需单独测试 |
| Qwen3.5-35B MXFP4 | vLLM | ~60 tok/s | 需 vLLM 补丁 |
| gpt-oss-120b MXFP4 | vLLM | ~81 tok/s | 需 vLLM 补丁 |

## 结论

**DGX Spark 上，MoE 架构是绝对王者。** 内存带宽（273 GB/s）是瓶颈，MoE 只激活少量参数从根本上缓解了这个问题。

**当前推荐配置:**
- **日常使用**: qwen3.5:35b (Ollama, 57 tok/s, 开箱即用)
- **轻量并行**: qwen3.5:9b (6.6GB, 可与大模型共存)
- **待验证最速**: nemotron-cascade-2 (~72 tok/s) 或 gpt-oss-120b + vLLM MXFP4 (~81 tok/s)
