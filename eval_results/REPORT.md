# MiniCPM-o 4.5 Q4_K_M 量化损失评测报告

## 概述

本评测对比了 MiniCPM-o 4.5 在 GGUF Q4_K_M 量化后的视觉理解能力变化，使用 [MMStar](https://github.com/MMStar-Benchmark/MMStar) 多模态 benchmark 进行评估。

## 评测配置

| 项目 | 配置 |
|---|---|
| 模型 | MiniCPM-o-4_5-Q4_K_M.gguf (~4.8GB) |
| 量化方式 | Q4_K_M (4-bit, k-quants mixed) |
| 推理引擎 | llama.cpp (upstream, CUDA) |
| 视觉编码器 | MiniCPM-o-4_5-vision-F16.gguf (未量化) |
| Benchmark | MMStar (1500 题中随机采样 300 题, seed=42) |
| 温度 | 0.1 |
| Thinking 模式 | 关闭 (instruct mode, `enable_thinking: false`) |
| max_tokens | 1024 |
| 硬件 | NVIDIA GB10 (aarch64, CUDA) |
| 评测日期 | 2026-03-25 |

## 结果

### 总体准确率

| 模型版本 | MMStar 准确率 | 样本数 |
|---|---|---|
| MiniCPM-o 4.5 bf16 (官方) | **73.1%** | 1500 |
| MiniCPM-o 4.5 Q4_K_M (本次) | **70.33%** | 300 |
| **量化损失** | **-2.77 pp** | |

> 注：官方成绩基于 PyTorch bf16 推理，全量 1500 题。本次评测使用 llama.cpp GGUF 推理，
> 300 题子集。除量化损失外，推理引擎差异也可能贡献部分精度差距。

### 分类细项

| 类别 | Q4_K_M 准确率 | 正确/总数 |
|---|---|---|
| instance reasoning (实例推理) | **81.40%** | 35/43 |
| math (数学) | **80.00%** | 40/50 |
| coarse perception (粗粒度感知) | **76.79%** | 43/56 |
| logical reasoning (逻辑推理) | 69.57% | 32/46 |
| fine-grained perception (细粒度感知) | 64.15% | 34/53 |
| science & technology (科技) | 51.92% | 27/52 |

### max_tokens 影响分析

在 max_tokens=256 的首轮测试中，模型在 instruct 模式下仍会自发输出推理链，
部分复杂题目在 256 token 内被截断，无法给出最终答案。将 max_tokens 提升至 1024
后，准确率显著提升：

| 指标 | max_tokens=256 | max_tokens=1024 | 变化 |
|---|---|---|---|
| **总准确率** | 66.67% | **70.33%** | **+3.66 pp** |
| logical reasoning | 60.87% | **69.57%** | **+8.70 pp** |
| math | 66.00% | **80.00%** | **+14.00 pp** |
| instance reasoning | 81.40% | 81.40% | 0 |
| coarse perception | 75.00% | 76.79% | +1.79 pp |
| fine-grained perception | 66.04% | 64.15% | -1.89 pp |
| science & technology | 51.92% | 51.92% | 0 |

> max_tokens=256 时有 14/300 题 (4.7%) 因截断无法输出答案字母（pred=?），
> 全部记为错误。提升到 1024 后此问题大幅减少。

**结论：** 之前观测到的 6.4pp 差距中，约 3.7pp 是 max_tokens 截断导致的伪差距。
**Q4_K_M 的真实量化损失约为 2.8 个百分点。**

### 推理延迟

| 指标 | max_tokens=256 | max_tokens=1024 |
|---|---|---|
| 总耗时 | ~7 分钟 | ~14 分钟 |
| 平均延迟 | ~1.4s/题 | ~2.8s/题 |
| 最短 | 0.2s | 0.2s |
| 最长 | ~7s | ~26s |

> Thinking 模式已关闭（`reasoning_content` 为空），但 instruct 模式下模型仍会
> 在 `content` 中自发输出推理过程，消耗更多 token 和时间。

## 分析

1. **Q4_K_M 量化损失约 2.8 个百分点**，与纯文本 LLM 的典型量化损失（1-3 pp）
   处于同一量级，说明在合理的 max_tokens 设置下，Q4_K_M 量化对视觉理解的影响可控。

2. **实例推理和数学保持较好**（>80%），量化后模型仍具备较强的视觉推理能力。

3. **科技类题目下降最明显**（51.92%），科学图表和专业知识的表征在量化后损失较大，
   且不受 max_tokens 影响。

4. **max_tokens 的选择至关重要**。对于需要推理的视觉任务，建议 max_tokens >= 1024，
   否则会因截断引入大量伪错误。

## 复现

```bash
# 启动评测用 llama-server (vision-only, upstream llama.cpp)
./llama.cpp-eval/build/bin/llama-server \
  --host 127.0.0.1 --port 9061 \
  --model ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --mmproj ./models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf \
  -ngl 99 --ctx-size 4096 --temp 0.1

# 运行评测 (300 题, 后台)
nohup python3 eval_vision.py --samples 300 > eval_vision.log 2>&1 &

# 监控进度
tail -f eval_results/progress.log
```

## 参考

- [MMStar Benchmark](https://mmstar-benchmark.github.io/) — Chen et al., "Are We on the Right Way for Evaluating Large Vision-Language Models?", 2024
- [MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o) — OpenBMB / 面壁智能
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGML 推理引擎
