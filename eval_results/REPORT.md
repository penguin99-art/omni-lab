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
| Thinking 模式 | 关闭 (instruct mode) |
| 硬件 | NVIDIA GB10 (aarch64, CUDA) |
| 评测日期 | 2026-03-25 |

## 结果

### 总体准确率

| 模型版本 | MMStar 准确率 | 样本数 |
|---|---|---|
| MiniCPM-o 4.5 bf16 (官方) | **73.1%** | 1500 |
| MiniCPM-o 4.5 Q4_K_M (本次) | **66.67%** | 300 |
| **量化损失** | **-6.43 pp** | |

> 注：官方成绩基于 PyTorch bf16 推理，全量 1500 题。本次评测使用 llama.cpp GGUF 推理，
> 300 题子集。除量化损失外，推理引擎差异也可能贡献部分精度差距。

### 分类细项

| 类别 | Q4_K_M 准确率 | 正确/总数 |
|---|---|---|
| instance reasoning (实例推理) | **81.40%** | 35/43 |
| coarse perception (粗粒度感知) | **75.00%** | 42/56 |
| math (数学) | 66.00% | 33/50 |
| fine-grained perception (细粒度感知) | 66.04% | 35/53 |
| logical reasoning (逻辑推理) | 60.87% | 28/46 |
| science & technology (科技) | 51.92% | 27/52 |

### 推理延迟

| 指标 | 数值 |
|---|---|
| 平均延迟 | ~1.4s/题 |
| 最短 | 0.2s (直接输出字母) |
| 最长 | ~7s (含推理过程) |
| 300 题总耗时 | ~7 分钟 |

## 分析

1. **量化损失约 6.4 个百分点**，高于纯文本 LLM 的典型量化损失（1-3 pp），
   说明多模态任务对量化更敏感。

2. **实例推理和粗粒度感知保持较好**（>75%），这类任务更依赖整体图像理解，
   对权重精度要求相对较低。

3. **科技类题目下降最明显**（51.92%），可能因为科学图表和专业知识的表征
   在量化后损失较大。

4. **逻辑推理类偏低**（60.87%），部分原因是复杂推理题在 max_tokens=256 内
   无法完成推理链就被截断。

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
