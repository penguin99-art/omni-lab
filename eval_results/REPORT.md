# MiniCPM-o 4.5 量化对比评测报告

## 概述

本评测对比了 MiniCPM-o 4.5 在不同量化级别下的视觉理解能力，
使用 [MMStar](https://github.com/MMStar-Benchmark/MMStar) 多模态 benchmark，相同采样 (seed=42)。

## 评测配置

| 项目 | 配置 |
|---|---|
| 推理引擎 | llama.cpp (upstream, CUDA) |
| 视觉编码器 | MiniCPM-o-4_5-vision-F16.gguf (未量化) |
| Benchmark | MMStar (采样 300 题, seed=42) |
| 温度 | 0.1 |
| Thinking 模式 | 关闭 (`enable_thinking: false`) |
| max_tokens | 1024 |
| 硬件 | NVIDIA GB10 (aarch64, CUDA) |

## 总体准确率

| 模型版本 | MMStar 准确率 | 量化损失 | 平均延迟 |
|---|---|---|---|
| bf16 (官方) | **73.1%** | -- | -- |
| Q8_0 | **69.67%** | **+3.43 pp** | 3.97s |
| Q4_K_M | **70.33%** | **+2.77 pp** | 2.5s |

## 分类细项

| 类别 | Q8_0 | Q4_K_M | 差异 |
|---|---|---|---|
| coarse perception | 76.8% (43/56) | 76.8% (43/56) | +0.0 pp |
| fine-grained perception | 62.3% (33/53) | 64.2% (34/53) | -1.9 pp |
| instance reasoning | 83.7% (36/43) | 81.4% (35/43) | +2.3 pp |
| logical reasoning | 67.4% (31/46) | 69.6% (32/46) | -2.2 pp |
| math | 80.0% (40/50) | 80.0% (40/50) | +0.0 pp |
| science & technology | 50.0% (26/52) | 51.9% (27/52) | -1.9 pp |

## 延迟对比

| 指标 | Q8_0 | Q4_K_M |
|---|---|---|
| 平均 | 3.97s | 2.5s |
| P50 | 0.55s | 0.45s |
| P95 | 38.29s | 18.06s |
| 最大 | 40.95s | 26.28s |

## 分析

1. **Q8_0 vs Q4_K_M 准确率相当**: Q4_K_M 70.33% vs Q8_0 69.67%，差异仅 0.66pp，
   在 300 样本规模下属于统计噪声范围（95% 置信区间 ±~5pp）。

2. **Q8_0 延迟显著更高**: 平均 3.97s vs 2.5s（+59%），P95 更是 38s vs 18s。
   原因是 Q8_0 模型倾向于生成更长的推理链，部分数学题耗时 >38s 触及 max_tokens 上限，
   导致答案被截断（出现 `pred=?`），反而降低了准确率。

3. **Q4_K_M 是更优选择**: 在 GB10 上，Q4_K_M 仅占 ~5GB 显存，延迟更低，
   准确率与 Q8_0 相当。Q8_0 多占 ~3.5GB 显存却没有带来精度提升。

4. **量化损失集中在科技类**: 两种量化在 science & technology 类别上均仅 ~51%，
   远低于官方 bf16 水平，说明科学图表的表征在量化后损失较大。

5. **视觉编码器不是瓶颈**: 两者使用相同的 F16 视觉编码器，差异完全来自语言模型量化。

## 复现

```bash
# 启动评测用 llama-server
./llama.cpp-eval/build/bin/llama-server \
  --host 127.0.0.1 --port 9061 \
  --model ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-<QUANT>.gguf \
  --mmproj ./models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf \
  -ngl 99 --ctx-size 4096 --temp 0.1

# 运行评测
nohup python3 eval_vision.py --samples 300 --tag <QUANT> > eval_vision.log 2>&1 &
```

## 参考

- [MMStar Benchmark](https://mmstar-benchmark.github.io/)
- [MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o) — OpenBMB
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGML 推理引擎
