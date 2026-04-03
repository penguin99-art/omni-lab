# Round 2: Gemma 4 & Qwen 3.5 严格对比测试

> 日期: 2026-04-03
> 硬件: DGX Spark (GB10, 128GB LPDDR5x, 273 GB/s, CUDA 13.0)
> 引擎: Ollama 0.20.0
> 基准线: qwen3.5:35b @ 57.4 tok/s (Round 1 冠军)

## 目标

1. **量化选型**: 在 Spark 128GB 上找到每个模型的最佳量化版本（速度/质量平衡点）
2. **横向对比**: Gemma 4 全系列 vs Qwen 3.5:35b 的速度、质量、工具调用能力
3. **产出**: 明确的模型推荐矩阵

## 测试模型矩阵

### 基准组 (已有数据)

| # | 模型 | 量化 | 大小 | 架构 | 激活参数 | 已知速度 |
|---|------|------|------|------|----------|----------|
| B1 | qwen3.5:35b | Q4_K_M | 23GB | MoE | 3B | 57.4 tok/s |

### Gemma 4 E2B 组 (2B Effective)

| # | 模型标签 | 量化 | 大小 | 架构 | 备注 |
|---|----------|------|------|------|------|
| G1 | gemma4:e2b | Q4_K_M | 7.2GB | Dense PLE | 默认量化 |
| G2 | gemma4:e2b-it-q8_0 | Q8_0 | 8.1GB | Dense PLE | 高精度 |
| G3 | gemma4:e2b-it-bf16 | BF16 | 10GB | Dense PLE | 全精度 |

### Gemma 4 E4B 组 (4B Effective)

| # | 模型标签 | 量化 | 大小 | 架构 | 备注 |
|---|----------|------|------|------|------|
| G4 | gemma4:e4b | Q4_K_M | 9.6GB | Dense PLE | 默认量化 |
| G5 | gemma4:e4b-it-q8_0 | Q8_0 | 12GB | Dense PLE | 高精度 |
| G6 | gemma4:e4b-it-bf16 | BF16 | 16GB | Dense PLE | 全精度 |

### Gemma 4 26B 组 (4B Active MoE)

| # | 模型标签 | 量化 | 大小 | 架构 | 激活参数 | 备注 |
|---|----------|------|------|------|----------|------|
| G7 | gemma4:26b | Q4_K_M | 18GB | MoE | 4B | 默认量化，与 qwen3.5:35b 直接对标 |
| G8 | gemma4:26b-a4b-it-q8_0 | Q8_0 | 28GB | MoE | 4B | 高精度 |

### Gemma 4 31B 组 (Dense)

| # | 模型标签 | 量化 | 大小 | 架构 | 备注 |
|---|----------|------|------|------|------|
| G9 | gemma4:31b | Q4_K_M | 20GB | Dense | 默认量化 |
| G10 | gemma4:31b-it-q8_0 | Q8_0 | 34GB | Dense | 高精度 |
| G11 | gemma4:31b-it-bf16 | BF16 | 63GB | Dense | 全精度（可能较慢） |

## 测试流程

```
Phase 1: 环境准备
  ├── 1.1 升级 Ollama 到 0.20.0 (gemma4 依赖)
  ├── 1.2 验证升级成功
  └── 1.3 拉取所有待测模型

Phase 2: 量化筛选（quick suite）
  ├── 2.1 每个模型跑 quick suite (hello prompt)
  ├── 2.2 记录 tok/s + TTFT + 内存占用
  ├── 2.3 同模型不同量化的速度/大小对比
  └── 2.4 选出每个模型系列的最佳量化

Phase 3: 深度评测（standard suite）
  ├── 3.1 筛选后的模型跑 standard suite
  │     ├── hello (多语言)
  │     ├── reasoning (逻辑推理: 17只羊)
  │     ├── code (代码: 最长回文子串)
  │     ├── long_output (长文: 500词作文)
  │     └── chinese (中文理解: MoE解释)
  └── 3.2 qwen3.5:35b 重新跑一轮作为对照

Phase 4: 工具调用评测（toolcall suite）
  ├── 4.1 单工具调用 (get_weather)
  ├── 4.2 多工具链式调用 (read_file + search_text)
  └── 4.3 记录调用正确率

Phase 5: 结果汇总
  ├── 5.1 速度排行榜
  ├── 5.2 质量评分（人工 + 自动）
  ├── 5.3 量化影响分析
  └── 5.4 最终推荐
```

## 评估维度

### 速度指标
- **tok/s**: 生成 token 速率（核心指标）
- **TTFT**: 首 token 延迟
- **wall_sec**: 端到端耗时

### 质量指标
- **推理正确性**: 17只羊问题，答案是否为 9
- **代码质量**: 回文子串函数是否正确、有类型注解
- **中文流畅度**: MoE 解释是否清晰完整
- **长文连贯性**: 500词作文结构是否合理

### 效率指标
- **内存占用**: 模型加载后 + 推理峰值
- **速度/大小比**: tok/s per GB (越高越好)
- **量化损失率**: Q8 vs BF16 vs Q4 的质量差异

## 预期结论框架

| 使用场景 | 候选模型 | 评判标准 |
|----------|----------|----------|
| 日常对话（速度优先） | gemma4:26b vs qwen3.5:35b | tok/s 最高且质量可接受 |
| 代码助手 | gemma4:e4b-bf16 vs qwen3.5:35b | 代码质量 + 速度 |
| 轻量嵌入 | gemma4:e2b vs gemma4:e4b | 最小内存、可并发运行 |
| 高质量推理 | gemma4:31b-q8 vs qwen3.5:35b | 推理正确率 + 中文能力 |
| Agent/工具调用 | gemma4 tools vs qwen3.5:35b | 工具调用准确率 |

## 文件产出

```
bench/
├── TEST_PLAN_R2.md              # 本文件
├── bench.py                     # 已更新：新增 Gemma 4 模型
├── results/
│   ├── bench_*_gemma4_quant.csv # Phase 2: 量化筛选结果
│   ├── bench_*_gemma4_full.csv  # Phase 3: 深度评测结果
│   ├── bench_*_gemma4_tools.csv # Phase 4: 工具调用结果
│   └── *.json                   # 对应的 JSON 格式
└── RESULTS_R2.md                # Phase 5: 汇总分析报告
```
