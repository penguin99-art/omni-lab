# GPU Model Bench

跨平台的大模型标准化评测工具。自动检测硬件平台，按内存过滤兼容模型，生成可对比的标准化结果。

支持 DGX Spark、Jetson Orin、Mac Studio、RTX 桌面等任意算力平台。

---

## 快速开始

### 在任意新机器上

```bash
# 1. 自动检测硬件，查看兼容模型
python3 bench/bench.py --list

# 2. 一键跑所有已装模型
python3 bench/bench.py --engines ollama --suite standard

# 3. 测试指定模型
python3 bench/bench.py --models "qwen3.5:35b" --suite quick
```

### 指定平台

```bash
# 手动指定平台 (跳过自动检测)
python3 bench/bench.py --platform dgx-spark --list
python3 bench/bench.py --platform jetson-agx-orin-64 --list

# 已知平台: dgx-spark, jetson-agx-orin-64, jetson-agx-orin-32,
#           jetson-orin-nx-16, jetson-orin-nano-8,
#           mac-studio-m4-max, mac-studio-m4-ultra, rtx-4090, rtx-5090
```

### 测试套件

```bash
# quick    — 1 prompt, 30秒, 快速筛选
python3 bench/bench.py --models "qwen3.5:35b" --suite quick

# standard — 5 prompts (推理/代码/中文/长文), 完整评测 + 自动质量评分
python3 bench/bench.py --models "qwen3.5:35b" --suite standard

# toolcall — 工具调用 (原生 tools API)
python3 bench/bench.py --models "qwen3.5:35b" --suite toolcall
```

`standard` 报告会为每个 prompt 自动生成 `quality_score`，并汇总为模型综合排名。

### TTFT 对比测试

```bash
bash bench/start_vllm_fp8.sh                     # 启动 vLLM
python3 bench/ttft_compare.py --engine both       # Ollama vs vLLM
```

---

## 工作原理

```
bench.py --platform auto
    │
    ├── 1. 自动检测硬件 (GPU型号, 内存, CUDA版本)
    ├── 2. 匹配已知平台 profile 或生成 custom profile
    ├── 3. 按内存自动过滤不兼容模型 (80% headroom)
    ├── 4. 检查哪些模型已安装可用
    ├── 5. 跑标准化测试 (相同 prompts, 相同指标)
    └── 6. 自动生成报告 → results/{platform}/
        ├── bench_YYYYMMDD_HHMMSS.csv    # 原始数据
        ├── bench_YYYYMMDD_HHMMSS.json   # 结构化数据
        └── bench_YYYYMMDD_HHMMSS.md     # 自动报告 (速度榜 + 质量榜 + 分析)
```

---

## DGX Spark 测试结果

### 速度排行榜 (Ollama Q4_K_M, 2026-04-07)

| 排名 | 模型 | 架构 | 激活参数 | 大小 | tok/s | TTFT | tok/s/GB |
|------|------|------|----------|------|-------|------|----------|
| 1 | **gemma4:e2b** | Dense | 2B | 7GB | **109.8** | 0.02s | 15.2 |
| 2 | gpt-oss:20b | MoE | all | 13GB | 61.1 | 6.34s | 4.7 |
| 3 | gemma4:26b | MoE | 4B | 18GB | 59.9 | 0.07s | 3.3 |
| 4 | **qwen3.5:35b** | MoE | 3B | 23GB | **58.1** | 0.10s | 2.5 |
| 5 | gemma4:e4b | Dense | 4B | 10GB | 56.5 | 0.03s | 5.9 |
| 6 | qwen3.5:9b | Dense | 9B | 7GB | 34.2 | 0.06s | 5.2 |
| 7 | qwen3.5:122b-a10b | MoE | 10B | 81GB | 23.8 | 0.18s | 0.3 |
| 8 | qwen3.5:27b | Dense | 27B | 17GB | 11.5 | 0.15s | 0.7 |
| 9 | gemma4:31b | Dense | 31B | 20GB | 10.0 | 0.13s | 0.5 |
| 10 | qwen3:32b | Dense | 32B | 20GB | 9.8 | 0.15s | 0.5 |

### 推荐配置 (DGX Spark 128GB)

| 场景 | 推荐模型 | tok/s | 内存 | 理由 |
|------|----------|-------|------|------|
| 速度+效率冠军 | gemma4:e2b | 109.8 | 7GB | 15.2 tok/s/GB，最高效率 |
| 日常助手 (质量优先) | qwen3.5:35b | 58.1 | 23GB | MoE 高速 + 长输出 |
| Agent / 工具调用 | gemma4:26b | 59.9 | 18GB | TTFT 0.07s，MoE 高效 |
| 轻量嵌入 / 并发 | gemma4:e2b | 109.8 | 7GB | 仅 7GB，可多实例并行 |
| 多模态 (视觉+音频) | gemma4:e4b / 26b | 57-60 | 10-18GB | Gemma4 原生支持 |
| 最大开源模型 | qwen3.5:122b-a10b | 23.8 | 81GB | 1220 亿参数 MoE |

---

## 核心发现

1. **gemma4:e2b 效率惊人。** 109.8 tok/s，仅 7GB，效率达 15.2 tok/s/GB — 所有模型中最高
2. **MoE 是带宽受限平台的甜点架构。** MoE 均值 50.7 tok/s vs Dense 38.6 tok/s，领先 31%
3. **Q4_K_M 是最佳量化。** 更高精度反而更慢 (模型更大 → 带宽压力更大)
4. **Ollama 版本很重要。** Gemma 4 需要 Ollama 0.20+，旧版本会静默返回空结果
5. **统一内存架构需要 `--no-mmap`。** Spark/Orin 等统一内存平台使用 llama.cpp 必须禁用 mmap

---

## 添加新平台

### 自动模式 (推荐)

直接跑，自动检测：

```bash
python3 bench/bench.py --list          # 看检测到什么平台
python3 bench/bench.py --suite quick   # 跑起来
```

### 注册新平台 profile

在 `bench/bench.py` 的 `KNOWN_PLATFORMS` 字典中添加：

```python
"my-new-platform": Platform(
    name="my-new-platform",
    gpu="GPU Name",
    gpu_arch="sm_xxx",
    memory_gb=64,
    bandwidth_gbps=200,
    cuda_version="12.x",
    cpu="CPU Name",
    os_info="OS aarch64",
),
```

然后用 `--platform my-new-platform` 指定。

---

## 详细报告

| 报告 | 内容 |
|------|------|
| [使用指南](bench/USAGE.md) | 从 0 到 1 跑一遍 benchmark 的操作手册 |
| [测试方法论](bench/PLAN.md) | 测试套件、流程、指标定义、支持平台 |
| [Round 1: Qwen 系列基线](bench/RESULTS.md) | qwen3.5 全系列 (DGX Spark) |
| [Round 2: Gemma 4 vs Qwen](bench/RESULTS_R2.md) | 5 模型全维度对比 (DGX Spark) |
| [gemma4:26b vs qwen3.5:35b](bench/COMPARE_gemma4_26b_vs_qwen35.md) | 深度对比报告 |

### 自动报告内容

每次 benchmark 跑完会自动生成 Markdown 报告，包含：

1. 总体综合排名（平均质量优先，速度为辅）
2. 每个 prompt 的速度/质量排行榜
3. Key Findings（最快、最低 TTFT、最高效率、最佳单项质量）
4. MoE vs Dense 对比
5. 失败模型与错误原因

## 项目结构

```
├── bench/                          # Benchmark 套件
│   ├── bench.py                    #   核心: 平台检测 + 标准化测试
│   ├── ttft_compare.py             #   TTFT 对比 (Ollama vs vLLM)
│   ├── setup_models.sh             #   模型下载 & 引擎环境搭建
│   ├── start_vllm_fp8.sh           #   vLLM FP8 容器启动
│   ├── PLAN.md                     #   测试方法论
│   ├── USAGE.md                    #   使用指南 / 实战流程
│   └── results/                    #   原始数据 + 自动报告 (按平台分目录)
│       ├── dgx-spark/              #     DGX Spark (CSV + JSON + MD 报告)
│       └── {platform}/             #     其他平台数据
│
├── cases/                          # 过往实验
├── models/                         # 模型权重 (本地, 不提交)
└── llama.cpp-omni/                 # git submodule
```

## 参考资料

- [DGX Spark 多模型横评 (karaage0703)](https://zenn.dev/karaage0703/articles/fcca40c614dffd)
- [MiniMax M2.5 on DGX Spark (re-cinq)](https://github.com/re-cinq/minimax-m2.5-nvidia-dgx)
- [Ollama DGX Spark Performance](https://registry.ollama.ai/blog/nvidia-spark-performance)
- [vLLM SM121 patches (namake-taro)](https://github.com/namake-taro/vllm-custom)
