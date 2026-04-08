# GPU Model Bench — 测试方法论

跨平台标准化大模型评测方法论。同一套测试在不同算力平台上产出可对比的结果。

## 支持平台

| 平台 | GPU | 内存 | 带宽 | CUDA | 最大模型 (Q4) |
|------|-----|------|------|------|--------------|
| **DGX Spark** | GB10 | 128GB | 273 GB/s | 13.0 | ~102GB |
| Jetson AGX Orin 64 | Orin | 64GB | 204 GB/s | 12.6 | ~51GB |
| Jetson AGX Orin 32 | Orin | 32GB | 204 GB/s | 12.6 | ~25GB |
| Jetson Orin NX 16 | Orin NX | 16GB | 102 GB/s | 12.6 | ~12GB |
| Jetson Orin Nano 8 | Orin Nano | 8GB | 68 GB/s | 12.6 | ~6GB |
| Mac Studio M4 Max | M4 Max | 128GB | 546 GB/s | Metal | ~102GB |
| Mac Studio M4 Ultra | M4 Ultra | 192GB | 800 GB/s | Metal | ~153GB |
| RTX 4090 | RTX 4090 | 24GB | 1008 GB/s | 12.x | ~19GB |
| RTX 5090 | RTX 5090 | 32GB | 1792 GB/s | 12.8 | ~25GB |

平台通过 `--platform auto` 自动检测，或 `--platform <name>` 手动指定。
添加新平台只需在 `bench.py` 的 `KNOWN_PLATFORMS` 字典中添加一项。

## 测试引擎

| 引擎 | 安装方式 | 特点 | 适用场景 |
|------|---------|------|---------|
| **Ollama** | 预装 / 手动升级 | 开箱即用，GGUF 量化 | 快速评估、Q4_K_M/Q8_0 |
| **vLLM** | Docker / pip | OpenAI API，prefix caching | FP8 推理、TTFT 优化 |
| **llama.cpp** | Docker 编译 | 全平台支持 | 超大模型、边缘设备 |
| **SGLang** | Docker | LMSYS 出品 | 部分模型最优 |

## 测试套件

所有套件**跨平台完全一致** — 同样的 prompts、同样的指标、同样的评判标准。

### quick — 快速筛选 (1 prompt)

仅跑 hello prompt，用于量化版本对比、冷启动测试。

```bash
python3 bench/bench.py --models "模型名" --suite quick
```

### standard — 标准评测 (5 prompts)

| Prompt ID | 测试维度 | 内容 |
|-----------|---------|------|
| hello | 基础对话 | "Say hello in 3 languages." |
| reasoning | 逻辑推理 | "17只羊死了9只还剩几只" (答案: 9) |
| code | 代码生成 | 最长回文子串函数 |
| long_output | 长文生成 | 500 词科技文章 |
| chinese | 中文理解 | 用中文解释 MoE 架构 |

```bash
python3 bench/bench.py --models "模型名" --suite standard
```

`standard` 套件会基于启发式规则自动产出 `quality_score`，用于区分“速度快但回答质量弱”和“速度稍慢但结果更稳”的模型。

### toolcall — 工具调用 (2 prompts)

| Prompt ID | 测试内容 | 验证方式 |
|-----------|---------|---------|
| tool_weather | 单工具调用 (get_weather) | 检查 tool_calls 返回 |
| tool_multi | 多工具链式调用 (read_file → search) | 检查 tool_calls 返回 |

使用 Ollama/vLLM 原生 `tools` API 参数，非 prompt 注入。

```bash
python3 bench/bench.py --models "模型名" --suite toolcall
```

## 标准测试流程

```
Step 0: 平台检测
  └── bench.py --platform auto 自动识别 GPU/内存/CUDA
      或 --platform <name> 手动指定

Step 1: 环境准备
  ├── 确认 Ollama 版本 (ollama --version)
  ├── 确认目标模型已拉取 (ollama list)
  └── 如需 vLLM: 启动容器或安装环境

Step 2: 模型筛选
  ├── bench.py 自动过滤超出平台内存的模型
  ├── --list 查看兼容模型列表
  └── 选择目标模型 (或测所有已装)

Step 3: Quick 量化筛选
  ├── 同一模型不同量化版本跑 quick suite
  ├── 对比 tok/s 和内存占用
  └── 选出最佳量化版本

Step 4: Standard 深度评测
  ├── 选定的模型跑 standard suite
  ├── 记录 5 维度数据: tok/s, TTFT, wall_sec, output_tokens, 内容质量
  └── 同时跑一个已知基准模型作为对照组

Step 5: Toolcall 评测
  ├── 跑 toolcall suite
  └── 验证 tool_calls 返回正确性

Step 6: 结果汇总 (全自动)
  ├── CSV + JSON 原始数据 → results/{platform}/
  ├── Markdown 报告自动生成 (排行榜 + Key Findings + 架构对比)
  ├── 失败模型自动标记错误原因
  └── 跨平台结果可直接对比
```

## 核心指标

| 指标 | 含义 | 如何衡量 |
|------|------|---------|
| **tok/s** | 生成速率 | eval_count / eval_duration |
| **TTFT** | 首 token 延迟 | prompt_eval_duration (Ollama) / streaming 首 chunk (vLLM) |
| **wall_sec** | 端到端耗时 | 用户真实等待时间 |
| **output_tokens** | 输出 token 数 | 含 thinking tokens (Qwen) |
| **tok/s per GB** | 内存效率 | tok/s ÷ 模型大小 |
| **quality_score** | 启发式质量分 (0-1) | 按 prompt 规则自动打分 |
| **tool_call_correct** | 工具调用正确率 | API 返回的 tool_calls 结构是否正确 |

## 结果目录结构

```
bench/results/
├── dgx-spark/                     # DGX Spark (128GB) 数据
│   ├── bench_20260407_*.csv       #   原始数据 (CSV)
│   ├── bench_20260407_*.json      #   结构化数据 (JSON)
│   └── bench_20260407_*.md        #   自动生成报告 (排行榜 + 分析)
├── jetson-agx-orin-64/            # Jetson AGX Orin 64GB
│   └── bench_*.{csv,json,md}
├── mac-studio-m4-ultra/           # Mac Studio
│   └── bench_*.{csv,json,md}
└── custom-xxx/                    # 自动检测的自定义平台
    └── bench_*.{csv,json,md}
```

每条结果记录包含 `platform_name`, `platform_gpu`, `platform_memory_gb` 字段，
支持合并多平台数据进行横向对比。

### 自动报告内容

每次跑完自动生成 Markdown 报告，包含：

1. **平台概要** — GPU、内存、带宽、CUDA 版本
2. **排行榜** — 按 tok/s 排序，奖牌标记 (🥇🥈🥉)，含 tok/s per GB 效率指标
3. **综合排名** — 按平均质量优先、速度为辅进行模型排序
4. **Key Findings** — 自动识别最快模型、最低 TTFT、最高效率、最佳单项质量、MoE vs Dense 对比
5. **架构对比表** — MoE vs Dense 的 avg tok/s、TTFT、效率、质量
6. **失败模型列表** — 附错误原因 (如 Ollama 版本不兼容)
7. **原始数据引用** — 对应 CSV/JSON 文件名

## 关键发现 (跨平台通用经验)

1. **MoE 是带宽受限平台的甜点架构。** 内存带宽是瓶颈时，MoE 只激活少量参数 → 5-7 倍加速
2. **Q4_K_M 是最佳量化。** 更高精度反而更慢 (模型更大 → 带宽压力更大)
3. **tok/s ≠ 用户体感。** Thinking 模型 tok/s 高但包含不可见 thinking tokens
4. **TTFT 差异巨大。** 无 thinking 开销的模型首 token 更快
5. **统一内存需要 `--no-mmap`。** Spark/Orin/Mac 等统一内存平台使用 llama.cpp 必须禁用 mmap

## 添加新平台

在 `bench.py` 的 `KNOWN_PLATFORMS` 字典中添加:

```python
"my-platform": Platform(
    name="my-platform",
    gpu="GPU Name",
    gpu_arch="sm_xxx",
    memory_gb=64,
    bandwidth_gbps=200,
    cuda_version="12.x",
    cpu="CPU Name",
    os_info="OS arch",
),
```

或直接使用 `--platform auto`，系统会自动生成 custom profile。

## 文件说明

```
bench/
├── USAGE.md                        # 使用指南 / 实际操作流程
├── PLAN.md                         # 本文件：测试方法论
├── bench.py                        # 核心: 平台检测 + 标准化 benchmark
├── setup_models.sh                 # 模型下载和引擎环境搭建
├── ttft_compare.py                 # TTFT 对比测试 (Ollama vs vLLM)
├── start_vllm_fp8.sh               # vLLM FP8 容器启动脚本
├── RESULTS.md                      # DGX Spark Round 1 结果
├── RESULTS_R2.md                   # DGX Spark Round 2 结果
├── COMPARE_gemma4_26b_vs_qwen35.md # 深度对比报告
├── TEST_PLAN_R2.md                 # Round 2 计划 (留档)
└── results/                        # 原始数据 (按平台分目录)
    └── {platform}/                 #   CSV + JSON
```
