# DGX Spark 大模型测试方法论

## 硬件环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GB10 (SM 12.1 Blackwell) |
| 统一内存 | 128GB LPDDR5x (~273 GB/s) |
| 算力峰值 | 1 PFLOP (FP4) |
| CUDA | 13.0 |
| 磁盘 | 3.7TB NVMe |
| OS | Ubuntu 24.04 (DGX OS, aarch64) |

## 测试引擎

| 引擎 | 安装方式 | 特点 | 适用场景 |
|------|---------|------|---------|
| **Ollama** | 预装 / 手动升级 | 开箱即用，GGUF 量化 | 快速评估、Q4_K_M/Q8_0 |
| **vLLM** (cu130-nightly) | Docker | OpenAI API，prefix caching | FP8 推理、TTFT 优化 |
| **vLLM** + namake-taro | pip + patch | MXFP4/NVFP4 量化 | 极限性能 |
| **llama.cpp** | Docker 编译 | SM 12.1 原生支持 | 超大模型 (200B+) |
| **SGLang** (spark) | Docker | LMSYS 出品 | 部分模型最优 |

## 测试套件

### quick — 快速筛选 (1 prompt)

仅跑 hello prompt，用于量化版本对比、冷启动测试。

```bash
python3 bench/bench.py --models "模型名" --suite quick
```

### standard — 标准评测 (5 prompts)

| Prompt ID | 测试维度 | 内容 |
|-----------|---------|------|
| hello | 基础对话 | "Hello, who are you?" |
| reasoning | 逻辑推理 | "17只羊死了9只还剩几只" (答案: 9) |
| code | 代码生成 | 最长回文子串函数 |
| long_output | 长文生成 | 500 词科技文章 |
| chinese | 中文理解 | 用中文解释 MoE 架构 |

```bash
python3 bench/bench.py --models "模型名" --suite standard
```

### toolcall — 工具调用 (2 prompts)

| Prompt ID | 测试内容 | 验证方式 |
|-----------|---------|---------|
| tool_weather | 单工具调用 (get_weather) | 检查 tool_calls 返回 |
| tool_multi | 多工具链式调用 (read_file → search) | 检查 tool_calls 返回 |

使用 Ollama/vLLM 原生 `tools` API 参数，非 prompt 注入。

```bash
python3 bench/bench.py --models "模型名" --suite toolcall
```

## 测试流程

```
Step 1: 环境准备
  ├── 确认 Ollama 版本 (ollama --version)
  ├── 确认目标模型已拉取 (ollama list)
  └── 如需 vLLM: 启动容器或安装环境

Step 2: Quick 筛选
  ├── 同一模型不同量化版本跑 quick suite
  ├── 对比 tok/s 和内存占用
  └── 选出最佳量化版本

Step 3: Standard 深度评测
  ├── 选定的模型跑 standard suite
  ├── 记录 5 维度数据: tok/s, TTFT, wall_sec, output_tokens, 内容质量
  └── 同时跑一个已知基准模型作为对照组

Step 4: Toolcall 评测
  ├── 跑 toolcall suite
  └── 验证 tool_calls 返回正确性

Step 5: 结果汇总
  ├── 数据自动保存到 bench/results/ (CSV + JSON)
  ├── 生成排行榜和对比分析
  └── 更新结果文档
```

## 核心指标

| 指标 | 含义 | 如何衡量 |
|------|------|---------|
| **tok/s** | 生成速率 | eval_count / eval_duration |
| **TTFT** | 首 token 延迟 | prompt_eval_duration (Ollama) / streaming 首 chunk (vLLM) |
| **wall_sec** | 端到端耗时 | 用户真实等待时间 |
| **output_tokens** | 输出 token 数 | 含 thinking tokens (Qwen) |
| **tok/s per GB** | 内存效率 | tok/s ÷ 模型大小 |
| **tool_call_correct** | 工具调用正确率 | API 返回的 tool_calls 结构是否正确 |

## 关键发现 (经验法则)

1. **MoE 是 Spark 的甜点架构。** 内存带宽 273 GB/s 是瓶颈，MoE 只激活少量参数 → 5-7 倍加速
2. **Q4_K_M 是最佳量化。** 更高精度反而更慢 (模型更大 → 带宽压力更大)
3. **tok/s ≠ 用户体感。** Thinking 模型 (Qwen3.5) tok/s 高但包含不可见 thinking tokens，端到端可能更慢
4. **TTFT 差异巨大。** Gemma4 无 thinking 开销 → TTFT 比 Qwen3.5 快 2-3 倍
5. **统一内存需要 `--no-mmap`。** llama.cpp 在 Spark 上必须禁用 mmap 否则性能崩溃

## 模型矩阵

### 已验证 (Ollama Q4_K_M)

| 模型 | 总参数 | 激活参数 | 大小 | 架构 | tok/s | 工具调用 |
|------|--------|----------|------|------|-------|---------|
| qwen3.5:35b | 35B | 3B | 23GB | MoE | 55.8 | ✓ |
| gemma4:26b | 26B | 4B | 18GB | MoE | 35.1 | ✓ |
| gemma4:e2b | ~2B | 2B | 7.2GB | Dense PLE | 33.9 | ✓ |
| gemma4:e4b | ~4B | 4B | 9.6GB | Dense PLE | 25.9 | ✓ |
| gemma4:31b | 31B | 31B | 20GB | Dense | 8.1 | ✓ |
| qwen3.5:9b | 9B | 9B | 6.6GB | Dense | 36.1 | - |
| qwen3.5:27b | 27B | 27B | 17GB | Dense | 11.3 | - |
| qwen3:32b | 32B | 32B | 20GB | Dense | 10.0 | - |

### 待验证

| 模型 | 引擎 | 社区预期 | 关注点 |
|------|------|----------|--------|
| nemotron-cascade-2 | Ollama | ~72 tok/s | MoE 3B, 可能是最快模型 |
| gpt-oss:120b | Ollama/vLLM | 41-81 tok/s | 工具调用精度高 |
| Qwen3.5-35B-FP8 | vLLM Docker | ~47 tok/s | prefix caching → TTFT 优化 |
| MiniMax M2.5 | llama.cpp | ~26 tok/s | 229B 总参数, Spark 上最大模型 |

## 文件说明

```
bench/
├── PLAN.md                        # 本文件：测试方法论
├── bench.py                       # 自动化 benchmark 脚本
├── setup_models.sh                # 模型下载和引擎环境搭建
├── ttft_compare.py                # TTFT 对比测试 (Ollama vs vLLM)
├── start_vllm_fp8.sh              # vLLM FP8 容器启动脚本
├── RESULTS.md                     # Round 1 结果 (Qwen 系列)
├── RESULTS_R2.md                  # Round 2 结果 (Gemma 4 vs Qwen)
├── COMPARE_gemma4_26b_vs_qwen35.md # gemma4:26b vs qwen3.5:35b 深度对比
├── TEST_PLAN_R2.md                # Round 2 测试计划 (已完成, 留档)
└── results/                       # 原始数据 (CSV + JSON, 自动生成)
```
