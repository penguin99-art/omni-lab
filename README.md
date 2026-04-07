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

# standard — 5 prompts (推理/代码/中文/长文), 完整评测
python3 bench/bench.py --models "qwen3.5:35b" --suite standard

# toolcall — 工具调用 (原生 tools API)
python3 bench/bench.py --models "qwen3.5:35b" --suite toolcall
```

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
    └── 6. 结果保存到 results/{platform}/ 目录
```

---

## DGX Spark 测试结果

### 速度排行榜 (Ollama Q4_K_M)

| 排名 | 模型 | 架构 | 激活参数 | 大小 | tok/s | TTFT | 工具调用 |
|------|------|------|----------|------|-------|------|---------|
| 1 | **qwen3.5:35b** | MoE | 3B | 23GB | **55.8** | 0.09s | ✓ |
| 2 | qwen3.5:9b | Dense | 9B | 6.6GB | 36.1 | 0.06s | - |
| 3 | gemma4:26b | MoE | 4B | 18GB | 35.1 | 0.07s | ✓ |
| 4 | gemma4:e2b | Dense PLE | 2B | 7.2GB | 33.9 | 0.03s | ✓ |
| 5 | gemma4:e4b | Dense PLE | 4B | 9.6GB | 25.9 | 0.05s | ✓ |
| 6 | qwen3.5:27b | Dense | 27B | 17GB | 11.3 | 0.24s | - |
| 7 | qwen3:32b | Dense | 32B | 20GB | 10.0 | 0.25s | - |
| 8 | gemma4:31b | Dense | 31B | 20GB | 8.1 | 0.17s | ✓ |

### 推荐配置 (DGX Spark 128GB)

| 场景 | 推荐模型 | tok/s | 内存 | 理由 |
|------|----------|-------|------|------|
| 日常助手 (速度优先) | qwen3.5:35b | 55.8 | 23GB | 绝对最快 |
| Agent / 工具调用 | gemma4:26b | 35.1 | 18GB | TTFT 快 2-3x |
| 轻量嵌入 / 并发 | gemma4:e2b | 33.9 | 7.2GB | 仅 7GB，效率最高 |
| 多模态 (视觉+音频) | gemma4:e4b / 26b | 26-35 | 10-18GB | Gemma4 原生支持 |
| 最大开源模型 | MiniMax M2.5 | ~26 | 101GB | 229B 总参 |

---

## 核心发现

1. **MoE 是带宽受限平台的甜点架构。** 内存带宽是瓶颈，MoE 只激活少量参数 → 比同级 Dense 模型快 **5-7 倍**
2. **Q4_K_M 是最佳量化。** 更高精度反而更慢 (模型更大 → 带宽压力更大)
3. **tok/s ≠ 用户体感。** Thinking 模型生成大量不可见 token，端到端未必最快
4. **Gemma 4 全系列支持工具调用。** 5 个模型全部通过原生 `tools` API 测试
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
| [测试方法论](bench/PLAN.md) | 测试套件、流程、指标定义、支持平台 |
| [Round 1: Qwen 系列基线](bench/RESULTS.md) | qwen3.5 全系列 (DGX Spark) |
| [Round 2: Gemma 4 vs Qwen](bench/RESULTS_R2.md) | 5 模型全维度对比 (DGX Spark) |
| [gemma4:26b vs qwen3.5:35b](bench/COMPARE_gemma4_26b_vs_qwen35.md) | 深度对比报告 |

## 项目结构

```
├── bench/                          # Benchmark 套件
│   ├── bench.py                    #   核心: 平台检测 + 标准化测试
│   ├── ttft_compare.py             #   TTFT 对比 (Ollama vs vLLM)
│   ├── setup_models.sh             #   模型下载 & 引擎环境搭建
│   ├── start_vllm_fp8.sh           #   vLLM FP8 容器启动
│   ├── PLAN.md                     #   测试方法论
│   └── results/                    #   原始数据 (按平台分目录)
│       ├── dgx-spark/              #     DGX Spark 数据
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
