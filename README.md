# DGX Spark Model Lab

NVIDIA DGX Spark (GB10, 128GB) 上的大模型系统性测试与优化实验室。

## 硬件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GB10 (Blackwell, SM 12.1) |
| 内存 | 128GB LPDDR5x 统一内存 (~273 GB/s) |
| 算力 | 1 PFLOP (FP4) |
| CUDA | 13.0 |
| 磁盘 | 3.7TB NVMe |
| OS | Ubuntu 24.04 (DGX OS, aarch64) |

---

## 快速开始

### 1. 查看所有模型

```bash
python3 bench/bench.py --list
```

### 2. 测试指定模型

```bash
# 快速测试 (1 prompt, 30秒)
python3 bench/bench.py --models "qwen3.5:35b" --suite quick

# 标准测试 (5 prompts, 涵盖推理/代码/中文/长文)
python3 bench/bench.py --models "qwen3.5:35b" --suite standard

# 工具调用测试 (使用原生 tools API)
python3 bench/bench.py --models "qwen3.5:35b" --suite toolcall

# 一次测试多个模型
python3 bench/bench.py --models "qwen3.5:35b,gemma4:26b" --suite standard

# 测试所有已安装的 Ollama 模型
python3 bench/bench.py --engines ollama --suite quick
```

### 3. TTFT 对比测试 (Ollama vs vLLM)

```bash
# 先启动 vLLM 容器
bash bench/start_vllm_fp8.sh

# 跑对比
python3 bench/ttft_compare.py --engine both
python3 bench/ttft_compare.py --engine ollama   # 仅 Ollama
python3 bench/ttft_compare.py --engine vllm     # 仅 vLLM
```

### 4. 环境搭建 (首次使用)

```bash
# 下载 Ollama 模型 + 配置 vLLM 环境
bash bench/setup_models.sh
```

---

## 测试结果

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

### 效率排行榜 (tok/s per GB)

| 排名 | 模型 | tok/s | 大小 | tok/s/GB |
|------|------|-------|------|----------|
| 1 | qwen3.5:9b | 36.1 | 6.6GB | 5.47 |
| 2 | **gemma4:e2b** | 33.9 | 7.2GB | **4.71** |
| 3 | gemma4:e4b | 25.9 | 9.6GB | 2.70 |
| 4 | qwen3.5:35b | 55.8 | 23GB | 2.43 |
| 5 | gemma4:26b | 35.1 | 18GB | 1.95 |

### 推荐配置

| 场景 | 推荐模型 | tok/s | 内存 | 理由 |
|------|----------|-------|------|------|
| 日常助手 (速度优先) | qwen3.5:35b | 55.8 | 23GB | 绝对最快 |
| Agent / 工具调用 | gemma4:26b | 35.1 | 18GB | TTFT 快 2-3x, 原生工具调用 |
| 轻量嵌入 / 并发 | gemma4:e2b | 33.9 | 7.2GB | 仅 7GB，效率最高 |
| 多模态 (视觉+音频) | gemma4:e4b / 26b | 26-35 | 10-18GB | Gemma4 原生支持 |
| 最大开源模型 | MiniMax M2.5 | ~26 | 101GB | 229B 总参, Q3_K_XL |

---

## 核心发现

1. **MoE 是 Spark 的甜点架构。** 内存带宽 273 GB/s 是瓶颈，MoE 只激活少量参数 → 比同级 Dense 模型快 **5-7 倍**
2. **Q4_K_M 是最佳量化。** 更高精度 (Q8/BF16) 增大模型，加剧带宽瓶颈，反而更慢
3. **tok/s ≠ 用户体感。** qwen3.5 的 thinking 模式生成大量不可见 token，端到端 gemma4 可能更快
4. **Gemma 4 全系列支持工具调用。** 5 个模型 (含 2B 级别 e2b) 全部通过原生 `tools` API 测试
5. **Spark 能跑 229B 模型。** MiniMax M2.5 使用 Unsloth Q3_K_XL (101GB) + llama.cpp 可在 Spark 上运行

---

## 详细报告

| 报告 | 内容 |
|------|------|
| [测试方法论](bench/PLAN.md) | 测试套件、流程、指标定义、模型矩阵 |
| [Round 1: Qwen 系列基线](bench/RESULTS.md) | qwen3.5 全系列 tok/s / TTFT / 质量评测 |
| [Round 2: Gemma 4 vs Qwen](bench/RESULTS_R2.md) | 5 模型全维度对比 + 量化建议 |
| [gemma4:26b vs qwen3.5:35b 深度对比](bench/COMPARE_gemma4_26b_vs_qwen35.md) | 速度 / 质量 / 效率 / 场景推荐 |

---

## 项目结构

```
├── bench/                          # Benchmark 套件
│   ├── bench.py                    #   自动化测试脚本 (Ollama / vLLM / SGLang)
│   ├── ttft_compare.py             #   TTFT 对比测试 (Ollama vs vLLM prefix caching)
│   ├── setup_models.sh             #   模型下载 & 引擎环境搭建
│   ├── start_vllm_fp8.sh           #   vLLM FP8 容器启动脚本
│   ├── PLAN.md                     #   测试方法论 (套件/流程/指标)
│   ├── RESULTS.md                  #   Round 1 结果
│   ├── RESULTS_R2.md               #   Round 2 结果
│   ├── COMPARE_gemma4_26b_vs_qwen35.md  #   深度对比报告
│   ├── TEST_PLAN_R2.md             #   Round 2 测试计划 (已完成, 留档)
│   └── results/                    #   原始数据 (CSV + JSON)
│
├── cases/                          # 过往实验 & Demo
│   ├── edge-agent/                 #   语音 AI Agent 框架
│   ├── ai-agent/                   #   极简工作站 Agent
│   └── notes/                      #   思考笔记
│
├── models/                         # 模型权重 (本地保留, 不提交)
└── llama.cpp-omni/                 # MiniCPM-o (git submodule)
```

## 待测清单

| 模型 | 引擎 | 社区预期 | 关注点 |
|------|------|----------|--------|
| nemotron-cascade-2 | Ollama | ~72 tok/s | MoE 3B, 可能是 Spark 最快 |
| gpt-oss:120b (MXFP4) | vLLM+patch | ~81 tok/s | 全场吞吐最高 |
| Qwen3.5-35B-FP8 | vLLM Docker | ~47 tok/s | prefix caching TTFT 优化 |
| MiniMax M2.5 (Q3_K_XL) | llama.cpp | ~26 tok/s | 229B, Spark 能跑的最大模型 |
| Gemma 4 Q8/BF16 | Ollama 0.20 | TBD | 量化精度 vs 速度对比 |

## 参考资料

- [DGX Spark 多模型横评 (karaage0703)](https://zenn.dev/karaage0703/articles/fcca40c614dffd) — Ollama/vLLM/SGLang 全引擎对比
- [Qwen3.5-35B on DGX Spark (adadrag)](https://github.com/adadrag/qwen3.5-dgx-spark) — vLLM 部署指南
- [vLLM SM121 patches (namake-taro)](https://github.com/namake-taro/vllm-custom) — MXFP4 + SM121 内核修复
- [MiniMax M2.5 on DGX Spark (re-cinq)](https://github.com/re-cinq/minimax-m2.5-nvidia-dgx) — 229B 模型 Spark 部署
- [Ollama DGX Spark Performance](https://registry.ollama.ai/blog/nvidia-spark-performance) — 官方性能数据
