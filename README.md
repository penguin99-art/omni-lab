# DGX Spark Model Lab

NVIDIA DGX Spark (GB10, 128GB) 上的大模型系统性测试与优化实验室。

## 硬件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GB10 (Blackwell, SM 12.1) |
| 内存 | 128GB LPDDR5x 统一内存 |
| 带宽 | ~273 GB/s |
| 算力 | 1 PFLOP (FP4) |
| CUDA | 13.0 |
| 磁盘 | 3.7TB NVMe |
| OS | Ubuntu 24.04 (DGX OS) |

## 快速开始

```bash
# 查看所有可测模型及可用状态
python bench/bench.py --list

# 测试指定模型
python bench/bench.py --models "qwen3.5:35b" --suite standard

# 测试所有已安装的 Ollama 模型
python bench/bench.py --engines ollama --suite quick

# 跑 tool calling 测试
python bench/bench.py --models "qwen3.5:35b" --suite toolcall
```

## 项目结构

```
├── bench/                 # [主体] Benchmark 套件
│   ├── bench.py           #   自动化测试脚本 (Ollama / vLLM / SGLang)
│   ├── setup_models.sh    #   模型下载 & 环境搭建
│   ├── PLAN.md            #   测试方案与模型矩阵
│   ├── RESULTS.md         #   Round 1 结果 (Qwen 系列)
│   ├── TEST_PLAN_R2.md    #   Round 2 测试计划 (Gemma 4)
│   ├── RESULTS_R2.md      #   Round 2 结果 (Gemma 4 vs Qwen)
│   └── results/           #   原始数据 (CSV + JSON)
│
├── models/                # 模型权重 (gitignored)
│
├── cases/                 # 过往实验 & Demo
│   ├── edge-agent/        #   语音 AI Agent 框架 (edge_agent)
│   ├── ai-agent/          #   极简工作站 Agent
│   ├── llama-cpp-eval/    #   llama.cpp 评测
│   ├── ref/               #   参考代码 (claw-code 等)
│   └── notes/             #   思考笔记
│
└── llama.cpp-omni         # MiniCPM-o (git submodule)
```

## 测试引擎

| 引擎 | 安装方式 | 特点 |
|------|---------|------|
| **Ollama** | 预装 | 开箱即用，模型库丰富 |
| **vLLM** (cu130-nightly) | Docker | OpenAI 兼容 API，支持 MXFP4/NVFP4 量化 |
| **vLLM** + namake-taro | pip + patch | SM12.1 优化内核，MXFP4 最佳性能 |
| **SGLang** (spark) | Docker | LMSYS 出品，部分模型最优 |

## 已测结果摘要

### Round 2: Gemma 4 vs Qwen 3.5 (2026-04-03)

| 模型 | 架构 | 激活参数 | 大小 | tok/s | TTFT | 工具调用 | 备注 |
|------|------|----------|------|-------|------|---------|------|
| **qwen3.5:35b** | MoE | 3B | 23GB | **55.8** | 0.09s | ✓ | 速度冠军 |
| gemma4:26b | MoE | 4B | 18GB | 35.1 | 0.07s | ✓ | Gemma 最快，原生工具调用 |
| gemma4:e2b | Dense PLE | 2B | 7.2GB | 33.9 | 0.03s | ✓ | 最轻量，效率最高 |
| gemma4:e4b | Dense PLE | 4B | 9.6GB | 25.9 | 0.05s | ✓ | 支持视觉/音频 |
| gemma4:31b | Dense | 31B | 20GB | 8.1 | 0.17s | ✓ | 参数最大但太慢 |

### Round 1: Qwen 系列基线 (2026-04-02)

| 模型 | 架构 | 大小 | tok/s | TTFT | 备注 |
|------|------|------|-------|------|------|
| **qwen3.5:35b** | MoE 3B | 23GB | **57.4** | 0.09s | 最佳日用 |
| qwen3.5:9b | Dense | 6.6GB | 36.1 | 0.06s | 轻量快速 |
| qwen3.5:27b | Dense | 17GB | 11.3 | 0.24s | 交互慢 |
| qwen3:32b | Dense | 20GB | 10.0 | 0.25s | 交互慢 |

详见 [Round 2 报告](bench/RESULTS_R2.md) | [Round 1 报告](bench/RESULTS.md)

## 核心发现

1. **MoE 是 DGX Spark 的甜点架构。** 内存带宽 (273 GB/s) 是主要瓶颈，MoE 只激活少量参数，速度比同级 Dense 模型快 **5-7 倍**
2. **Q4_K_M 是最佳量化。** 更高精度 (Q8/BF16) 增大模型体积，加剧带宽瓶颈，反而更慢
3. **Gemma 4 全系列支持工具调用。** 5 个模型（含 2B 级别的 e2b）全部通过单工具 + 多工具链式调用测试
4. **gemma4:e2b 效率惊人。** 仅 7.2GB 跑出 33.9 tok/s，tok/s/GB = 4.71，适合轻量并发

## 推荐配置

| 场景 | 推荐模型 | tok/s | 内存 |
|------|----------|-------|------|
| 日常助手（速度优先） | qwen3.5:35b | 55.8 | 23GB |
| 轻量嵌入/并发 | gemma4:e2b | 33.9 | 7.2GB |
| Agent/工具调用 | gemma4:26b | 35.1 | 18GB |
| 多模态（视觉+音频） | gemma4:e4b / 26b | 26-35 | 10-18GB |

## 待测清单

| 模型 | 引擎 | 社区预期 | 关注点 |
|------|------|----------|--------|
| nemotron-cascade-2 | Ollama | ~72 tok/s | 速度+质量最佳比 |
| gpt-oss:120b | Ollama | ~41 tok/s | 工具调用精度高 |
| Qwen3.5-35B MXFP4 | vLLM+patch | ~60 tok/s | 量化 vs Ollama 对比 |
| gpt-oss-120b MXFP4 | vLLM+patch | ~81 tok/s | 全场最快大模型 |
| gemma4 Q8/BF16 | Ollama 0.20 | TBD | 量化精度 vs 速度对比 |

## 参考资料

- [DGX Spark 多模型横评 (karaage0703)](https://zenn.dev/karaage0703/articles/fcca40c614dffd) — Ollama/vLLM/SGLang 全引擎对比
- [Qwen3.5-35B on DGX Spark (adadrag)](https://github.com/adadrag/qwen3.5-dgx-spark) — vLLM 部署详细指南
- [vLLM SM121 patches (namake-taro)](https://github.com/namake-taro/vllm-custom) — MXFP4 + SM121 内核修复
- [122B NVFP4 Docker (JungkwanBan)](https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4) — Spark 专用 Docker 构建
