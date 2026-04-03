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
│   ├── RESULTS.md         #   测试结果分析
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

| 模型 | 架构 | 大小 | tok/s | TTFT | 备注 |
|------|------|------|-------|------|------|
| **qwen3.5:35b** | MoE 3B active | 23GB | **57.4** | 0.09s | 当前最佳日用 |
| qwen3.5:9b | Dense | 6.6GB | 36.1 | 0.06s | 轻量快速 |
| qwen3.5:27b | Dense | 17GB | 11.3 | 0.24s | 交互慢 |
| qwen3:32b | Dense | 20GB | 10.0 | 0.25s | 交互慢 |

详见 [bench/RESULTS.md](bench/RESULTS.md)。

## 核心发现

**MoE 架构是 DGX Spark 的甜点。** 内存带宽 (273 GB/s) 是主要瓶颈，MoE 只激活少量参数 (3B)，速度比同级 Dense 模型 (27-32B) 快 **5-6 倍**。

## 待测清单

| 模型 | 引擎 | 社区预期 | 关注点 |
|------|------|----------|--------|
| nemotron-cascade-2 | Ollama | ~72 tok/s | 速度+质量最佳比 |
| gpt-oss:120b | Ollama | ~41 tok/s | 工具调用精度高 |
| Qwen3.5-35B MXFP4 | vLLM+patch | ~60 tok/s | 量化 vs Ollama 对比 |
| gpt-oss-120b MXFP4 | vLLM+patch | ~81 tok/s | 全场最快大模型 |
| Qwen3.5-27B-FP8 | vLLM Docker | ~6 tok/s | 工具调用精度最高 |
| Qwen3.5-122B NVFP4 | JungkwanBan Docker | TBD | 百亿参数 + FP4 |

## 参考资料

- [DGX Spark 多模型横评 (karaage0703)](https://zenn.dev/karaage0703/articles/fcca40c614dffd) — Ollama/vLLM/SGLang 全引擎对比
- [Qwen3.5-35B on DGX Spark (adadrag)](https://github.com/adadrag/qwen3.5-dgx-spark) — vLLM 部署详细指南
- [vLLM SM121 patches (namake-taro)](https://github.com/namake-taro/vllm-custom) — MXFP4 + SM121 内核修复
- [122B NVFP4 Docker (JungkwanBan)](https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4) — Spark 专用 Docker 构建
