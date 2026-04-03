# DGX Spark 大模型系统性测试方案

## 硬件环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GB10 (SM 12.1 Blackwell) |
| 统一内存 | 128GB LPDDR5x |
| 内存带宽 | ~273 GB/s |
| CUDA | 13.0 |
| 算力峰值 | 1 PFLOP (FP4) |
| 磁盘 | 3.7TB (3.1TB 可用) |
| OS | Ubuntu 24.04 |

## 待测模型矩阵

### Tier 1: Ollama（开箱即用）

| 模型 | 大小 | 架构 | 激活参数 | 社区实测 | Tool Call | 备注 |
|------|------|------|----------|----------|-----------|------|
| qwen3.5:9b | 6.6GB | Dense | 9B | ~35 tok/s | ✗ | 轻量快速 |
| qwen3.5:35b (A3B) | 23GB | MoE | 3B | ~57 tok/s | △ | 速度质量平衡 |
| qwen3.5:27b | 17GB | Dense | 27B | ~13 tok/s | △ | 慢但强 |
| qwen3.5:122b (A10B) | 81GB | MoE | 10B | ~23 tok/s | △ | 最大 Qwen MoE |
| qwen3:32b | 20GB | Dense | 32B | ~10 tok/s | △ | 对比基准 |
| **gpt-oss:120b** | 65GB | MoE | - | ~41 tok/s | **✓** | NVIDIA 出品，工具调用佳 |
| gpt-oss:20b | 13GB | MoE | - | ~58 tok/s | ✗ | 快速推理 |
| **nemotron-3-nano** | 24GB | MoE | 3B | **~69 tok/s** | ✗ | Spark 最快 |
| nemotron-3-super | 86GB | MoE | 12B | ~20 tok/s | △ | 质量优先 |
| **nemotron-cascade-2** | 24GB | MoE | 3B | **~72 tok/s** | △ | 速度+质量最佳比 |

### Tier 2: vLLM + namake-taro 补丁（性能最优）

| 模型 | 量化 | 社区实测 | Tool Call | 备注 |
|------|------|----------|-----------|------|
| Qwen3.5-35B-A3B | MXFP4 | ~60 tok/s | △ | 比 Ollama 快 |
| gpt-oss-120b | MXFP4 | **~81 tok/s** | 未验证 | 全场最快大模型 |

### Tier 3: vLLM Docker 官方镜像

| 模型 | 量化 | 社区实测 | Tool Call | 备注 |
|------|------|----------|-----------|------|
| Qwen3.5-27B-FP8 | FP8 | ~6 tok/s | **✓✓** | **工具调用精度最高** |

### Tier 4: SGLang Docker

| 模型 | 社区实测 | 备注 |
|------|----------|------|
| gpt-oss-20b | ~61 tok/s | SGLang 原生优化 |

## 测试维度

### 1. 速度（tok/s）
- 短输出（greeting）
- 长输出（500 词作文）
- 中文输出
- 代码生成

### 2. 首 Token 时间（TTFT）
- Ollama 原生报告
- vLLM/SGLang 近似估算

### 3. 质量
- 推理题（all but 9 sheep）
- 代码题（palindrome substring）
- 中文理解（MoE 解释）

### 4. 工具调用（Tool Calling）
- 单工具调用
- 多工具链式调用
- 调用正确率

### 5. 内存占用
- 模型加载后内存
- 推理时峰值内存
- 是否可与其他工作负载共存

## 测试流程

```
Step 1: 下载缺失模型（setup_models.sh Phase 1）
Step 2: Ollama 全模型 benchmark（bench.py --engines ollama）
Step 3: 安装 vLLM + 补丁（setup_models.sh Phase 2）
Step 4: vLLM 模型 benchmark
Step 5: 对比分析，选择最优模型组合
```

## 选型建议（基于社区数据，待验证）

| 使用场景 | 推荐模型 | 原因 |
|----------|----------|------|
| 日常助手（快速对话） | nemotron-cascade-2 | 72 tok/s + 较好质量 |
| 工具调用 Agent | gpt-oss:120b 或 Qwen3.5-27B-FP8 | 工具调用准确率高 |
| 代码生成 | qwen3.5:122b-a10b | 参数量大，代码理解强 |
| 后台推理任务 | gpt-oss-120b (vLLM MXFP4) | 81 tok/s，吞吐最高 |
| 轻量并行 | qwen3.5:9b | 6.6GB，可与大模型共存 |

## 文件说明

```
bench/
├── PLAN.md              # 本文件：测试方案
├── bench.py             # 自动化 benchmark 脚本
├── setup_models.sh      # 模型下载和环境搭建
└── results/             # 测试结果（CSV + JSON）
```
