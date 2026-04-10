# GPU Model Bench — 使用指南

本指南面向第一次使用仓库的人，目标是用最少命令完成一次标准化 benchmark，并拿到可读报告。

## 1. 前置条件

- Python 3 可用：`python3 --version`
- Ollama 服务可访问
- 目标模型已经拉取完成
- 如果要跑 Gemma 4，**必须使用 Ollama 0.20+**（推荐 0.20.4+，gemma4:26b 可获得约 11% 提速）

在当前 DGX Spark 环境中，Gemma 4 通过独立的 Ollama 0.20.4 实例运行：

```bash
export OLLAMA_HOST=http://localhost:11436
curl -s $OLLAMA_HOST/api/version
```

预期输出：

```json
{"version":"0.20.4"}
```

## 2. 推荐流程

推荐按“两阶段”执行：

1. `--list` 查看当前平台和可用模型
2. `--suite quick` 先筛出速度/效率最好的候选
3. 对候选模型跑 `--suite standard`
4. 如需工具调用，再跑 `--suite toolcall`
5. 如需首 token 延迟，再跑 `--suite ttft`
6. 如需跨引擎对比，再加 `--matrix`

## 3. 常用命令

### 查看平台与模型

```bash
OLLAMA_HOST=http://localhost:11436 python3 bench/bench.py --list
```

### 快速筛选

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models gemma4:e2b qwen3.5:35b gemma4:26b \
  --suite quick \
  --tag workflow_quick
```

### 标准评测

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models gemma4:e2b \
  --suite standard \
  --tag workflow_standard
```

### 工具调用评测

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models qwen3.5:35b \
  --suite toolcall \
  --tag toolcall_check
```

### TTFT 专项测试

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models gemma4:e2b \
  --suite ttft \
  --ttft-runs 3 \
  --tag ttft_check
```

### 引擎矩阵

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --suite quick \
  --matrix \
  --matrix-group qwen3.5-35b \
  --tag matrix_check
```

## 4. 输出文件

每次运行会自动生成 3 个文件：

- `bench_*.csv`：原始表格数据
- `bench_*.json`：结构化数据
- `bench_*.md`：自动报告

自动报告包含：

1. 综合排名
2. 每个 prompt 的速度/质量表
3. Key Findings
4. 内存峰值 / swap 压力
5. 架构对比
6. 失败模型和错误原因

额外报告：

- `ttft_*.md`：TTFT 专项报告
- `matrix_*.md`：引擎矩阵报告

## 5. 本次已验证流程

下面是本次在 DGX Spark 上实际跑通的一组命令与结果。

### Step 1: 查看平台

```bash
OLLAMA_HOST=http://localhost:11436 python3 bench/bench.py --list
```

确认平台识别为：

- `dgx-spark`
- GPU: `GB10`
- Memory: `122GB`

### Step 2: quick 筛选

执行命令：

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models gemma4:e2b qwen3.5:35b gemma4:26b \
  --suite quick \
  --tag workflow_quick
```

生成文件：

- `bench/results/dgx-spark/bench_20260408_115629_workflow_quick.csv`
- `bench/results/dgx-spark/bench_20260408_115629_workflow_quick.json`
- `bench/results/dgx-spark/bench_20260408_115629_workflow_quick.md`

结果摘要：

- `gemma4:e2b`: `107.1 tok/s`, `TTFT 0.02s`
- `gemma4:26b`: `59.2 tok/s`, `TTFT 0.07s`
- `qwen3.5:35b`: `58.0 tok/s`, `TTFT 0.08s`

本轮 winner：

- **速度优先**：`gemma4:e2b`
- **质量/复杂任务候选**：`qwen3.5:35b`、`gemma4:26b`

### Step 3: standard 深测

执行命令：

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models gemma4:e2b \
  --suite standard \
  --tag workflow_standard
```

生成文件：

- `bench/results/dgx-spark/bench_20260408_115731_workflow_standard.csv`
- `bench/results/dgx-spark/bench_20260408_115731_workflow_standard.json`
- `bench/results/dgx-spark/bench_20260408_115731_workflow_standard.md`

结果摘要：

- 平均速度：`108.0 tok/s`
- 平均 TTFT：`0.02s`
- 综合排名：`#1`
- 各 prompt `quality_score`：全部为 `1.00`

### Step 4: TTFT 专项验证

执行命令：

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --models gemma4:e2b \
  --suite ttft \
  --ttft-runs 2 \
  --tag ttft_v1
```

生成文件：

- `bench/results/dgx-spark/ttft_20260408_122934_ttft_v1.csv`
- `bench/results/dgx-spark/ttft_20260408_122934_ttft_v1.json`
- `bench/results/dgx-spark/ttft_20260408_122934_ttft_v1.md`

结果摘要：

- `short_cold`: `1.446s`
- `short_warm`: `1.443s`
- `long_cold`: `1.489s`
- `long_warm`: `1.462s`

当前结论：

- Ollama 在这组测试里 **warm 几乎没有 prefix cache 收益**

### Step 5: 引擎矩阵验证

执行命令：

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py \
  --suite quick \
  --matrix \
  --matrix-group qwen3.5-35b \
  --tag matrixcheck
```

生成文件：

- `bench/results/dgx-spark/matrix_20260408_134326_matrixcheck.md`

结果摘要：

- 当前 `qwen3.5-35b` compare group 中，`ollama` 可用
- `vllm` 版本已注册，但当前机器上未启动，因此矩阵报告会自动提示缺失引擎

## 6. 如何解读报告

### `quick` 报告看什么

- `tok/s`：吞吐速度
- `TTFT`：首 token 体感延迟
- `tok/s per GB`：内存效率

### `standard` 报告看什么

- `Avg quality`：平均质量分
- `Composite`：综合排名分
- `Best single-prompt quality`：哪一项答得最好
- `Mem peak / Swap peak`：运行过程的内存压力
- `Architecture Comparison`：MoE / Dense 在当前平台上的差异

### `ttft` 报告看什么

- `Case Summary`：short/long × cold/warm 的聚合结果
- `Cache Impact`：warm 相比 cold 是否明显更快
- `Raw Trials`：每次试验的明细

### `matrix` 报告看什么

- 同一 compare group 下各引擎的平均质量 / 速度 / TTFT / 内存峰值
- 每个 prompt 的逐项拆分
- 当前缺失哪些引擎

## 7. 常见问题

### Gemma 4 返回空结果

原因通常是 Ollama 版本太旧。Gemma 4 需要 Ollama 0.20+。

检查：

```bash
curl -s http://localhost:11436/api/version
```

### 跑分文件保存在哪里

默认保存在：

```text
bench/results/{platform}/
```

例如 DGX Spark：

```text
bench/results/dgx-spark/
```

### 怎么只测一个模型

```bash
OLLAMA_HOST=http://localhost:11436 \
python3 bench/bench.py --models gemma4:e2b --suite quick
```

## 8. 建议用法

- 新平台第一次上机：先 `--list`
- 候选模型很多：先跑 `quick`
- 选出 Top 1 / Top 3 后：跑 `standard`
- 需要 Agent 能力：额外跑 `toolcall`
