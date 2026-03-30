# Edge Agent

> 端侧私有 AI 助手框架，采用双系统认知架构。

把一台带 GPU 的设备变成完全本地的 AI 终端：能听、能看、能说、能调用工具、能长期记忆。默认不把对话和记忆发到云端。

## 当前状态

这个仓库现在已经具备一套可跑通的最小系统：

- `CLI` 助手可用
- 浏览器端 `WebSocket` 语音界面可用
- `WebSocketChannel.send()` 已实现，System 2 结果可以主动推送到前端
- `VisualScene` 已接入，带画面帧时会更新视觉上下文
- `MemoryStore` 已支持语义检索接口，未安装 embedding 依赖时自动回退到关键词检索
- `StateMachine` 已接入编排流程
- `OllamaProvider` 增加了基础重试和重连
- 提供了 `OpenAI-compatible API gateway`
- 提供了基础自动化测试和 `24/7 health check` 脚本
- Web 前端已经从 Python 字符串中拆出，改成独立静态文件

仍然需要你在真实硬件上做的事：

- Qwen / MiniCPM-o 的真实性能基准
- 30 分钟到 24/7 的长期稳定性实测
- MiniCPM-o 真正的多模态 API 对外封装深化
- systemd / Docker 等部署层

## 双系统架构

```text
System 1（快系统）                   System 2（慢系统）
MiniCPM-o 4.5                        Qwen3.5
────────────────────                 ────────────────────
实时语音 / 摄像头 / TTS              深度推理 / 工具调用 / 规划
持续在线                              按需激活
低延迟感知                            高质量决策
```

两个系统通过 `GPUScheduler` 串行使用 GPU，同一时刻只有一个系统在做推理。

## 安装

### 基础安装

```bash
git clone https://github.com/pineapi/gy.git
cd gy
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 可选依赖

```bash
# 开发与测试
pip install -e ".[dev]"

# 语义记忆检索
pip install -e ".[memory]"

# 桌面操控
pip install -e ".[computer]"
```

说明：

- 默认安装保持轻量，不强制安装 `sentence-transformers` / `torch`
- 安装 `memory` extra 后，`MemoryStore.search_memory()` 会自动启用 embedding 检索
- 不安装 `memory` extra 时，仍可使用记忆功能，只是回退到关键词检索

## 快速开始

### 模式一：CLI 助手

先启动 Ollama：

```bash
ollama serve
```

再启动助手：

```bash
source .venv/bin/activate
python3 apps/assistant.py
```

默认使用 `Qwen3.5` 作为 System 2，支持：

- 联网搜索与网页抓取
- 文件读写与编辑
- Shell 命令执行
- 长期记忆写入
- 多轮工具调用

### 模式二：Second Brain

需要额外准备 MiniCPM-o 和 `llama-server`。

```bash
# 1. 下载模型
bash download_models.sh

# 2. 编译 llama-server
cd llama.cpp-omni
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON
make -j"$(nproc)" llama-server
cd ../..

# 3. 启动 Ollama
ollama serve

# 4. 启动 MiniCPM-o 服务
llama.cpp-omni/build/bin/llama-server   --model models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf   --port 9060   --gpu-layers 99

# 5. 启动应用
source .venv/bin/activate
python3 apps/second_brain.py
```

然后打开：

- `http://localhost:8080`
- 如果有 `ssl_cert.pem` 和 `ssl_key.pem`，则也支持 `https://localhost:8080`

当前浏览器端具备：

- 简洁暗色 UI
- 麦克风开始/停止控制
- 实时字幕显示
- 摄像头预览
- System 1 / System 2 状态指示
- System 2 工具提示与结果回推

## API Gateway

新增 `edge_agent/api.py`，提供轻量 OpenAI 兼容网关。

启动方式：

```bash
source .venv/bin/activate
python3 -m edge_agent.api --host 0.0.0.0 --port 8000
```

可用端点：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

路由策略：

- 纯文本请求优先走 Ollama
- 带图像 / 音频内容的请求预留给 MiniCPM-o 路径

说明：

- 目前文本链路是可用的
- MiniCPM-o 的 OpenAI 兼容多模态包装还是 v0 版本，适合作为局域网网关骨架，不是最终协议实现

## 记忆系统

当前记忆结构：

```text
memory/
├── SOUL.md
├── USER.md
├── MEMORY.md
└── vectors.npz   # 安装 memory extra 后自动生成
```

行为说明：

- `save_fact()` 保持原有 Markdown 追加写入
- 若 embedding 可用，会同时写入 `vectors.npz`
- `build_context(query=...)` 会优先注入与当前问题最相关的记忆
- 没有 embedding 时自动降级，不会影响系统运行

## 稳定性工具

新增脚本：

```bash
python3 scripts/health_check.py
```

它会检查：

- Ollama
- MiniCPM-o
- Web 应用状态接口
- GPU 基础状态

输出会写入 `stability.jsonl`，适合配合 `cron` 使用。

## 测试

当前已加入的基础测试覆盖：

- `tests/test_memory.py`
- `tests/test_tools.py`
- `tests/test_events.py`
- `tests/test_ollama.py`

运行：

```bash
source .venv/bin/activate
python3 -m pytest tests -q
```

## 项目结构

```text
edge_agent/
├── __init__.py
├── api.py
├── events.py
├── memory.py
├── router.py
├── scheduler.py
├── state.py
├── tools.py
├── channels/
│   ├── cli.py
│   └── websocket.py
├── providers/
│   ├── minicpm.py
│   └── ollama.py
├── tools_builtin/
└── web/
    ├── index.html
    ├── style.css
    └── app.js
```

## 已完成的关键改动

- [x] `WebSocketChannel.send()` 实现
- [x] 独立前端静态文件
- [x] `VisualScene` 发射
- [x] 语义记忆检索接口
- [x] `StateMachine` 接入
- [x] `OllamaProvider` 自动重试
- [x] `MiniCPMProvider` 幂等启动
- [x] API gateway
- [x] health check 脚本
- [x] pytest 基础测试
- [x] `pyproject.toml` 安装链路修复

## 暂未完成

- [ ] 真实硬件性能基准
- [ ] MiniCPM-o 真正的 OpenAI 兼容多模态对外协议
- [ ] systemd / Docker 部署
- [ ] 长时间 dogfood 结果沉淀
- [ ] 被动感知 / 主动提醒 / 记忆压缩

## 文档

- `docs/ARCHITECTURE.md`
- `docs/SCENARIOS.md`
- `PLAN.md`

## License

MIT
