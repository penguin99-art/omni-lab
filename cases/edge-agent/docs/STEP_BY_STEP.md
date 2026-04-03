# Step By Step 使用与测试流程

这份文档按「先跑通，再验证，再扩展」的顺序写，目标是让你从 0 到 1 把这个项目用起来，并知道每一步该怎么测。

## 0. 你会得到什么

这个仓库目前有 3 条主路径：

1. `CLI` 助手：最快跑通，只依赖 Ollama。
2. `Second Brain`：浏览器 + 麦克风 + 摄像头 + MiniCPM-o + Qwen。
3. `API Gateway`：把这台机器当成本地 OpenAI 兼容接口服务。

建议顺序：

1. 先跑 `CLI`
2. 再跑 `pytest`
3. 再跑 `API Gateway`
4. 最后跑 `Second Brain`

## 1. 环境准备

### 1.1 系统要求

- Linux
- Python 3.11+
- NVIDIA GPU + CUDA
- 已安装 [Ollama](https://ollama.com)

### 1.2 克隆与虚拟环境

```bash
git clone https://github.com/pineapi/gy.git
cd gy
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 安装项目

基础安装：

```bash
pip install -e .
```

如果你要跑测试：

```bash
pip install -e ".[dev]"
```

如果你要启用语义记忆检索：

```bash
pip install -e ".[memory]"
```

如果你要启用桌面操控工具：

```bash
pip install -e ".[computer]"
```

### 1.4 拉取 System 2 模型

```bash
ollama pull qwen3.5:27b
```

显存不够时可以先改成更小模型验证链路，例如：

```bash
ollama pull qwen3.5:14b
```

然后通过环境变量覆盖模型名：

```bash
OLLAMA_MODEL=qwen3.5:14b python3 apps/assistant.py
```

## 2. 第一条路径：先跑通 CLI 助手

这是最短路径，建议先确认这个能用。

### 2.1 启动 Ollama

终端 A：

```bash
ollama serve
```

### 2.2 启动 CLI 助手

终端 B：

```bash
cd /home/pineapi/gy
source .venv/bin/activate
python3 apps/assistant.py
```

### 2.3 CLI 快速验收

依次输入下面这些句子：

```text
你好，你是谁？
帮我搜一下 Python 3.13 有什么新特性
记住：下周三下午 3 点和张总开会
上次我让你记住了什么？
```

### 2.4 你应该看到什么

- 普通闲聊有回复，且回复中体现了 SOUL.md 中定义的身份
- 带"帮我搜一下"的请求会触发 System 2 工具调用
- `记住` 会通过 `memory_save` 工具写入 `memory/MEMORY.md`
- "上次我让你记住了什么" 能从对话上下文或长期记忆中找回

### 2.5 关键日志信息

正常启动时你会看到：

```text
EdgeAgent v0.3.0 starting...
System 2 (reasoning) health: True
CLI channel started. Type your message and press Enter.
EdgeAgent ready. Tools: ['web_search', 'web_fetch', 'read_file', ...]
```

工具调用时会看到：

```text
ReAct iteration 1/20
Tool call: memory_save({'fact': '...'})
Turn completed: 21 chars, 1 tools, 20805ms
```

### 2.6 CLI 失败时先看哪里

- Ollama 没启动：先确认 `ollama serve`
- 模型没拉下来：先 `ollama pull qwen3.5:27b`
- Python 依赖没装：重新执行 `pip install -e .`

## 3. 第二条路径：跑测试

这一步用来确认代码层面没有坏。

### 3.1 安装测试依赖

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3.2 跑自动化测试

```bash
python3 -m pytest tests -q
```

当前预期结果：

```text
49 passed
```

### 3.3 做一次语法检查

```bash
python3 -m compileall edge_agent apps tests scripts
```

### 3.4 测试覆盖了什么

| 测试文件 | 覆盖范围 |
|---------|---------|
| `tests/test_context.py` | ContextBuilder、OllamaContextRenderer、ConversationEngine pipeline |
| `tests/test_memory.py` | MemoryStore 读写、语义检索、事实解析、便捷别名 |
| `tests/test_tools.py` | Tool 协议、ToolPool 编排、deny list、并行执行 |
| `tests/test_events.py` | EventBus 事件发布订阅 |
| `tests/test_ollama.py` | OllamaProvider + QueryLoop mock |

### 3.5 没覆盖什么

- 真实浏览器音频链路
- 真实 MiniCPM-o 服务交互
- 长时间稳定性
- GPU 性能与延迟

这些要靠后面的手工测试。

## 4. 第三条路径：跑 API Gateway

这一步用于把本机变成本地 OpenAI 兼容服务。

### 4.1 启动 Ollama

```bash
ollama serve
```

### 4.2 启动 API Gateway

新终端：

```bash
cd /home/pineapi/gy
source .venv/bin/activate
python3 -m edge_agent.api --host 0.0.0.0 --port 8000
```

### 4.3 健康检查

```bash
curl http://localhost:8000/health
```

预期返回类似：

```json
{"ollama": true, "minicpm": false}
```

如果你还没启动 MiniCPM-o，`minicpm: false` 是正常的。

### 4.4 查看模型列表

```bash
curl http://localhost:8000/v1/models
```

### 4.5 发一个最小 chat request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:27b",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ],
    "stream": false
  }'
```

### 4.6 这一步验证的是什么

- `edge_agent/api.py` 能正常启动
- 纯文本请求能正确转发到 Ollama
- 局域网 API 入口已经可用

注意：API Gateway 目前还是独立路径，尚未接入 ConversationEngine + ContextBuilder 统一 runtime。

## 5. 第四条路径：跑 Second Brain

这是完整体验路径，但依赖最多。

### 5.1 下载 MiniCPM-o 模型

```bash
bash download_models.sh
```

### 5.2 编译 llama-server

```bash
cd llama.cpp-omni
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON
make -j"$(nproc)" llama-server
cd ../..
```

### 5.3 启动 Ollama

终端 A：

```bash
ollama serve
```

### 5.4 启动 MiniCPM-o 服务

终端 B：

```bash
cd /home/pineapi/gy
llama.cpp-omni/build/bin/llama-server \
  --model models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --port 9060 \
  --gpu-layers 99
```

### 5.5 启动应用

终端 C：

```bash
cd /home/pineapi/gy
source .venv/bin/activate
python3 apps/second_brain.py
```

### 5.6 打开浏览器

访问：

- `http://localhost:8080`
- 如果你目录里有 `ssl_cert.pem` 和 `ssl_key.pem`，也可以试 `https://localhost:8080`

### 5.7 浏览器权限

首次打开时：

1. 允许麦克风权限
2. 允许摄像头权限
3. 点击中间的麦克风按钮开始

### 5.8 Second Brain 手工测试顺序

建议按这个顺序：

#### 测试 A：基础连通

说一句：`你好`

预期：
- 页面状态变成 `LISTENING`
- 说话后能看到字幕
- 有语音或文本回复

#### 测试 B：System 2 路由

说：`帮我搜一下今天的 NVIDIA 新闻`

预期：
- 状态切到 `THINKING`
- 前端出现 System 2 返回结果
- 工具提示区出现工具名

#### 测试 C：记忆写入

说：`记住：我的工位在靠窗第二排`

然后检查：

```bash
cat memory/MEMORY.md
```

预期：新增一条带时间戳的记忆

#### 测试 D：记忆找回

说：`上次我说工位在哪里？`

预期：System 2 能从记忆中找回接近的内容

#### 测试 E：视觉链路

对着摄像头举一个明显物体，然后说：`我现在手里拿的是什么？`

预期：请求会带 frame，回答应包含当前视觉上下文

## 6. 稳定性检查

### 6.1 手动跑一次健康检查

```bash
cd /home/pineapi/gy
source .venv/bin/activate
python3 scripts/health_check.py
```

### 6.2 配 cron 每小时跑一次

```bash
crontab -e
```

加入：

```cron
0 * * * * cd /home/pineapi/gy && /home/pineapi/gy/.venv/bin/python scripts/health_check.py >> /tmp/edge-agent-health.log 2>&1
```

### 6.3 长稳测试建议

最少做这 3 项：

1. 连续对话 30 分钟
2. 静置 12 小时后再唤醒一次
3. 连续运行 24 小时，观察 `stability.jsonl`

重点看：OOM、响应劣化、WebSocket 断开、MiniCPM-o 是否需要重启。

## 7. 推荐验收顺序

快速验证：

1. `pip install -e .`
2. `ollama serve`
3. `python3 apps/assistant.py`
4. 用 4 句测试语跑一遍 CLI
5. `pip install -e ".[dev]"`
6. `python3 -m pytest tests -q`（预期 49 passed）

完整验证：

1. CLI 跑通
2. 自动化测试通过（49 tests）
3. API Gateway 跑通
4. MiniCPM-o 服务跑通
5. 浏览器 Second Brain 跑通
6. health check 跑通
7. 做 30 分钟对话稳定性测试

## 8. 常见问题

### `python: command not found`

用 `python3`。

### `No module named pytest`

```bash
pip install -e ".[dev]"
```

### `pip install -e .` 很慢

正常现象。先只装基础依赖，需要语义检索时再装 `.[memory]`。

### 浏览器打开了但没有声音 / 没有字幕

1. 浏览器是否给了麦克风权限
2. `apps/second_brain.py` 是否在运行
3. `llama-server` 是否监听在 `9060`
4. `ollama serve` 是否正常

### 视觉问答效果差

区分两件事：代码链路是否通了、模型理解质量是否够。链路检查：浏览器摄像头预览正常 → 发问时带了 frame → System 2 回复中体现视觉上下文。
