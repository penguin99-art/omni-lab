# Edge Agent

> 端侧私有 AI 助手框架 — 双系统认知架构

将任何一台带 GPU 的设备变成完全私有的智能终端：能听、能看、能说、能操作电脑、能记住你。所有数据永远不出设备。

## 双系统架构

基于 Kahneman 双系统认知理论，两个模型**串行共享 GPU**，同一时刻只有一个在推理：

```
System 1（快系统）                   System 2（慢系统）
MiniCPM-o 4.5  ~5GB                Qwen3.5 27B  ~16GB
────────────────────                ────────────────────
实时语音对话（全双工）               深度推理 + 规划
摄像头/屏幕视觉理解                 工具调用（搜索/文件/Shell）
TTS 语音合成                        电脑操控（鼠标/键盘）
始终在线, <1s 延迟                  按需激活, 3-10s 延迟
处理 80% 日常交互                   处理 20% 需要行动的请求
```

用户感知到的是一个既能闲聊又能办事的统一助手。

---

## 快速开始

### 环境要求

- Python 3.11+
- NVIDIA GPU + CUDA
- [Ollama](https://ollama.com) 已安装

### 安装

```bash
git clone https://github.com/pineapi/gy.git && cd gy
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
ollama pull qwen3.5:27b    # 或更小的: qwen3.5:14b / qwen3.5:4b
```

### 模式一：CLI 助手（最简单，只需 Ollama）

```bash
# 终端 1
ollama serve

# 终端 2
source .venv/bin/activate
python apps/assistant.py
```

在终端输入对话即可。能力：

| 能力 | 示例 |
|------|------|
| 搜索互联网 | "帮我搜一下 CUDA kernel fusion 最新进展" |
| 抓取网页 | "帮我看看这个链接讲了什么" |
| 读写文件 | "看看 pyproject.toml 的内容" / "帮我写一个 config.yaml" |
| 编辑文件 | "把 README 里的 v0.1 改成 v0.2" |
| 执行命令 | "运行 nvidia-smi 看看 GPU 状态" |
| 保存记忆 | "记住，下周三要给张总做技术分享" |
| 多轮工具链 | "帮我搜一下 FastAPI 最新版本，然后写到 notes.md 里" |

所有对话和工具调用都在本地完成。AI 自动判断何时调用哪个工具，可连续调用多个。

### 模式二：第二大脑（System 1 + System 2，语音+摄像头）

需要额外下载 MiniCPM-o 模型并编译 llama-server。

```bash
# 1. 下载 MiniCPM-o 模型（约 10GB）
bash download_models.sh

# 2. 编译 llama-server（需要 CUDA 开发环境）
cd llama.cpp-omni
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON && make -j$(nproc) llama-server
cd ../..

# 3. 启动三个进程
ollama serve                                    # 终端 1

llama.cpp-omni/build/bin/llama-server \         # 终端 2
    --model models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    --port 9060 --gpu-layers 99

source .venv/bin/activate && python apps/second_brain.py   # 终端 3
```

浏览器打开 `https://localhost:8080`，点 Start 授权麦克风和摄像头。

在模式一的基础上增加：
- **实时语音对话**：全双工，<1s 延迟
- **摄像头视觉理解**：MiniCPM-o 能看到你面前的东西
- **TTS 语音合成**：AI 回复可直接语音播放
- **语音触发工具调用**：说"帮我搜一下..."，自动路由到 System 2

### 可选：HTTPS（远程访问需要）

浏览器要求 HTTPS 才能访问麦克风。生成自签证书：

```bash
openssl req -x509 -newkey rsa:2048 -keyout ssl_key.pem -out ssl_cert.pem \
    -days 365 -nodes -subj '/CN=localhost'
```

### 可选：桌面操控工具

```bash
pip install -e ".[computer]"
```

安装后 AI 可以截屏、点击、输入、滚动、按快捷键。

---

## 项目结构

```
gy/
├── edge_agent/                      # 框架核心（~1700 行 Python）
│   ├── __init__.py                  # EdgeAgent 编排器
│   ├── events.py                    # 12 种事件类型 + EventBus
│   ├── state.py                     # StateMachine（7 种状态）
│   ├── scheduler.py                 # GPUScheduler（串行 GPU 调度）
│   ├── router.py                    # IntentRouter（关键词路由 fast/slow）
│   ├── memory.py                    # MemoryStore（三层 Markdown 记忆）
│   ├── tools.py                     # ToolRegistry + @tool 装饰器
│   │
│   ├── providers/                   # 模型适配器（可替换）
│   │   ├── __init__.py              # PerceptionProvider / ReasoningProvider 协议
│   │   ├── minicpm.py               # System 1: MiniCPM-o via llama-server HTTP
│   │   └── ollama.py                # System 2: Qwen via Ollama + ReAct 循环
│   │
│   ├── channels/                    # 交互频道（可替换）
│   │   ├── __init__.py              # Channel 协议
│   │   ├── cli.py                   # 终端频道（stdin/stdout）
│   │   └── websocket.py             # 浏览器频道（语音+摄像头+前端 UI）
│   │
│   └── tools_builtin/               # 12 个内置工具
│       ├── web.py                   # web_search (DuckDuckGo), web_fetch
│       ├── filesystem.py            # read_file, write_file, edit_file, list_dir
│       ├── shell.py                 # shell（带危险命令过滤）
│       ├── computer.py              # screenshot, click, type_text, scroll, hotkey
│       └── system.py                # memory_save
│
├── apps/                            # 应用入口
│   ├── assistant.py                 # CLI 助手（~37 行）
│   └── second_brain.py              # 完整双系统（~77 行）
│
├── memory/                          # Agent 记忆空间（纯文本 Markdown）
│   ├── SOUL.md                      # AI 身份定义（可自定义）
│   ├── USER.md                      # 用户画像（AI 自动或手动填充）
│   └── MEMORY.md                    # 长期事实记忆（AI 自动写入）
│
├── docs/
│   ├── ARCHITECTURE.md              # 详细架构设计文档（中文，1254 行）
│   └── SCENARIOS.md                 # "私有第二大脑"场景文档（中文，671 行）
│
├── llama.cpp-omni/                  # Git 子模块：支持 MiniCPM-o 的 llama-server
├── download_models.sh               # MiniCPM-o 模型一键下载脚本
├── official_ref_audio.wav           # TTS 参考音频
└── pyproject.toml                   # Python 包定义
```

## 核心组件

### EdgeAgent — 编排器

顶层入口，组合所有组件，驱动主循环。一个 App 只需 ~30 行代码：

```python
from edge_agent import EdgeAgent
from edge_agent.providers.ollama import OllamaProvider
from edge_agent.channels.cli import CLIChannel
from edge_agent.tools_builtin.web import web_search
from edge_agent.tools_builtin.filesystem import read_file, write_file
from edge_agent.tools_builtin.shell import shell
from edge_agent.tools_builtin.system import memory_save

agent = EdgeAgent(
    reasoning=OllamaProvider(model="qwen3.5:27b"),
    tools=[web_search, read_file, write_file, shell, memory_save],
    channels=[CLIChannel()],
)
asyncio.run(agent.run())
```

### EventBus — 事件总线

异步 pub/sub，组件间解耦通信。12 种事件类型：

```
感知层:  Utterance, UserSpeech, VisualScene, Silence
路由层:  IntentDecision
推理层:  ThinkingStarted, ToolExecuting, ReasoningDone
输出层:  SpeakRequest
频道层:  ChannelMessage
系统层:  MemoryUpdated, HealthCheck
```

### GPUScheduler — GPU 调度器

`use_reasoning()` 上下文管理器确保两个模型串行推理：

```
默认:  System 1 活跃（实时感知）
需要推理时:  pause System 1 → System 2 推理 → resume System 1
```

两个模型常驻显存，只调度计算时序，不做加载/卸载。

### IntentRouter — 意图路由

决定用户输入走 System 1（fast）还是 System 2（slow）：

- **V1（当前）**：关键词匹配，零延迟。包含"搜索/帮我/记住/分析/执行"等中文触发词
- **V2（计划）**：用 LLM 做意图分类，更准但有 ~2s 延迟

### MemoryStore — 三层记忆

```
memory/
├── SOUL.md      AI 是谁（人格定义，用户编写，只读）
├── USER.md      用户是谁（用户画像，可手动/自动填充）
└── MEMORY.md    AI 记住了什么（事实记忆，AI 自动写入，跨会话持久）
```

所有记忆都是**纯文本 Markdown**，用户可随时查看、编辑、删除。
构建 System Prompt 时按 SOUL → USER → MEMORY → 近期对话 的顺序注入。

### ToolRegistry — 工具注册表

`@tool` 装饰器自动生成 OpenAI function-calling schema。AI 在 ReAct 循环中自主决定调用哪些工具：

| 工具 | 说明 |
|------|------|
| `web_search` | DuckDuckGo 搜索，无需 API Key |
| `web_fetch` | 抓取网页，HTML 自动转纯文本 |
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件（覆盖） |
| `edit_file` | 替换文件中的指定文本 |
| `list_dir` | 列出目录内容 |
| `shell` | 执行 Shell 命令（过滤危险命令） |
| `screenshot` | 截取屏幕截图 |
| `click` | 点击屏幕坐标 |
| `type_text` | 键盘输入文本 |
| `scroll` | 滚动页面 |
| `hotkey` | 按下快捷键 |
| `memory_save` | 保存事实到长期记忆 |

### Provider 协议

模型通过 Provider 接口接入，可替换：

- **PerceptionProvider**（System 1）：`start`, `pause`, `resume`, `feed`, `inject_context`, `reset`
- **ReasoningProvider**（System 2）：`reason`, `health`

当前实现：`MiniCPMProvider`（HTTP → llama-server）和 `OllamaProvider`（Ollama SDK + ReAct）

### Channel 协议

交互频道可替换：

- **CLIChannel**：终端输入输出，开发调试用
- **WebSocketChannel**：Flask + flask-sock，内嵌前端，支持摄像头/麦克风/语音识别/HTTPS

## 硬件兼容

| 硬件 | 显存 | System 1 | System 2 |
|------|------|----------|----------|
| GB10 / GB200 | 128 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 27B Q4 (16GB) |
| RTX 4090 | 24 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 9B Q4 (6GB) |
| RTX 3060 | 12 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 4B Q4 (3GB) |
| Jetson Orin | 32-64 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 14B Q4 (9GB) |

框架不绑定具体模型，通过 Provider 接口选择适合硬件的组合。

## 隐私

- 所有模型推理 100% 本地
- 所有数据存储在本地文件系统（纯文本 Markdown）
- `web_search` / `web_fetch` 是唯一出网点（可不注册）
- 记忆完全可审计：`cat memory/MEMORY.md`

## 实现状态

### 已完成（v0.1）

- [x] 框架核心：EdgeAgent, EventBus, StateMachine, GPUScheduler
- [x] IntentRouter 关键词路由（中文 + 英文触发词）
- [x] MemoryStore 三层 Markdown 记忆
- [x] ToolRegistry + 12 个内置工具
- [x] OllamaProvider：多轮 ReAct 工具调用循环
- [x] MiniCPMProvider：llama-server 全套 API（init/prefill/decode/break/reset/TTS）
- [x] CLIChannel：终端助手
- [x] WebSocketChannel：浏览器 UI（摄像头+麦克风+语音识别）
- [x] GPU 串行调度（pause S1 → run S2 → resume S1）
- [x] 上下文注入（System 2 结果 → System 1 感知）
- [x] 上下文溢出自动 reset

### TODO

- [ ] **StateMachine 接入**：已定义但 EdgeAgent 未调用 `transition()`
- [ ] **VisualScene 管线**：handler 存在但无 emitter，System 2 拿不到当前画面
- [ ] **被动截屏→摘要→记忆**：事件类型已定义（`CaptureEvent`/`DigestRequest`/`ProactiveHint`），缺发射端和定时循环
- [ ] **WebSocketChannel.send()**：当前是空实现
- [ ] **NanobotProvider**：文档设计了但未实现
- [ ] **Telegram/飞书频道**：文档设计了但未实现
- [ ] **LLMRouter V2**：用 System 2 做意图分类
- [ ] **记忆压缩**：每日合并相似记忆
- [ ] **记忆向量检索**：embedding 语义搜索
- [ ] **定时任务 / Cron**：主动提醒
- [ ] **Docker Compose**：容器化部署
- [ ] **自动化测试**

## 文档

- [架构设计文档](docs/ARCHITECTURE.md)（中文）：双系统认知模型、事件系统、状态机、GPU 调度、Provider/Channel/工具设计、安全模型、实施路线图
- [私有第二大脑场景](docs/SCENARIOS.md)（中文）：用户画像、三种交互模式、信息捕获流水线、记忆架构、Demo 脚本、竞品对比

## License

MIT
