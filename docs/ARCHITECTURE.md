# 端侧私有智能终端 — 架构设计文档

> 将任何一台带 GPU 的本地设备变成一个完全私有的智能终端：它能听、能看、能说、能操作电脑、能记住你、能主动帮你 —— 所有数据永远不出设备。

## 一、认知模型

架构基于 Kahneman 的双系统认知理论（System 1 / System 2），在机器人领域已被验证（Fast-in-Slow VLA, DuoCore-FS），但尚无人将其应用于端侧 AI 助手框架：

```
System 1 (快系统)                    System 2 (慢系统)
MiniCPM-o 4.5  ~5GB                 Qwen3.5 27B  ~16GB
────────────────────                ────────────────────
• 实时语音对话 (全双工)              • 深度推理 + 规划
• 摄像头/屏幕视觉理解               • 工具调用 (搜索/文件/Shell)
• TTS 语音合成                      • 电脑操控 (鼠标/键盘)
• 始终在线, <1s 延迟                • 按需激活, 3-10s 延迟
• 处理 80% 的日常交互               • 处理 20% 需要行动的请求
```

两个模型**串行使用 GPU**，同一时刻只有一个在推理。用户感知到的是一个既能闲聊又能办事的统一助手。

---

## 二、设计原则

| 原则 | 说明 |
|------|------|
| **认知分层** | System 1 (快/感知) + System 2 (慢/推理)，非分流 |
| **事件驱动** | 组件间通过 EventBus 通信，天然适配流式多模态 |
| **Provider 模式** | 每个模型/工具/频道都是可替换的 Provider |
| **串行 GPU** | 同一时刻只有一个模型推理，通过 GPUScheduler 协调 |
| **极简核心** | 框架核心 < 2500 行 Python，任何人可读懂 |
| **约定优于配置** | SOUL.md / MEMORY.md / skills/ 目录约定 |

---

## 三、系统架构

### 3.1 分层总览

```
Layer 4 ─ Applications
           assistant.py / hypeman.py / narrator.py
           每个应用 ~30 行，组合不同的配置和 Provider

Layer 3 ─ EdgeAgent Runtime
           EventBus + StateMachine + IntentRouter + GPUScheduler
           框架核心，约 500 行

Layer 2 ─ Services
           ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐
           │ System 1         │  │ System 2          │  │ Shared        │
           │ PerceptionLayer  │  │ ReasoningLayer    │  │               │
           │                  │  │                   │  │ MemoryStore   │
           │ • ASR            │  │ • ReAct Loop      │  │ ToolRegistry  │
           │ • Vision         │  │ • Tool Execution  │  │ ChannelMgr    │
           │ • TTS            │  │ • Context Mgmt    │  │               │
           └────────┬─────────┘  └────────┬──────────┘  └───────────────┘
                    │                     │
Layer 1 ─ Providers (可替换)
           ┌────────┴─────────┐  ┌────────┴──────────┐
           │ MiniCPMProvider  │  │ NanobotProvider   │  ← 推荐 (功能最全)
           │ WhisperProvider  │  │ OllamaProvider    │  ← 备选 (零依赖)
           └──────────────────┘  └───────────────────┘

Layer 0 ─ Infrastructure
           GPU (shared) / Camera / Mic / Ollama / llama-server
```

### 3.2 组件关系

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EdgeAgent                                    │
│                                                                      │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │ EventBus │◄──►│ StateMachine │◄──►│ GPUScheduler │              │
│   └────┬─────┘    └──────────────┘    └──────┬───────┘              │
│        │                                      │                      │
│   ┌────┴────────────────────────┐    ┌───────┴───────────────┐      │
│   │       IntentRouter          │    │  pause() / resume()   │      │
│   │  classify(text) → fast|slow │    │                       │      │
│   └────┬───────────────┬────────┘    └───────────────────────┘      │
│        │               │                                             │
│   ┌────┴────┐    ┌─────┴──────┐    ┌──────────────┐                │
│   │System 1 │    │ System 2   │    │   Shared     │                │
│   │Percept. │    │ Reasoning  │    │              │                │
│   │Provider │    │ Provider   │    │ MemoryStore  │                │
│   │         │    │            │    │ ToolRegistry │                │
│   │ start() │    │ reason()   │    │ ChannelMgr   │                │
│   │ pause() │    │ health()   │    │              │                │
│   │ resume()│    │            │    │              │                │
│   │ feed()  │    │            │    │              │                │
│   │ inject()│    │            │    │              │                │
│   └─────────┘    └────────────┘    └──────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 进程模型

```
Process 1: ollama serve           (systemd, 常驻, :11434)
Process 2: llama-server           (手动启动, 常驻, :9060)
Process 3: python apps/assistant.py  (EdgeAgent, :8080)
              ├── EventBus
              ├── MiniCPMProvider   (HTTP → Process 2)
              ├── NanobotProvider   (内嵌 AgentLoop → Process 1)
              ├── WebSocketChannel  (← 浏览器连接)
              └── GPUScheduler      (协调 Process 1 & 2 的 GPU 使用)
Process 4: nanobot gateway        (可选, 用于 Telegram/飞书, :19001)
```

Phase 1-2 只需 Process 1 + 3 (CLI 模式)。
Phase 3 加上 Process 2 (接入 MiniCPM-o)。
Phase 4 加上 Process 4 (多频道)。

---

## 四、事件系统

组件间通过事件通信，不直接调用。使每个组件可独立替换和测试。

### 4.1 事件定义

```python
from dataclasses import dataclass, field
import time

@dataclass
class Event:
    """所有事件的基类"""
    ts: float = field(default_factory=time.time)

# ── 感知层事件 (System 1 产出) ──

@dataclass
class Utterance(Event):
    """System 1 生成了一句回复"""
    text: str = ""

@dataclass
class UserSpeech(Event):
    """用户说了一句话 (Web Speech API 转录)"""
    text: str = ""

@dataclass
class VisualScene(Event):
    """视觉场景描述更新"""
    description: str = ""

@dataclass
class Silence(Event):
    """持续静默"""
    duration_s: float = 0.0

# ── 路由事件 ──

@dataclass
class IntentDecision(Event):
    """路由决策结果"""
    intent: str = "fast"   # "fast" (System 1) | "slow" (System 2)
    text: str = ""

# ── 推理层事件 (System 2 产出) ──

@dataclass
class ThinkingStarted(Event):
    """System 2 开始推理"""
    query: str = ""

@dataclass
class ToolExecuting(Event):
    """正在执行工具"""
    tool: str = ""
    args: dict = field(default_factory=dict)

@dataclass
class ReasoningDone(Event):
    """System 2 完成推理，产出最终结果"""
    text: str = ""
    tools_used: list = field(default_factory=list)

# ── 输出事件 ──

@dataclass
class SpeakRequest(Event):
    """请求语音输出"""
    text: str = ""
    via: str = "browser"   # "browser" (speechSynthesis) | "system1" (Token2Wav)

# ── 频道事件 ──

@dataclass
class ChannelMessage(Event):
    """来自外部频道 (Telegram/飞书等) 的消息"""
    channel: str = ""
    sender: str = ""
    text: str = ""
    media: bytes = None

# ── 系统事件 ──

@dataclass
class MemoryUpdated(Event):
    """长期记忆更新"""
    fact: str = ""

@dataclass
class HealthCheck(Event):
    """健康检查"""
    system1_ok: bool = True
    system2_ok: bool = True
```

### 4.2 EventBus

```python
from collections import defaultdict
from typing import Callable, Type
import asyncio

class EventBus:
    """
    轻量级 async pub/sub。
    组件通过 bus.on() 订阅事件，通过 bus.emit() 发布事件。
    """

    def __init__(self):
        self._handlers: dict[Type[Event], list[Callable]] = defaultdict(list)

    def on(self, event_type: Type[Event], handler: Callable) -> None:
        """订阅事件类型"""
        self._handlers[event_type].append(handler)

    async def emit(self, event: Event) -> None:
        """发布事件，异步通知所有订阅者"""
        for handler in self._handlers[type(event)]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"[EventBus] handler error for {type(event).__name__}: {e}")
```

---

## 五、状态机

EdgeAgent 的行为由显式状态机驱动，确保 GPU 使用和用户体验可预测。

### 5.1 状态转移图

```
                    ┌────────────────────────┐
                    │                        │
                    ▼         audio_chunk    │
              ┌──────────┐ ──────────► ┌────┴───────┐
   run() ───► │ LISTENING │             │ PERCEIVING │
              └──────────┘ ◄────────── └─────┬──────┘
                    ▲        [LISTEN]         │
                    │                    user_text
                    │                         │
                    │                    ┌────▼─────┐
                    │         fast       │ ROUTING  │
                    │      ┌─────────────┴──────────┘
                    │      │                  │ slow
                    │      │             ┌────▼─────┐
                    │      │             │ THINKING │◄──┐
                    │      │             └────┬─────┘   │
                    │      │                  │     tool iteration
                    │      │                  │         │
                    │      │             ┌────▼─────┐   │
                    │      │             │ SPEAKING │───┘
                    │      │             └────┬─────┘
                    │      │                  │ done
                    └──────┴──────────────────┘
```

### 5.2 GPU 占用规则

| 状态 | System 1 (MiniCPM-o) | System 2 (Qwen) | 说明 |
|------|----------------------|-----------------|------|
| LISTENING | idle | idle | 等待用户输入 |
| PERCEIVING | **推理中** | idle | 处理音频+视频帧 |
| ROUTING | idle | idle | 关键词匹配, 瞬时完成 |
| THINKING | **已暂停 (break)** | **推理中** | Qwen 做工具调用链 |
| SPEAKING | idle 或 TTS 中 | idle | 浏览器 TTS 或 Token2Wav |

**核心保障**：PERCEIVING 和 THINKING 永不同时发生。转入 THINKING 前必须先 break System 1。

### 5.3 实现

```python
class State:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PERCEIVING = "PERCEIVING"
    ROUTING = "ROUTING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"

VALID_TRANSITIONS = {
    State.IDLE:       {State.LISTENING},
    State.LISTENING:  {State.PERCEIVING, State.IDLE},
    State.PERCEIVING: {State.LISTENING, State.ROUTING},
    State.ROUTING:    {State.PERCEIVING, State.THINKING},
    State.THINKING:   {State.SPEAKING, State.THINKING},  # tool iteration
    State.SPEAKING:   {State.LISTENING},
}

class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self._listeners: list[Callable] = []

    def transition(self, new_state: str) -> None:
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            raise InvalidTransition(f"{self.state} -> {new_state}")
        old = self.state
        self.state = new_state
        for listener in self._listeners:
            listener(old, new_state)

    def on_change(self, callback: Callable) -> None:
        self._listeners.append(callback)
```

---

## 六、Provider 接口

### 6.1 PerceptionProvider (System 1)

```python
from typing import Protocol, AsyncIterator

class PerceptionProvider(Protocol):
    """
    实时多模态感知接口。
    与传统 LLM Provider (request-response) 不同，这是 STREAMING 接口。
    """

    async def start(self, system_prompt: str) -> None:
        """初始化模型，开始处理流水线"""
        ...

    async def pause(self) -> None:
        """暂停推理，释放 GPU 计算资源。
        对应 llama-server /v1/stream/break"""
        ...

    async def resume(self) -> None:
        """恢复推理。
        下一次 feed() 调用会自动触发恢复"""
        ...

    async def feed(self, audio_b64: str, frame_b64: str = "") -> "PerceptionResult":
        """喂入一帧音频 + 可选图像，返回感知结果。
        对应 llama-server prefill + decode"""
        ...

    async def inject_context(self, text: str) -> None:
        """注入外部信息到 System 1 上下文。
        用于告知 System 2 的执行结果。
        对应 /v1/stream/update_session_config"""
        ...

    async def reset(self) -> None:
        """清空上下文，重新初始化。
        对应 /v1/stream/reset + omni_init"""
        ...
```

**MiniCPMProvider** 实现此接口，内部通过 HTTP 调用 llama-server。代码主要从现有 `omni_web_demo.py` 的 `_do_init`、`_do_prefill`、`_do_decode` 重构而来。

### 6.2 ReasoningProvider (System 2)

```python
@dataclass
class ReasoningResult:
    text: str                              # 最终回复文本
    tools_used: list[dict] = field(default_factory=list)
    tokens_used: int = 0

class ReasoningProvider(Protocol):
    """
    深度推理接口，支持工具调用。
    Request-response 模式，单次调用可能内部执行多轮 ReAct。
    """

    async def reason(
        self,
        message: str,
        context: "Context",
        tools: list[dict],
        max_iterations: int = 20,
    ) -> ReasoningResult:
        """
        执行 ReAct 循环:
        1. 发送 message + tools 给 LLM
        2. 如果 LLM 返回 tool_calls → 执行工具 → 追加结果 → 回到 1
        3. 如果 LLM 返回纯文本 → 作为最终回复返回
        """
        ...

    async def health(self) -> bool:
        """检查底层模型是否可用"""
        ...
```

### 6.3 两种 ReasoningProvider 实现

#### NanobotProvider（推荐）

包装 Nanobot 的 `AgentLoop.process_direct()`，免费获得完整的工具链、记忆系统、多频道、Skills 生态。

```python
class NanobotProvider:
    """
    System 2 实现：基于 Nanobot AgentLoop。

    优势:
    - 7 个内置工具 + MCP 扩展 + ClawHub 13K 技能
    - 完整记忆系统 (MemoryConsolidator)
    - 10+ 频道 (Telegram/飞书/微信/钉钉/QQ)
    - Cron + Heartbeat 主动行为
    - 会话持久化 + Skills 系统

    安装: pip install nanobot
    配置: ~/.nanobot/config.json 指向 Ollama
    """

    def __init__(self, config_path: str = None):
        self._loop_thread = None
        self._async_loop = None
        self._agent_loop = None

    async def start(self):
        """
        初始化 Nanobot 组件:
        1. 加载 config.json (provider: custom -> Ollama)
        2. 创建 LLMProvider
        3. 创建 MessageBus
        4. 创建 AgentLoop (含 ToolRegistry, ContextBuilder 等)
        5. 在独立线程启动 asyncio event loop
        """
        from nanobot.config.loader import load_config
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        config = load_config(self._config_path)
        provider = config.match_provider()
        bus = MessageBus()

        self._agent_loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            max_iterations=config.agents.defaults.max_tool_iterations,
            mcp_servers=config.tools.mcp_servers,
        )

    async def reason(self, message, context, tools, max_iterations=20):
        """通过 process_direct 单次调用 Nanobot AgentLoop"""
        result_text = await self._agent_loop.process_direct(message)
        return ReasoningResult(text=result_text)

    async def health(self):
        try:
            import httpx
            r = await httpx.AsyncClient().get("http://localhost:11434/api/tags")
            return r.status_code == 200
        except Exception:
            return False
```

#### OllamaProvider（轻量备选）

直接调用 Ollama Python SDK，自建最小 ReAct 循环。零外部框架依赖。

```python
class OllamaProvider:
    """
    System 2 实现：直接调用 Ollama API。

    优势:
    - 零额外依赖 (仅 ollama Python SDK)
    - 完全可控，代码 < 100 行
    - 适合嵌入式或不需要多频道的场景

    劣势:
    - 无内置多频道
    - 无 Skills/ClawHub 生态
    - 记忆系统需自行管理
    """

    def __init__(self, model: str = "qwen3.5:27b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def reason(self, message, context, tools, max_iterations=20):
        import ollama

        messages = context.to_ollama_messages()
        messages.append({"role": "user", "content": message})
        tools_used = []

        for _ in range(max_iterations):
            response = await ollama.AsyncClient(host=self.base_url).chat(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                think=False,
            )

            if not response.message.tool_calls:
                return ReasoningResult(
                    text=response.message.content or "",
                    tools_used=tools_used,
                )

            messages.append(response.message)
            for call in response.message.tool_calls:
                result = await self._execute_tool(
                    call.function.name,
                    call.function.arguments,
                )
                tools_used.append({
                    "tool": call.function.name,
                    "args": call.function.arguments,
                    "result": str(result)[:2000],
                })
                messages.append({"role": "tool", "content": str(result)})

        return ReasoningResult(text="推理轮数已达上限，请简化请求。")

    async def health(self):
        try:
            import ollama
            await ollama.AsyncClient(host=self.base_url).list()
            return True
        except Exception:
            return False
```

---

## 七、GPUScheduler

```python
from contextlib import asynccontextmanager
import asyncio

class GPUScheduler:
    """
    确保共享 GPU 上的模型串行推理。

    设计要点:
    - 两个模型常驻显存 (128GB 足够)，不做 load/unload
    - 只管理计算时序：同一时刻只有一个模型做 forward pass
    - 通过 PerceptionProvider.pause()/resume() 实现切换

    协议:
    1. 默认状态：System 1 活跃 (感知始终在线)
    2. 需要 System 2 时：
       scheduler.use_reasoning() 上下文管理器:
       → pause System 1
       → (yield) System 2 推理
       → resume System 1
    """

    def __init__(self, perception: PerceptionProvider):
        self._perception = perception
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def use_reasoning(self):
        """上下文管理器：暂停感知 → 执行推理 → 恢复感知"""
        async with self._lock:
            await self._perception.pause()
            try:
                yield
            finally:
                await self._perception.resume()
```

---

## 八、IntentRouter

```python
from abc import ABC, abstractmethod

class IntentRouter(ABC):
    """判断用户话语走 System 1 (fast) 还是 System 2 (slow)"""

    @abstractmethod
    def classify(self, text: str, visual_context: str = "") -> str:
        """返回 "fast" 或 "slow" """
        ...

class KeywordRouter(IntentRouter):
    """
    V1: 关键词匹配。零延迟，零额外推理。

    设计原则:
    - 默认走 fast (System 1)，只有明确触发才走 slow
    - 漏判代价低 (System 1 会尝试回答)
    - 误判代价高 (打断全双工 + 等待延迟)
    - 所以触发词要精确，宁可漏不可误
    """

    TRIGGERS = [
        # 搜索意图
        "搜索", "搜一下", "查一下", "查查", "找一下", "找找",
        # 请求帮助
        "帮我", "帮忙",
        # 记忆意图
        "记住", "记一下", "别忘了",
        # 提醒意图
        "提醒", "提醒我",
        # 计算/分析
        "计算", "算一下", "算算", "分析",
        # 文件/系统操作
        "打开", "运行", "执行", "下载", "创建", "写一个",
        # 翻译/总结
        "总结", "翻译",
    ]

    def classify(self, text: str, visual_context: str = "") -> str:
        for trigger in self.TRIGGERS:
            if trigger in text:
                return "slow"
        return "fast"


class LLMRouter(IntentRouter):
    """
    V2 (Phase 5): 用 System 2 自身做意图分类。
    更准确，但引入 ~2s 额外延迟。
    适合复杂场景，如模糊意图 ("这个东西怎么弄")。
    """

    def classify(self, text: str, visual_context: str = "") -> str:
        # 用 Qwen 做一次快速分类
        # prompt: "判断以下用户请求是否需要工具调用，回复 fast 或 slow"
        ...
```

---

## 九、MemoryStore

```python
from pathlib import Path
import json
import time

class MemoryStore:
    """
    三层记忆系统。

    文件布局 (约定优于配置):
        memory/
        ├── SOUL.md       身份定义 (只读，定义 AI 是谁)
        ├── USER.md       用户画像 (只读，定义用户是谁)
        ├── MEMORY.md     长期记忆 (AI 自动写入，跨会话持久)
        └── sessions/     会话历史 (自动管理)

    注入顺序 (构建 system prompt):
        1. SOUL.md 内容
        2. USER.md 内容
        3. MEMORY.md 最近 N 条事实
        4. 当前会话最近 K 轮对话
    """

    def __init__(self, base_dir: str = "./memory"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._turns: list[dict] = []
        self._max_turns = 100

    # ── 读取身份和记忆 ──

    def soul(self) -> str:
        return self._read_md("SOUL.md")

    def user_profile(self) -> str:
        return self._read_md("USER.md")

    def long_term_memory(self) -> str:
        return self._read_md("MEMORY.md")

    # ── 短期记忆 (对话历史) ──

    def append_turn(self, role: str, text: str) -> None:
        self._turns.append({"role": role, "text": text, "ts": time.time()})
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns // 2:]

    def recent_turns(self, n: int = 20) -> list[dict]:
        return self._turns[-n:]

    # ── 长期记忆写入 ──

    def save_fact(self, fact: str) -> None:
        """追加事实到 MEMORY.md (由 System 2 的 memory_save 工具调用)"""
        mem_path = self.base / "MEMORY.md"
        with open(mem_path, "a", encoding="utf-8") as f:
            f.write(f"\n- [{time.strftime('%Y-%m-%d')}] {fact}")

    # ── 构建上下文 ──

    def build_context(self) -> "Context":
        return Context(
            soul=self.soul(),
            user_profile=self.user_profile(),
            long_term=self.long_term_memory(),
            recent_turns=self.recent_turns(),
        )

    def build_system_prompt(self) -> str:
        parts = []
        soul = self.soul()
        if soul:
            parts.append(soul)
        user = self.user_profile()
        if user:
            parts.append(f"\n关于用户:\n{user}")
        mem = self.long_term_memory()
        if mem:
            lines = mem.strip().split("\n")
            recent_facts = lines[-20:] if len(lines) > 20 else lines
            parts.append(f"\n你记住的事实:\n" + "\n".join(recent_facts))
        return "\n\n".join(parts)

    def _read_md(self, filename: str) -> str:
        path = self.base / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""


@dataclass
class Context:
    """传递给 ReasoningProvider 的上下文对象"""
    soul: str = ""
    user_profile: str = ""
    long_term: str = ""
    recent_turns: list[dict] = field(default_factory=list)

    def to_ollama_messages(self) -> list[dict]:
        """转换为 Ollama chat 格式的 messages"""
        messages = []
        system_parts = [self.soul]
        if self.user_profile:
            system_parts.append(f"关于用户: {self.user_profile}")
        if self.long_term:
            system_parts.append(f"你记住的事实:\n{self.long_term}")
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for turn in self.recent_turns:
            messages.append({
                "role": turn["role"] if turn["role"] in ("user", "assistant") else "user",
                "content": turn["text"],
            })
        return messages
```

---

## 十、ToolRegistry

```python
from typing import Callable, get_type_hints
import inspect
import json

def tool(description: str):
    """装饰器：将普通函数注册为可被 LLM 调用的工具"""
    def decorator(func):
        func._tool_description = description
        return func
    return decorator


class ToolRegistry:
    """
    工具注册表。
    将 Python 函数自动转换为 OpenAI function-calling schema。
    """

    def __init__(self, tools: list[Callable] = None):
        self._tools: dict[str, Callable] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, func: Callable) -> None:
        self._tools[func.__name__] = func

    def definitions(self) -> list[dict]:
        """生成 OpenAI function-calling 格式的工具定义"""
        defs = []
        for name, func in self._tools.items():
            desc = getattr(func, "_tool_description", func.__doc__ or "")
            hints = get_type_hints(func)
            params = {}
            sig = inspect.signature(func)
            for pname, param in sig.parameters.items():
                ptype = hints.get(pname, str)
                params[pname] = {
                    "type": _python_type_to_json(ptype),
                    "description": pname,
                }
            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": list(sig.parameters.keys()),
                    },
                },
            })
        return defs

    async def execute(self, name: str, args: dict) -> str:
        func = self._tools.get(name)
        if not func:
            return f"未知工具: {name}"
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return str(result)
        except Exception as e:
            return f"工具执行失败: {e}"
```

### 10.1 内置工具集

五层能力，按需激活：

```python
# ── Level 3: 信息工具 ──

@tool("搜索互联网")
def web_search(query: str) -> str:
    """通过 DuckDuckGo 搜索，无需 API key"""
    ...

@tool("抓取网页内容")
def web_fetch(url: str) -> str:
    """获取 URL 内容，HTML 转纯文本"""
    ...

@tool("读取文件内容")
def read_file(path: str) -> str: ...

@tool("写入文件")
def write_file(path: str, content: str) -> str: ...

@tool("编辑文件中的内容")
def edit_file(path: str, old_text: str, new_text: str) -> str: ...

@tool("列出目录内容")
def list_dir(path: str) -> str: ...

@tool("执行 Shell 命令")
def shell(command: str) -> str:
    """带安全过滤，拒绝 rm -rf 等危险命令"""
    ...

# ── Level 4: 电脑操控工具 ──

@tool("截取屏幕截图并描述内容")
def screenshot() -> str:
    """截图后发给 System 1 (MiniCPM-o) 理解屏幕内容"""
    ...

@tool("点击屏幕坐标")
def click(x: int, y: int) -> str: ...

@tool("键盘输入文本")
def type_text(text: str) -> str: ...

@tool("滚动页面")
def scroll(direction: str, amount: int = 3) -> str: ...

@tool("按下快捷键")
def hotkey(keys: str) -> str:
    """如 'ctrl+c', 'alt+tab'"""
    ...

# ── Level 5: 自治工具 ──

@tool("保存一条重要事实到长期记忆")
def memory_save(fact: str) -> str: ...

@tool("创建定时任务")
def cron_add(schedule: str, task: str) -> str: ...

@tool("通过 Telegram 发送消息")
def send_message(channel: str, text: str) -> str: ...
```

---

## 十一、Channel 系统

```python
class Channel(Protocol):
    """频道抽象：任何用户可以发消息/接收消息的通道"""

    name: str

    async def start(self, bus: EventBus) -> None:
        """启动频道，开始监听消息"""
        ...

    async def send(self, text: str, media: bytes = None) -> None:
        """发送消息到频道"""
        ...


class WebSocketChannel:
    """
    主频道：浏览器语音 + 摄像头。
    这是唯一连接 System 1 的频道。

    协议 (沿用现有设计):
    Client → {type:"prepare", scenario:"default"}
    Server → {type:"prepared"}
    Client → {type:"audio_chunk", audio:base64, frame:base64}
    Server → {type:"result", text:"...", is_listen:bool}
    Server → {type:"audio", chunks:[...]}
    Client → {type:"user_text", text:"..."}     ← 触发 IntentRouter
    Server → {type:"agent_status", status:"thinking"}
    Server → {type:"agent_result", text:"..."}
    """
    name = "websocket"


class TelegramChannel:
    """
    文字 + 图片频道。直连 System 2，不经过 System 1。
    使用 NanobotProvider 时，可直接用 Nanobot 内置的 TelegramChannel。
    """
    name = "telegram"


class CLIChannel:
    """开发调试频道。stdin/stdout。"""
    name = "cli"
```

**频道与系统的关系**:

```
WebSocket ──── System 1 ──── IntentRouter ──── System 2
                  │                               │
                  └── 语音+摄像头，全双工 ──────────┘

Telegram ───── System 2 (直连)
飞书     ───── System 2 (直连)
CLI      ───── System 2 (直连)
```

面对面用 WebSocket（语音+摄像头+双系统），远程用 Telegram/飞书（文字+System 2）。

---

## 十二、EdgeAgent 主编排

```python
class EdgeAgent:
    """
    顶层编排器：组合所有组件，驱动主循环。

    用法:
        agent = EdgeAgent(
            perception=MiniCPMProvider(
                server_url="http://localhost:9060",
                model_dir="./models/MiniCPM-o-4_5-gguf",
            ),
            reasoning=NanobotProvider(),  # 或 OllamaProvider(model="qwen3.5:27b")
            tools=[web_search, shell, read_file, screenshot, click, memory_save],
            memory_dir="./memory",
        )
        await agent.run(port=8080)
    """

    def __init__(
        self,
        perception: PerceptionProvider,
        reasoning: ReasoningProvider,
        router: IntentRouter = None,
        tools: list[Callable] = None,
        memory_dir: str = "./memory",
        channels: list[Channel] = None,
    ):
        self.perception = perception
        self.reasoning = reasoning
        self.router = router or KeywordRouter()
        self.bus = EventBus()
        self.state = StateMachine()
        self.scheduler = GPUScheduler(perception)
        self.memory = MemoryStore(memory_dir)
        self.tool_registry = ToolRegistry(tools)
        self.channels = channels or []

        self._last_visual = ""

        # 注册核心事件处理
        self.bus.on(UserSpeech, self._handle_user_speech)
        self.bus.on(ChannelMessage, self._handle_channel_message)
        self.bus.on(VisualScene, self._handle_visual_update)

    async def _handle_user_speech(self, event: UserSpeech):
        """
        用户说了一句话 (来自 Web Speech API)。
        核心分发逻辑。
        """
        self.memory.append_turn("user", event.text)
        intent = self.router.classify(event.text, self._last_visual)

        if intent == "slow":
            await self._delegate_to_system2(event.text)

    async def _delegate_to_system2(self, text: str):
        """暂停 System 1 → System 2 推理 → 恢复 System 1"""
        self.state.transition(State.THINKING)
        await self.bus.emit(ThinkingStarted(query=text))

        async with self.scheduler.use_reasoning():
            context = self.memory.build_context()
            # 如果有视觉上下文，附加到消息中
            full_message = text
            if self._last_visual:
                full_message += f"\n\n[当前视觉场景: {self._last_visual}]"

            result = await self.reasoning.reason(
                message=full_message,
                context=context,
                tools=self.tool_registry.definitions(),
            )

        self.memory.append_turn("assistant", result.text)
        self.state.transition(State.SPEAKING)
        await self.bus.emit(ReasoningDone(text=result.text))
        # 通知 System 1 结果 (使后续对话连贯)
        await self.perception.inject_context(
            f"你刚才帮用户完成了: {result.text[:200]}"
        )
        self.state.transition(State.LISTENING)

    async def _handle_channel_message(self, event: ChannelMessage):
        """来自 Telegram/飞书 等外部频道。直走 System 2。"""
        self.memory.append_turn("user", event.text)

        async with self.scheduler.use_reasoning():
            context = self.memory.build_context()
            result = await self.reasoning.reason(
                message=event.text,
                context=context,
                tools=self.tool_registry.definitions(),
            )

        self.memory.append_turn("assistant", result.text)
        for ch in self.channels:
            if ch.name == event.channel:
                await ch.send(result.text)

    async def _handle_visual_update(self, event: VisualScene):
        self._last_visual = event.description

    async def run(self, host="0.0.0.0", port=8080):
        """启动所有服务"""
        # 1. 启动 System 1
        system_prompt = self.memory.build_system_prompt()
        await self.perception.start(system_prompt)

        # 2. 启动外部频道
        for ch in self.channels:
            await ch.start(self.bus)

        # 3. 启动 WebSocket 服务
        # (Flask + flask_sock, 复用现有前端)
        ...
```

---

## 十三、项目目录结构

```
omni-lab/
│
├── edge_agent/                      # 框架核心 (~2500 行)
│   ├── __init__.py                  # EdgeAgent, 公共 API
│   ├── events.py                    # Event 定义 + EventBus (~100 行)
│   ├── state.py                     # StateMachine (~80 行)
│   ├── scheduler.py                 # GPUScheduler (~40 行)
│   ├── router.py                    # IntentRouter + KeywordRouter (~60 行)
│   ├── memory.py                    # MemoryStore + Context (~150 行)
│   ├── tools.py                     # ToolRegistry + @tool 装饰器 (~120 行)
│   │
│   ├── providers/                   # 模型适配器
│   │   ├── __init__.py              # Protocol 定义 (~80 行)
│   │   ├── minicpm.py               # MiniCPMProvider (~300 行, 从 omni_web_demo.py 重构)
│   │   ├── nanobot.py               # NanobotProvider (~100 行)
│   │   └── ollama.py                # OllamaProvider (~100 行)
│   │
│   ├── channels/                    # 频道适配器
│   │   ├── __init__.py              # ChannelManager
│   │   ├── websocket.py             # WebSocket + 前端 (~500 行, 含 HTML/JS)
│   │   ├── telegram.py              # Telegram (~200 行)
│   │   └── cli.py                   # CLI 调试 (~80 行)
│   │
│   └── tools_builtin/              # 内置工具
│       ├── web.py                   # web_search, web_fetch (~150 行)
│       ├── filesystem.py            # read/write/edit/list (~100 行)
│       ├── shell.py                 # shell + 安全过滤 (~80 行)
│       ├── computer.py              # screenshot/click/type/scroll (~120 行)
│       └── system.py                # memory_save/cron/send_message (~80 行)
│
├── apps/                            # 应用 (使用框架的具体场景)
│   ├── assistant.py                 # 私有助手 (~30 行)
│   ├── hypeman.py                   # AI 夸夸机 (~30 行)
│   └── narrator.py                  # 实时解说 (~30 行)
│
├── memory/                          # Agent 记忆空间 (git tracked)
│   ├── SOUL.md                      # AI 身份定义
│   ├── USER.md                      # 用户画像
│   └── MEMORY.md                    # 长期记忆 (AI 自动写入)
│
├── docs/
│   └── ARCHITECTURE.md              # 本文档
│
├── llama.cpp-omni/                  # 现有子模块
├── models/                          # 模型文件 (gitignored)
├── eval/                            # 评测脚本 (现有)
├── scripts/
│   ├── install.sh                   # 一键安装
│   ├── start.sh                     # 一键启动
│   └── download_models.sh           # 模型下载
├── pyproject.toml                   # Python 包定义
└── README.md
```

---

## 十四、硬件兼容矩阵

| 硬件 | 显存 | System 1 | System 2 | 能力等级 |
|------|------|---------|---------|---------|
| **GB10 / GB200** | 128 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 27B Q4 (16GB) | L1-L5 全部 |
| RTX 4090 | 24 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 9B Q4 (6GB) | L1-L4 |
| RTX 3060 | 12 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 4B Q4 (3GB) | L1-L3 |
| Jetson Orin | 32-64 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 14B Q4 (9GB) | L1-L4 |
| Mac M4 Ultra | 192 GB | MiniCPM-o 4.5 (mlx) | Qwen3.5 27B (mlx) | L1-L5 |

框架不绑定具体模型。用户通过 Provider 接口选择适合自己硬件的模型组合。

---

## 十五、安全模型

```
1. 工具沙箱
   - shell: 过滤 rm -rf, mkfs, shutdown, fork bomb 等
   - filesystem: 可配置 restrict_to_workspace
   - computer use: 敏感操作前要求用户确认

2. 网络隔离
   - 所有模型推理完全本地
   - web_search / web_fetch 是唯一出网点，可关闭

3. 数据主权
   - 所有数据存储在本地文件系统
   - MEMORY.md 是纯文本 Markdown，用户可随时审查/编辑
   - 会话历史是 JSONL，可导出/删除

4. 频道安全
   - Bot Token 本地存储
   - WebSocket 通过 HTTPS + 自签证书加密
```

---

## 十六、实施路线图

### Month 1: L1 + L2 (语音对话 + 视觉理解)

**Week 1**: 框架核心 + CLI 模式
- `events.py`, `state.py`, `scheduler.py`, `router.py`
- `memory.py`, `tools.py`
- `providers/ollama.py` (OllamaProvider, 备选 ReAct)
- `providers/nanobot.py` (NanobotProvider, 推荐)
- `channels/cli.py`
- 验证: CLI 对话 + 工具调用 + 记忆

**Week 2**: System 1 接入
- `providers/minicpm.py` (从 omni_web_demo.py 重构)
- `channels/websocket.py` (含前端)
- GPUScheduler 串行调度验证
- 验证: 语音对话 + 摄像头理解

**Week 3-4**: 打磨
- IntentRouter 调优
- 双系统切换体验优化
- 上下文自动管理
- 记忆系统端到端验证

### Month 2: L3 + L4 (工具使用 + 电脑操控)

**Week 5-6**: 工具链 + Computer Use
- `tools_builtin/` 完善
- screenshot -> MiniCPM-o 理解 -> Qwen 操控
- 安全机制

**Week 7-8**: 多频道 + 记忆深化
- Telegram / 飞书频道
- 长期记忆压缩
- 跨频道共享记忆

### Month 3: L5 + 产品化

- Cron / Heartbeat 主动行为
- 一键安装脚本
- docker-compose
- 开源发布 v0.1.0

---

## 十七、与现有方案的对比

| 维度 | OpenClaw | Nanobot | LangChain | **本框架** |
|------|---------|---------|-----------|-----------|
| 输入模式 | 文字 | 文字 | 文字 | **实时音视频 + 文字** |
| 模型架构 | 单模型 | 单模型 | 单模型 | **双系统 (快+慢)** |
| GPU 管理 | 无 | 无 | 无 | **内置调度器** |
| 交互模式 | 半双工 | 半双工 | 半双工 | **全双工** |
| 部署 | Docker/云 | pip | pip/云 | **端侧原生** |
| 隐私 | 可选 | 可选 | 需云 | **强制本地** |
| 核心代码 | 50K+ | 5K+ | 100K+ | **~2.5K** |
| Computer Use | 无 | 无 | 无 | **内置** |
| 语言 | Node.js | Python | Python | **Python** |
