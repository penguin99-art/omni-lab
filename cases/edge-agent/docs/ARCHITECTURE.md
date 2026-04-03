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
| **上下文分层** | 装配 (ContextBuilder) → 快照 (ContextSnapshot) → 渲染 (Renderer)，与 Provider 解耦 |
| **串行 GPU** | 同一时刻只有一个模型推理，通过 GPUScheduler 协调 |
| **依赖注入** | query_loop 通过 QueryDeps 注入模型调用和工具执行，可独立测试 |
| **极简核心** | 框架核心 ~3200 行 Python，任何人可读懂 |
| **约定优于配置** | SOUL.md / MEMORY.md / skills/ 目录约定 |

---

## 三、系统架构

### 3.1 分层总览

```
Layer 5 ─ Applications
           assistant.py / second_brain.py
           每个应用 ~40 行，组合不同的配置和 Provider

Layer 4 ─ EdgeAgent Runtime
           EventBus + StateMachine + IntentRouter + GPUScheduler
           框架编排核心

Layer 3 ─ Agent Core
           ┌──────────────────┐  ┌──────────────────┐
           │ ConversationEngine│  │ query_loop()     │
           │ Turn 生命周期管理  │  │ ReAct 状态机      │
           │ submit_message()  │  │ async generator   │
           └────────┬─────────┘  └────────┬──────────┘
                    │                     │
Layer 2 ─ Context + Services
           ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐
           │ Context Layer    │  │ System 1         │  │ Shared        │
           │                  │  │ PerceptionLayer  │  │               │
           │ ContextBuilder   │  │                  │  │ MemoryStore   │
           │ ContextSnapshot  │  │ • ASR            │  │ ToolPool      │
           │ RenderedContext  │  │ • Vision         │  │ ChannelMgr    │
           │ Renderers        │  │ • TTS            │  │               │
           └────────┬─────────┘  └────────┬─────────┘  └───────────────┘
                    │                     │
Layer 1 ─ Providers (可替换)
           ┌────────┴─────────┐  ┌────────┴──────────┐
           │ MiniCPMProvider  │  │ OllamaProvider    │
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
│   ┌────┴────┐    ┌─────┴──────────────────┐    ┌──────────────┐    │
│   │System 1 │    │ ConversationEngine      │    │   Shared     │    │
│   │Percept. │    │  ┌─────────────────┐    │    │              │    │
│   │Provider │    │  │ ContextBuilder  │    │    │ MemoryStore  │    │
│   │         │    │  │      ↓          │    │    │ ToolPool     │    │
│   │ start() │    │  │ContextSnapshot  │    │    │ ChannelMgr   │    │
│   │ pause() │    │  │      ↓          │    │    │              │    │
│   │ resume()│    │  │ Renderer        │    │    │              │    │
│   │ feed()  │    │  │      ↓          │    │    │              │    │
│   │ inject()│    │  │ query_loop()    │    │    │              │    │
│   └─────────┘    │  └─────────────────┘    │    └──────────────┘    │
│                  │  submit_message()        │                        │
│                  │  abort()                 │                        │
│                  └─────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 进程模型

```
Process 1: ollama serve           (systemd, 常驻, :11434)
Process 2: llama-server           (手动启动, 常驻, :9060)
Process 3: python apps/assistant.py  (EdgeAgent, :8080)
              ├── EventBus
              ├── MiniCPMProvider   (HTTP → Process 2)
              ├── OllamaProvider    (→ Process 1)
              ├── ConversationEngine
              │     ├── ContextBuilder
              │     └── query_loop → ToolPool
              ├── WebSocketChannel  (← 浏览器连接)
              └── GPUScheduler      (协调 Process 1 & 2 的 GPU 使用)
```

只跑 CLI 助手时只需 Process 1 + 3。
加上 MiniCPM-o 的完整体验需要 Process 1 + 2 + 3。

---

## 四、上下文层（Context Layer）

上下文层将"从各处收集信息"和"格式化给特定 Provider"解耦为两步。

### 4.1 数据模型

```python
@dataclass(frozen=True)
class TurnInput:
    """用户输入的一次请求。"""
    text: str
    channel: str = "cli"
    sender: str = ""
    visual_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ContextSnapshot:
    """Provider 无关的上下文快照，一次 turn 的所有上下文。"""
    identity: str = ""
    user_profile: str = ""
    recent_turns: list[dict] = field(default_factory=list)
    relevant_memories: list[MemoryHit] = field(default_factory=list)
    visual_context: str = ""
    channel_context: str = ""
    tool_names: list[str] = field(default_factory=list)
    runtime_notes: list[str] = field(default_factory=list)
    turn: TurnInput | None = None

@dataclass(frozen=True)
class RenderedContext:
    """Provider 可直接消费的格式化输出。"""
    system_prompt: str
    messages: list[dict]
    debug: dict[str, Any] = field(default_factory=dict)
```

### 4.2 ContextBuilder

```python
class ContextBuilder:
    """从 MemoryStore、ToolPool、TurnInput 装配 ContextSnapshot。"""

    def build(self, turn: TurnInput | None = None) -> ContextSnapshot:
        # 1. 加载身份 (SOUL.md)
        # 2. 加载用户画像 (USER.md)
        # 3. 获取最近对话历史
        # 4. 语义检索相关记忆（或回退到最近记忆）
        # 5. 收集视觉上下文、频道信息、可用工具
        # 6. 组装为 frozen ContextSnapshot
        ...

    def bootstrap_prompt(self) -> str:
        """为 System 1 启动生成精简 system prompt。"""
        ...
```

### 4.3 ContextRenderer

```python
class OllamaContextRenderer:
    """将 ContextSnapshot 渲染为 Ollama chat 格式。"""

    def render(self, snapshot: ContextSnapshot) -> RenderedContext:
        # system prompt = identity + user_profile + memories + visual + runtime_notes
        # messages = system + recent_turns + current turn
        ...
```

新增 Provider 时只需实现新的 Renderer（如 `OpenAIContextRenderer`），不用改 Builder。

### 4.4 数据流

```
TurnInput
  │
  ▼
ContextBuilder.build()
  ├── MemoryStore.soul()           → identity
  ├── MemoryStore.user_profile()   → user_profile
  ├── MemoryStore.recent_turns()   → recent_turns
  ├── MemoryStore.search_memory()  → relevant_memories (语义) or parse_facts() (回退)
  ├── TurnInput.visual_context     → visual_context
  └── ToolPool.names               → tool_names
  │
  ▼
ContextSnapshot (frozen, immutable)
  │
  ▼
OllamaContextRenderer.render()
  │
  ▼
RenderedContext
  ├── system_prompt: str
  └── messages: list[dict]       ← 直接传给 OllamaProvider
```

---

## 五、事件系统

组件间通过事件通信，不直接调用。使每个组件可独立替换和测试。

### 5.1 事件定义

```python
# ── 感知层事件 (System 1 产出) ──
class Utterance(Event):        # System 1 生成了一句回复
class UserSpeech(Event):       # 用户说了一句话 (Web Speech API)
class VisualScene(Event):      # 视觉场景描述更新
class Silence(Event):          # 持续静默

# ── 路由事件 ──
class IntentDecision(Event):   # 路由决策: fast | slow

# ── 推理层事件 (System 2 产出) ──
class ThinkingStarted(Event):  # System 2 开始推理
class ToolExecuting(Event):    # 正在执行工具
class ReasoningDone(Event):    # System 2 完成推理

# ── 输出事件 ──
class SpeakRequest(Event):     # 请求语音输出

# ── 频道事件 ──
class ChannelMessage(Event):   # 来自外部频道的消息

# ── 系统事件 ──
class MemoryUpdated(Event):    # 长期记忆更新
class HealthCheck(Event):      # 健康检查
```

### 5.2 EventBus

```python
class EventBus:
    def on(self, event_type: Type[Event], handler: Callable) -> None:
        """订阅事件类型"""

    async def emit(self, event: Event) -> None:
        """发布事件，异步通知所有订阅者"""
```

---

## 六、状态机

### 6.1 状态转移图

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

### 6.2 GPU 占用规则

| 状态 | System 1 (MiniCPM-o) | System 2 (Qwen) | 说明 |
|------|----------------------|-----------------|------|
| LISTENING | idle | idle | 等待用户输入 |
| PERCEIVING | **推理中** | idle | 处理音频+视频帧 |
| ROUTING | idle | idle | 关键词匹配, 瞬时完成 |
| THINKING | **已暂停 (break)** | **推理中** | Qwen 做工具调用链 |
| SPEAKING | idle 或 TTS 中 | idle | 浏览器 TTS 或 Token2Wav |

**核心保障**：PERCEIVING 和 THINKING 永不同时发生。

---

## 七、Agent Core（推理核心）

### 7.1 ConversationEngine

```python
class ConversationEngine:
    """
    管理一次完整对话的 turn 生命周期。
    脱离传输层，只关心 turn 执行。
    """

    def __init__(
        self,
        tool_pool: ToolPool,
        memory: MemoryStore,
        reason_fn: Callable,
        context_builder: ContextBuilder,
        context_renderer: OllamaContextRenderer,
        max_iterations: int = 20,
    ): ...

    async def submit_message(
        self,
        text: str,
        channel: str = "cli",
        sender: str = "",
        visual_context: str = "",
        metadata: dict | None = None,
    ) -> TurnResult:
        """
        执行一个完整 turn:
        1. 构造 TurnInput
        2. ContextBuilder.build() → ContextSnapshot
        3. ContextRenderer.render() → RenderedContext
        4. 调用 reason_fn 进入 query_loop
        5. 写入记忆
        6. 返回 TurnResult
        """
        ...

    def abort(self) -> None:
        """中断当前 turn"""
        ...
```

### 7.2 query_loop

```python
@dataclass
class QueryState:
    messages: list[dict]
    tools: list[dict]
    max_iterations: int
    tools_used: list[dict]

class QueryDeps(Protocol):
    async def call_model(self, messages, tools) -> ModelResponse: ...
    async def execute_tool(self, name, args) -> str: ...

async def query_loop(
    state: QueryState,
    deps: QueryDeps,
    abort: asyncio.Event | None = None,
) -> AsyncGenerator[QueryEvent, None]:
    """
    ReAct 状态机：
    for iteration in range(max_iterations):
        → call_model()
        → if tool_calls: execute_tools → append results → continue
        → if text_only: yield QueryComplete → break
    """
```

事件类型：`IterationStart`, `ModelResponse`, `ToolCallStart`, `ToolCallDone`, `QueryComplete`, `QueryError`

---

## 八、Provider 接口

### 8.1 PerceptionProvider (System 1)

```python
class PerceptionProvider(Protocol):
    async def start(self, system_prompt: str) -> None: ...
    async def pause(self) -> None: ...
    async def resume(self) -> None: ...
    async def feed(self, audio_b64: str, frame_b64: str = "") -> PerceptionResult: ...
    async def inject_context(self, text: str) -> None: ...
    async def reset(self) -> None: ...
```

### 8.2 ReasoningProvider (System 2)

```python
class ReasoningProvider(Protocol):
    async def reason(
        self,
        message: str,
        context: RenderedContext,
        tools: list[dict],
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = 20,
    ) -> ReasoningResult: ...

    async def health(self) -> bool: ...
```

注意：`context` 参数已从旧的 `Context` 对象改为 `RenderedContext`，由 ConversationEngine 在调用前完成渲染。

---

## 九、GPUScheduler

```python
class GPUScheduler:
    @asynccontextmanager
    async def use_reasoning(self):
        """暂停 System 1 → yield → 恢复 System 1"""
        async with self._lock:
            await self._perception.pause()
            try:
                yield
            finally:
                await self._perception.resume()
```

---

## 十、IntentRouter

```python
class KeywordRouter(IntentRouter):
    TRIGGERS = ["搜索", "帮我", "记住", "提醒", "计算", "打开", "运行", "总结", "翻译", ...]

    def classify(self, text: str, visual_context: str = "") -> str:
        # 匹配触发词 → "slow"，否则 → "fast"
```

---

## 十一、MemoryStore

```python
class MemoryStore:
    """
    三层记忆系统，纯存储和检索。
    不再承担上下文组装和 prompt 渲染的职责（已拆到 Context Layer）。

    文件布局:
        memory/
        ├── SOUL.md       身份定义 (只读)
        ├── USER.md       用户画像 (只读)
        ├── MEMORY.md     长期记忆 (AI 自动写入)
        └── vectors.npz   向量索引 (可选)
    """

    def soul(self) -> str: ...
    def user_profile(self) -> str: ...
    def long_term_memory(self) -> str: ...
    def append_turn(self, role: str, text: str) -> None: ...
    def recent_turns(self, n: int = 20) -> list[dict]: ...
    def save_fact(self, fact: str) -> None: ...
    def search_memory(self, query: str, top_k: int = 5) -> list[tuple]: ...
    def parse_facts(self) -> list[str]: ...

    # 便捷别名
    def add_user(self, text: str) -> None: ...
    def add_assistant(self, text: str) -> None: ...
```

---

## 十二、ToolPool

```python
class Tool(ABC):
    """工具抽象基类，定义工具的完整协议。"""
    name: str
    description: str

    @abstractmethod
    def parameters_schema(self) -> dict: ...
    def validate_input(self, args: dict) -> None: ...
    def is_read_only(self) -> bool: ...
    def is_parallel_safe(self) -> bool: ...
    def is_enabled(self) -> bool: ...
    async def execute(self, **kwargs) -> str: ...
    def openai_schema(self) -> dict: ...


class ToolPool:
    """
    工具编排器，支持：
    - deny list 按名禁用工具
    - partition: 连续的 parallel_safe 工具分批
    - execute_batch: 安全批次用 asyncio.gather 并行
    - 结果截断 (MAX_RESULT_CHARS = 30_000)
    """

    def __init__(self, tools: list, deny: list[str] | None = None): ...
    def definitions(self) -> list[dict]: ...
    async def execute(self, name: str, args: dict) -> str: ...
    async def execute_batch(self, calls: list[dict], max_concurrency: int = 5) -> list[str]: ...
```

内置工具标注了 `read_only` / `parallel_safe`，ToolPool 自动并行调度。

---

## 十三、Channel 系统

```python
class Channel(Protocol):
    name: str
    async def start(self, bus: EventBus) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, text: str, media: bytes = None) -> None: ...
```

已实现频道：

| 频道 | 文件 | 说明 |
|------|------|------|
| CLIChannel | `channels/cli.py` | stdin/stdout，开发调试 |
| WebSocketChannel | `channels/websocket.py` | 浏览器语音+摄像头+双系统 |

---

## 十四、EdgeAgent 主编排

```python
class EdgeAgent:
    def __init__(
        self,
        perception=None,      # System 1 Provider
        reasoning=None,        # System 2 Provider
        router=None,           # IntentRouter (默认 KeywordRouter)
        tools=None,            # 工具函数列表
        memory_dir="./memory",
        channels=None,
        max_iterations=20,
        tool_deny=None,
    ):
        # 组装所有组件
        self.tool_pool = ToolPool(tools, deny=tool_deny)
        self.memory = MemoryStore(memory_dir)
        self.context_builder = ContextBuilder(memory=self.memory, tool_pool=self.tool_pool)
        self.context_renderer = OllamaContextRenderer()
        # ConversationEngine 在 run() 中创建

    async def run(self, host, port):
        # 1. 创建 ConversationEngine
        # 2. 启动 System 1 (bootstrap_prompt)
        # 3. 健康检查 System 2
        # 4. 启动 Channel
        # 5. await stop_event
```

---

## 十五、项目目录结构

```
edge_agent/                         # 框架核心 (~3200 行)
├── __init__.py                     # EdgeAgent 编排器 (274 行)
├── config.py                       # EdgeConfig 统一配置 (60 行)
├── events.py                       # Event 定义 + EventBus (168 行)
├── state.py                        # StateMachine (55 行)
├── scheduler.py                    # GPUScheduler (42 行)
├── router.py                       # IntentRouter + KeywordRouter (38 行)
├── memory.py                       # MemoryStore (215 行)
├── tools.py                        # Tool ABC + ToolPool (286 行)
├── errors.py                       # 结构化错误 (56 行)
├── api.py                          # OpenAI 兼容 API Gateway (216 行)
│
├── agent/                          # 推理核心
│   ├── query.py                    # QueryState + query_loop (226 行)
│   └── conversation.py             # ConversationEngine (122 行)
│
├── context/                        # 上下文层
│   ├── types.py                    # TurnInput, ContextSnapshot, RenderedContext (50 行)
│   ├── builder.py                  # ContextBuilder (77 行)
│   └── renderers.py                # OllamaContextRenderer (57 行)
│
├── providers/
│   ├── __init__.py                 # Protocol 定义 (64 行)
│   ├── minicpm.py                  # MiniCPMProvider (324 行)
│   └── ollama.py                   # OllamaProvider (142 行)
│
├── channels/
│   ├── cli.py                      # CLIChannel (58 行)
│   └── websocket.py                # WebSocketChannel (307 行)
│
├── tools_builtin/                  # 内置工具
│   ├── web.py                      # web_search, web_fetch
│   ├── filesystem.py               # read/write/edit/list
│   ├── shell.py                    # shell + 安全过滤
│   ├── computer.py                 # screenshot/click/type/scroll
│   └── system.py                   # memory_save
│
└── web/
    ├── index.html
    ├── style.css
    └── app.js

apps/
├── assistant.py                    # CLI 私有助手
└── second_brain.py                 # 浏览器双系统

tests/                              # 49 个自动化测试
├── test_context.py
├── test_memory.py
├── test_tools.py
├── test_events.py
└── test_ollama.py
```

---

## 十六、硬件兼容矩阵

| 硬件 | 显存 | System 1 | System 2 | 能力等级 |
|------|------|---------|---------|---------|
| **GB10 / GB200** | 128 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 27B Q4 (16GB) | L1-L5 全部 |
| RTX 4090 | 24 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 9B Q4 (6GB) | L1-L4 |
| RTX 3060 | 12 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 4B Q4 (3GB) | L1-L3 |
| Jetson Orin | 32-64 GB | MiniCPM-o 4.5 Q4 (5GB) | Qwen3.5 14B Q4 (9GB) | L1-L4 |
| Mac M4 Ultra | 192 GB | MiniCPM-o 4.5 (mlx) | Qwen3.5 27B (mlx) | L1-L5 |

---

## 十七、安全模型

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
   - 会话历史在内存中，不自动持久化

4. 工具权限
   - ToolPool 支持 deny list，禁用指定工具
   - Tool 协议包含 read_only / parallel_safe 标记
   - 结果自动截断 (30K 字符上限)
```
