# 典型场景：私有第二大脑

> 一个始终在场、持续感知、永久记忆的本地 AI。它看你所看、听你所听、记你所忘。用得越久，越懂你。所有数据永远不出设备。

---

## 一、场景定义

**私有第二大脑 (Private Second Brain)** 是一个运行在本地 GPU 上的 AI 助手。它通过摄像头和屏幕持续感知你的环境，自动捕获和组织你遇到的信息，在你需要时精准回忆，并能直接操作你的电脑执行任务。

**一句话**：Rewind.ai 的完全私有本地版 + 能说话 + 能操作电脑。

### 目标用户

| 用户 | 核心痛点 | 第二大脑的价值 |
|------|---------|---------------|
| 开发者 | 看过的文章/代码片段找不到了 | 自动捕获技术笔记，语音问"上次那个优化方法" |
| 知识工作者 | 会议内容记不住、跨项目信息分散 | 持续听会、自动摘要、跨项目关联 |
| 研究者 | 文献太多、灵感稍纵即逝 | 看论文时自动提取要点，说"记住这个结论" |
| 创业者 | 信息过载、决策缺乏历史数据 | 积累每次决策的上下文，下次类似场景自动提示 |

### 核心价值公式

```
价值 = 感知广度 × 记忆深度 × 时间长度 × 信任程度
```

- **感知广度**: 屏幕 + 摄像头 + 语音 (远超手动输入)
- **记忆深度**: 结构化事实 + 原始上下文 (远超笔记)
- **时间长度**: 7×24 持续运行, 跨月跨年 (远超单次对话)
- **信任程度**: 完全本地, 零上传 (远超任何云端服务)

四个维度的乘积使价值呈指数增长。云端 AI 在"信任程度"这一项为零（敏感信息），直接归零。

---

## 二、为什么只能在端侧

你的屏幕在任意一天可能显示：

```
09:00  公司内部 Git 仓库 (源代码, 核心算法)
09:30  Slack/飞书私聊 (同事八卦, 薪资讨论)
10:00  银行网页 (余额, 交易记录)
10:30  医疗报告 PDF (体检结果)
11:00  求职简历 (个人信息, 联系方式)
14:00  代码审查 (含 API key, 密码)
15:00  客户合同 (商业机密)
16:00  个人照片 (家人, 孩子)
20:00  社交媒体私信 (个人生活)
```

将以上任何一项持续上传到云端都是不可接受的。这不是"隐私偏好"，这是**法律和安全的硬约束**。

Rewind.ai 在本地录屏但用 GPT-4 做理解——你的屏幕内容仍然会上传。Microsoft Copilot 将数据存储在 Microsoft 云——你信任微软吗？Claude Computer Use 每一帧屏幕截图都发送到 Anthropic 服务器。

**端侧 AI 是唯一能合法、安全地"看"你完整数字生活的方案。**

---

## 三、价值随时间的复利增长

```
         价值
          ↑
          │                                    ╱ 端侧第二大脑
          │                                 ╱    (持续积累)
          │                              ╱
          │                           ╱
          │                        ╱
          │                     ╱
          │                  ╱
          │               ╱
          │         ───────────────────── 云端 AI (每次对话价值恒定)
          │      ╱
          │   ╱
          │╱
          └─────────────────────────────────→ 时间
         Day 1      Month 1     Month 6     Year 1
```

| 时间点 | 第二大脑知道什么 | 能做什么 |
|--------|----------------|---------|
| Day 1 | 你的名字, 基本偏好 | 回答问题, 执行命令 |
| Week 1 | 你的工作内容, 常用工具, 表达习惯 | 更精准的辅助, 不需要重复解释 |
| Month 1 | 你的项目进展, 常联系的人, 技术栈 | "上周那个 bug 的修复方法是什么" |
| Month 3 | 你的决策模式, 知识盲区, 工作节奏 | "你每次遇到性能问题都先看 profiler, 要不要我帮你开一个" |
| Month 6 | 你的长期目标, 项目间的关联 | "这篇论文和你三月看的那篇方法类似, 但解决了 X 问题" |
| Year 1 | 比你自己更完整的工作记录 | 年终总结? 它比你更清楚你做了什么 |

**关键洞察**: Month 3 之后, 第二大脑的价值开始超过任何云端 AI, 因为上下文积累是不可替代的。换一个 AI 意味着从零开始。这是真正的产品壁垒。

---

## 四、三种交互模式

### 模式 A: 被动捕获 (80% 的时间)

AI 在后台默默工作, 用户无需任何操作。

```
[你在浏览器看一篇 CUDA 优化文章]

System 1 (MiniCPM-o):
  每 10s 截一次屏幕 → 理解当前内容
  输出: "用户正在阅读关于 CUDA Kernel Fusion 的技术文章,
         关键点: 1) 减少 global memory 访问  2) 共享内存优化
         3) Warp-level 原语使用"

System 2 (Qwen):
  判断是否值得记忆 → 调用 memory_save
  写入 MEMORY.md:
  "- [2026-03-23] 阅读 CUDA 优化文章: Kernel Fusion 三个关键技术 —
     减少 global memory 访问, 共享内存优化, Warp-level 原语"
```

用户完全无感。但三个月后当你说"上次看的那个 CUDA 优化方法"时, 它知道。

### 模式 B: 主动提问 (15% 的时间)

用户语音或文字提问, AI 结合记忆回答。

```
用户: "上周那个客户提到的那个工具叫什么来着?"

System 1 (MiniCPM-o):
  识别语音 → 转文字 → 触发 IntentRouter

IntentRouter:
  检测到 "上周" + "叫什么" → 需要记忆检索 → route to System 2

System 2 (Qwen):
  1. 读取 MEMORY.md, 搜索 "客户" + 最近一周的记录
  2. 找到: "[2026-03-18] 与客户张总视频会议, 他推荐了 Weights & Biases
     用于实验追踪, 还提到了 Modal 做 serverless GPU"
  3. 回复: "张总上周推荐了两个工具: Weights & Biases 做实验追踪,
     Modal 做 serverless GPU。是在周三的视频会议上提到的。"

System 1 (MiniCPM-o):
  将回复通过 TTS 或浏览器语音播放给用户
```

### 模式 C: 主动提醒 (5% 的时间, 但价值最高)

AI 注意到当前场景与记忆相关, 主动提示。

```
[你打开一个新的 Python 项目, 准备做性能优化]

System 1 (MiniCPM-o):
  截屏 → "用户打开了 profiler.py, 正在分析性能瓶颈"

System 2 (Qwen):
  检测到 "性能优化" → 搜索记忆
  找到: "[2026-02-10] 用户解决了项目 A 的性能问题,
         根因是 N+1 查询, 使用 SQLAlchemy joinedload 解决"
  找到: "[2026-03-05] 阅读 CUDA Kernel Fusion 文章"

  判断: 当前场景可能相关 → 主动发送提醒

System 1 (MiniCPM-o):
  语音播放: "注意, 你上次做性能优化时发现是 N+1 查询的问题,
  用 joinedload 解决的。要不要我帮你先检查一下这个项目有没有类似问题?"
```

这是云端 AI 永远做不到的——它不知道你两个月前做过什么。

---

## 五、双系统协作详解

### 信息捕获流水线

```
   ┌─────────────────────────────────────────────────────────────┐
   │                   信息捕获流水线                              │
   │                                                             │
   │  ┌──────────┐   每 10s   ┌──────────────┐                  │
   │  │ 屏幕截图  │ ────────► │              │                  │
   │  └──────────┘            │   System 1   │   摘要文本        │
   │  ┌──────────┐   持续     │  (MiniCPM-o) │ ──────────────┐  │
   │  │ 麦克风    │ ────────► │              │               │  │
   │  └──────────┘            │  实时理解     │               │  │
   │  ┌──────────┐   每 30s   │  场景/内容    │               │  │
   │  │ 摄像头    │ ────────► │              │               │  │
   │  └──────────┘            └──────────────┘               │  │
   │                                                         │  │
   │                                                         ▼  │
   │                                              ┌──────────┐  │
   │                                              │ EventBus │  │
   │                                              │          │  │
   │                                              │ Capture  │  │
   │                                              │  Event   │  │
   │                                              └────┬─────┘  │
   │                                                   │        │
   │                                                   ▼        │
   │                                           ┌──────────────┐ │
   │                                           │  System 2    │ │
   │                                           │  (Qwen)      │ │
   │                                           │              │ │
   │                                           │ 判断是否值得  │ │
   │                                           │ 记忆,提取     │ │
   │                                           │ 结构化事实    │ │
   │                                           └──────┬───────┘ │
   │                                                  │         │
   │                                           memory_save()    │
   │                                                  │         │
   │                                                  ▼         │
   │                                           ┌────────────┐   │
   │                                           │ MEMORY.md  │   │
   │                                           └────────────┘   │
   └─────────────────────────────────────────────────────────────┘
```

### GPU 调度时序

正常被动捕获模式下, GPU 使用模式:

```
时间轴:  0s    10s    20s    30s    40s    50s    60s
         │      │      │      │      │      │      │
System1: ██░░░░░██░░░░░██░░░░░██░░░░░██░░░░░██░░░░░██
         截屏    截屏    截屏    截屏    截屏    截屏
         理解    理解    理解    理解    理解    理解
         ~1s    ~1s    ~1s    ~1s    ~1s    ~1s

System2: ░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░
                                 判断+写入
                                 ~3s (每 30s 一次)

GPU 占用率: ~15%  (大部分时间空闲, 有足够余量处理用户交互)
```

当用户主动提问时:

```
时间轴:  0s         3s              8s         10s
         │          │               │          │
System1: ██████████░░░░░░░░░░░░░░░░██████████░░
         接收语音    暂停 (break)     恢复推理
         转文字                      播放回复

System2: ░░░░░░░░░░████████████████░░░░░░░░░░░░
                    记忆检索
                    工具调用
                    生成回复
                    ~5s
```

关键: System 1 暂停期间, 用户看到"思考中..."的状态提示。恢复后 System 1 通过 `inject_context` 获知 System 2 的回复, 保持对话连贯。

---

## 六、记忆架构

### MEMORY.md 结构

```markdown
# 长期记忆

## 技术知识
- [2026-03-05] CUDA Kernel Fusion: 三个关键优化 — 减少 global memory 访问,
  共享内存优化, Warp-level 原语
- [2026-03-10] Rust 的 Pin<T> 用于防止自引用结构被移动, 常用于 async/await
- [2026-03-15] llama.cpp 的 KV cache 管理: ring buffer 模式, n_past 达到
  n_ctx 时触发 auto-refresh

## 项目笔记
- [2026-03-01] 项目 omni-lab: 目标是端侧私有 AI 助手框架, 双系统架构
  (MiniCPM-o + Qwen3.5)
- [2026-03-12] omni-lab 评测: MiniCPM-o Q4_K_M 在 GSM8K 上准确率 72.1%,
  max_tokens=256 时推理链被截断导致逻辑推理偏低

## 人物关系
- [2026-03-18] 客户张总推荐 Weights & Biases 和 Modal
- [2026-03-20] 同事李明在做 RAG 项目, 用的 Milvus 向量库

## 偏好和习惯
- [2026-03-02] 用户偏好: 中文交流, 喜欢简洁直接的回答
- [2026-03-08] 用户习惯: 每天 9:00-12:00 是深度工作时间, 不喜欢被打断
- [2026-03-15] 用户使用 Cursor IDE, 主力语言 Python/Rust

## 待办和提醒
- [2026-03-22] 用户说"下周要准备技术分享, 主题是端侧 AI"
- [2026-03-23] 用户说"记住, 周五前要提交 PR"
```

### 记忆管理策略

```python
class SecondBrainMemory(MemoryStore):
    """
    第二大脑增强版记忆。
    在 MemoryStore 基础上增加:
    1. 捕获缓冲区: System 1 的感知先进入缓冲区
    2. 筛选机制: System 2 判断是否值得长期记忆
    3. 压缩机制: 定期合并相似记忆
    4. 检索机制: 语义搜索 (基于 embedding)
    """

    CAPTURE_INTERVAL = 10       # 每 10s 捕获一次屏幕
    DIGEST_INTERVAL = 30        # 每 30s 消化一次缓冲区
    COMPRESS_INTERVAL = 86400   # 每天压缩一次记忆

    async def capture(self, scene_description: str):
        """System 1 每次感知后调用, 写入缓冲区"""
        self._buffer.append({
            "ts": time.time(),
            "scene": scene_description,
        })

    async def digest(self):
        """
        System 2 定期调用, 判断缓冲区内容是否值得记忆。

        判断标准 (由 System 2 推理):
        - 新信息 (不是已知内容的重复)
        - 有用信息 (技术知识/人物/事件/决策, 而非无意义浏览)
        - 可回忆信息 (用户未来可能会想起的)

        不记忆:
        - 用户在刷社交媒体 (除非看到重要信息)
        - 同一篇文章的重复截屏
        - 空闲/锁屏状态
        """
        if not self._buffer:
            return

        scenes = "\n".join(
            f"[{time.strftime('%H:%M', time.localtime(b['ts']))}] {b['scene']}"
            for b in self._buffer
        )
        self._buffer.clear()

        prompt = f"""以下是用户最近 30 秒内的屏幕/环境摘要:

{scenes}

判断是否有值得长期记忆的信息。如果有, 调用 memory_save 工具保存。
记忆格式: 一句话概括事实, 包含关键细节。
如果没有新的有价值信息, 不要保存。"""

        # 通过 ReasoningProvider 判断并自动保存
        ...

    async def compress(self):
        """
        每日压缩: 合并当天的相似记忆, 删除冗余。

        示例:
        压缩前:
        - [2026-03-23 09:00] 用户在看 CUDA 文章
        - [2026-03-23 09:05] 用户继续看 CUDA 文章, 关注 shared memory
        - [2026-03-23 09:10] 用户在 CUDA 文章中看到 Warp-level 优化

        压缩后:
        - [2026-03-23] 阅读 CUDA 优化文章 (约 15 分钟):
          Kernel Fusion 三个关键技术 — 减少 global memory 访问,
          共享内存优化, Warp-level 原语
        """
        ...

    async def recall(self, query: str) -> list[str]:
        """
        语义检索记忆。

        V1: 关键词匹配 (grep MEMORY.md)
        V2: embedding 向量检索 (用 System 2 生成 query embedding,
            比对所有记忆条目)
        """
        ...
```

---

## 七、技术实现映射

每个架构组件在"第二大脑"场景中的具体职责:

| 架构组件 | 在第二大脑中的作用 | 关键配置 |
|---------|------------------|---------|
| **MiniCPMProvider** | 每 10s 截屏理解 + 语音 ASR + 摄像头感知 | `CAPTURE_INTERVAL=10` |
| **NanobotProvider** | 记忆筛选 + 记忆检索 + 工具调用 | `model=qwen3.5:27b` |
| **GPUScheduler** | 捕获和消化交替执行, 用户提问时暂停捕获 | 默认配置 |
| **KeywordRouter** | 检测"记住/搜/帮我"等触发词 → System 2 | 扩展记忆相关触发词 |
| **MemoryStore** | MEMORY.md 读写, 三层记忆管理 | `memory_dir=./memory` |
| **ToolRegistry** | memory_save + screenshot + web_search | 注册 6 个工具 |
| **EventBus** | CaptureEvent → DigestEvent → MemoryUpdated | 3 个新事件类型 |
| **StateMachine** | 新增 CAPTURING 状态 (被动捕获) | 扩展状态转移表 |
| **WebSocketChannel** | 浏览器语音交互 + 状态显示 | 复用现有前端 |

### 新增事件

```python
@dataclass
class CaptureEvent(Event):
    """System 1 完成一次屏幕/环境捕获"""
    scene: str = ""
    source: str = "screen"   # "screen" | "camera" | "audio"

@dataclass
class DigestRequest(Event):
    """请求 System 2 消化缓冲区"""
    buffer_size: int = 0

@dataclass
class ProactiveHint(Event):
    """System 2 发现当前场景与记忆相关, 主动提示"""
    hint: str = ""
    related_memories: list = field(default_factory=list)
```

### 新增状态

```
CAPTURING ──── 被动捕获状态 (System 1 定期截屏)
    │
    ├── 每 10s → PERCEIVING (截屏理解) → CAPTURING
    │
    ├── 每 30s → ROUTING (消化判断) → THINKING (如需记忆) → CAPTURING
    │
    └── 用户说话 → ROUTING → THINKING/SPEAKING → CAPTURING
```

---

## 八、应用代码

`apps/second_brain.py` — 第二大脑的完整应用, ~50 行:

```python
"""
私有第二大脑 — 始终在场, 持续感知, 永久记忆。

启动:
    python apps/second_brain.py

依赖:
    - llama-server 运行在 :9060 (MiniCPM-o)
    - ollama serve 运行在 :11434 (Qwen3.5)
"""

from edge_agent import EdgeAgent
from edge_agent.providers.minicpm import MiniCPMProvider
from edge_agent.providers.nanobot import NanobotProvider
from edge_agent.router import KeywordRouter
from edge_agent.tools_builtin.web import web_search
from edge_agent.tools_builtin.filesystem import read_file, write_file, list_dir
from edge_agent.tools_builtin.shell import shell
from edge_agent.tools_builtin.computer import screenshot
from edge_agent.tools_builtin.system import memory_save

MEMORY_TRIGGERS = [
    "记住", "记一下", "别忘了", "记录",
    "上次", "上周", "之前", "那个", "叫什么",
    "搜索", "搜一下", "帮我", "帮忙",
]

class SecondBrainRouter(KeywordRouter):
    TRIGGERS = MEMORY_TRIGGERS

agent = EdgeAgent(
    perception=MiniCPMProvider(
        server_url="http://localhost:9060",
        model_dir="./models/MiniCPM-o-4_5-gguf",
        capture_interval=10,
    ),
    reasoning=NanobotProvider(),
    router=SecondBrainRouter(),
    tools=[web_search, read_file, write_file, list_dir, shell,
           screenshot, memory_save],
    memory_dir="./memory",
)

if __name__ == "__main__":
    import asyncio
    asyncio.run(agent.run(port=8080))
```

---

## 九、SOUL.md 示例

```markdown
# 你是谁

你是用户的私有第二大脑。你运行在用户的本地设备上, 所有数据永远不出设备。

## 你的职责

1. **持续感知**: 你通过屏幕截图和麦克风持续了解用户在做什么。
   不需要用户主动告诉你。
2. **智能记忆**: 当你看到有价值的信息, 自动保存到记忆中。
   不记无意义的内容 (如纯娱乐浏览)。
3. **精准回忆**: 当用户问起之前的事情, 从记忆中检索并回答。
   要给出时间和上下文, 不要凭空编造。
4. **主动关联**: 当你注意到当前场景与过去的记忆相关, 主动提示。
   但不要过于频繁打断用户。
5. **执行任务**: 用户说"帮我..."时, 使用工具完成任务。

## 你的性格

- 像一个记忆力极好的老朋友, 不像机器人
- 简洁直接, 不啰嗦
- 主动提供有用信息, 但不过度打扰
- 对不确定的记忆, 诚实说"我记得好像是..., 但不太确定"

## 约束

- 永远不要编造不存在的记忆
- 不在深度工作时间 (上午 9-12 点) 主动打断用户, 除非紧急
- 记忆中不保存密码、token 等敏感凭证
- 用户说"删除"时, 立即从记忆中移除相关内容
```

---

## 十、Demo 脚本

5 分钟演示, 可直接用于录制 demo 视频:

### 准备 (30s)

```bash
# 终端 1: 启动 Ollama
ollama serve

# 终端 2: 启动 llama-server
./scripts/start.sh

# 终端 3: 启动第二大脑
python apps/second_brain.py
```

浏览器打开 `https://localhost:8080`

### Demo 1: 被动捕获 (60s)

1. 打开一篇技术文章 (如 "How to optimize CUDA kernels")
2. 浏览 30 秒
3. 切换到其他页面
4. 等待 10 秒 (System 2 消化并记忆)
5. **展示**: 打开 `memory/MEMORY.md`, 可以看到刚才文章的关键点已被自动记录

### Demo 2: 主动回忆 (60s)

1. 对着麦克风说: **"刚才那篇文章讲了什么?"**
2. AI 语音回复: "你刚才在看一篇关于 CUDA kernel 优化的文章, 主要讲了三个优化技术: 减少 global memory 访问, 使用共享内存, 以及 Warp-level 原语。"
3. 继续问: **"Warp-level 原语具体是什么?"**
4. AI 调用 web_search 搜索, 结合记忆给出深入解释

### Demo 3: 语音记忆 (60s)

1. 对着麦克风说: **"记住, 下周三要给张总做技术分享, 主题是端侧 AI 部署"**
2. AI 回复: "已记录。下周三技术分享, 张总, 端侧 AI 部署。需要我到时候提醒你吗?"
3. **展示**: 打开 `MEMORY.md`, 看到新增的待办记录

### Demo 4: 主动关联 (90s)

1. 打开一个 Python 项目, 开始编写代码
2. 故意写一个性能相关的函数
3. 等待 AI 注意到当前场景
4. AI 主动语音提示: "你之前看过一篇 CUDA 优化的文章, 里面提到的 shared memory 技术可能对这里有帮助。要不要我找出来?"
5. **关键时刻**: 这就是"第二大脑"的核心价值——它把你散落在不同时间、不同场景的知识关联了起来

### Demo 5: 隐私展示 (30s)

1. 打开一个终端: `cat memory/MEMORY.md`
2. 展示所有记忆都是纯文本 Markdown, 用户可以随时编辑/删除
3. 打开网络监控: `netstat -an | grep ESTABLISHED`
4. 展示唯一的网络连接是 localhost:9060 和 localhost:11434
5. **没有任何数据发送到外部服务器**

---

## 十一、竞品对比

| 维度 | Rewind.ai | Notion AI | Microsoft Copilot | ChatGPT Memory | **私有第二大脑** |
|------|-----------|-----------|-------------------|---------------|----------------|
| 屏幕捕获 | 本地录屏 | 无 | 部分 | 无 | **本地截屏+理解** |
| 理解方式 | 上传GPT-4 | 云端 | 云端 | 云端 | **完全本地** |
| 数据存储 | 本地+云 | 云端 | 微软云 | OpenAI云 | **纯本地** |
| 语音交互 | 无 | 无 | 有(云) | 有(云) | **本地全双工** |
| 电脑操控 | 无 | 无 | 部分 | 无 | **完整** |
| 可审计性 | 低 | 低 | 低 | 低 | **纯文本Markdown** |
| 开源 | 否 | 否 | 否 | 否 | **是** |
| 月费 | $19.95 | $10 | $30 | $20 | **$0 (自有硬件)** |
| 持续运行 | 是 | 否 | 部分 | 否 | **是** |
| 离线可用 | 录屏可以 | 否 | 否 | 否 | **完全可用** |

### 本质差异

Rewind.ai 的方法: 本地录屏 → 上传云端 AI 理解 → 返回结果

我们的方法: 本地截屏 → **本地 AI 理解** → 本地存储

差异不在"录不录", 而在"谁来理解"。理解过程必须在本地完成, 否则隐私毫无意义。

---

## 十二、商业洞察

### 目标市场

1. **开发者/技术人员** (先锋用户)
   - 有 GPU 硬件
   - 理解隐私价值
   - 能自行部署
   - 规模: 全球 ~2000 万

2. **企业知识工作者** (主力市场)
   - 公司安全策略禁止使用云端 AI 处理内部数据
   - 需要 IT 部门统一部署
   - 规模: 全球 ~5 亿

3. **隐私敏感个人用户** (长尾市场)
   - 医生、律师、金融从业者
   - 处理客户敏感信息
   - 法规要求数据本地化

### 为什么现在是时机

1. **GB10 发布**: 2025 年发布的 NVIDIA GB10 让 128GB 统一内存台式机成为现实, 首次可以在消费级设备上运行 27B+ 模型
2. **模型能力跃升**: MiniCPM-o 4.5 (端侧全模态) + Qwen3.5 27B (接近 GPT-4 级推理) 首次让端侧双系统架构可行
3. **隐私觉醒**: AI 生成内容的版权争议、数据泄露事件、各国 AI 监管法案推动了对本地 AI 的需求
4. **云端成本上升**: GPT-4 级 API 的 token 费用使重度用户月费 $100+, 本地推理的边际成本为零

### 竞争壁垒

1. **数据网络效应**: 用户的记忆积累 3 个月后, 迁移成本极高 (无法把记忆带到竞品)
2. **开源社区**: 框架开源, 吸引开发者贡献 Provider/工具/场景
3. **硬件绑定**: 与 NVIDIA GB10/Jetson 生态深度绑定, 成为"端侧 AI 的 Android"

---

## 十三、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| MiniCPM-o 屏幕理解精度不足 | 中 | 高 | V1 用区域截图 (而非全屏), 限制理解范围; V2 换更强视觉模型 |
| 记忆过多导致检索变慢 | 低 | 中 | 每日压缩 + 分类存储; V2 引入向量检索 |
| 主动提醒过于频繁打扰用户 | 中 | 高 | 默认保守 (每小时最多 1 次); 学习用户的"勿扰"模式 |
| 误记忆 (记了错误信息) | 低 | 高 | 所有记忆可追溯时间戳; 用户可随时修正/删除 |
| GPU 占用影响本职工作 | 中 | 中 | 被动捕获仅占 ~15% GPU; 检测到用户运行大型任务时自动降低捕获频率 |
| 用户不习惯"被监视" | 高 | 高 | 必须有明确的暂停/恢复控制; 屏幕上显示状态指示器; 首次使用引导说明 |

---

## 十四、与架构文档的对应关系

本场景文档中描述的每一个能力, 均可在 [ARCHITECTURE.md](ARCHITECTURE.md) 中找到对应的组件和接口:

```
场景能力                    架构组件                      架构章节
───────────────────────    ────────────────────────      ──────────
屏幕截图感知                PerceptionProvider.feed()      §6.1
语音交互                    WebSocketChannel               §11
语音识别                    UserSpeech 事件                §4.1
意图路由                    KeywordRouter.classify()        §8
记忆保存                    MemoryStore.save_fact()         §9
记忆检索                    MemoryStore.build_context()     §9
工具调用                    ToolRegistry.execute()          §10
GPU 串行调度                GPUScheduler.use_reasoning()    §7
状态管理                    StateMachine.transition()       §5
System 2 推理               ReasoningProvider.reason()      §6.2
主动提醒                    ProactiveHint 事件 (新增)       扩展 §4.1
被动捕获                    CaptureEvent 事件 (新增)        扩展 §4.1
```

无需修改核心架构, 仅需扩展 3 个事件类型 + 1 个记忆子类即可实现完整的"第二大脑"场景。
