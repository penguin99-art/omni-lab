# AI Workstation Agent — 思考过程与设计文档

> 记录从 "做一个语音助手框架" 到 "把自己的工作机变成 agent-native 环境" 的完整思考路径。

---

## 一、起点：审视 edge_agent

最初的项目是 `edge_agent`——一个端侧私有 AI 助手框架，双系统认知架构（System 1 快感知 + System 2 慢推理），~3200 行 Python，49 个测试。

做了什么：
- EventBus + StateMachine + GPUScheduler
- ContextBuilder → ContextSnapshot → RenderedContext 三层上下文解耦
- ConversationEngine + query_loop 状态机
- Tool ABC + ToolPool 并行编排
- OllamaProvider / MiniCPMProvider
- CLI / WebSocket 频道
- OpenAI 兼容 API Gateway

架构很精致。但跑起来之后，一个问题浮出水面：

**这个东西到底在给谁解决什么问题？**

---

## 二、连续追问，逐层剥皮

### "这值得做吗？"

直面现实：

1. **做的是胶水层。** 核心价值（模型推理、ASR、TTS）全是别人的。编排代码可替代性极高。
2. **架构精致 ≠ 产品亮点。** ContextBuilder 三层解耦，用户感知不到。49 个测试，没有外部用户。
3. **过早的工程优化。** 产品还没找到一个真实用户愿意每天用的场景，就在打磨内部架构。

### "一键 LLM → 语音助手"方向呢？

搜索后发现至少 7 个开源项目在做：
- **Pipecat** (10,900+ stars) — Python 语音 agent 编排
- **LiveKit Agents** — 工业级 WebRTC + agent
- **OpenVoiceUI** — 自托管语音 AI 工作台
- **Local Voice AI** — 全栈 Docker 化 ASR→LLM→TTS
- **Vox** — Rust 写的本地语音助手
- **Bolna** — 端到端语音 agent 平台
- **Stimm** — 双 agent 架构（跟我的 System 1/2 几乎一样）

结论：**赛道拥挤，没有竞争优势。**

### "那什么才值得做？"

回到第一性原理，问自己三个问题：

1. **我每天想用 AI 帮我做什么？** → 不是语音助手，是一个真正的工作伙伴。安排事情、对齐目标、给方案、有记忆。
2. **现有工具为什么不行？** → ChatGPT 无记忆、Siri 太蠢、Cursor 只管代码、所有工具都是"你问它答"。
3. **如果做到了值多少钱？** → 如果真的好用，愿意全天候跑在工作机上。

### "这种东西有人做了吗？"

搜索发现：
- **Hermes Agent** — 开源自治 agent，持久记忆，2200+ stars
- **OpenClaw** — 本地 AI 助手，MEMORY.md（跟我一模一样），cron 主动性
- **Thoth** — 23 工具 + 知识图谱
- **AI Chief of Staff 产品** — Alyna / Anna / Lucien，云端 SaaS

它们的共同问题：**太重了。** 多频道网关、40+ 工具、sub-agent 系统、复杂部署。做给所有人用的通用框架，装配了大量你不需要的东西。

### "Agent Computer 思路呢？"

更深一层的思考：

```
现在：  人 → 鼠标/键盘 → GUI → 应用 → 数据
Hack：  人 → AI → 截屏 → 点击鼠标（假装自己是人）
正确：  人（掌握者）→ 意图 → Agent → 系统 API → 数据
```

关键洞察：**Linux 本身就是半个 agent-friendly 系统。** 每个功能都有 CLI / API。不需要重新设计 OS，只需要加一层 agent shell。

但这需要分步走——先验证"每天跟 AI 协作"这个习惯本身是不是成立的。

---

## 三、结论：先做最小的事

停止做框架。做一个**只给自己用的、200 行的、明天就能用的东西**。

核心原则：
- **先验证习惯，再写代码**
- **不做产品，做自己的工具**
- **状态透明**：所有状态是你能读的 Markdown
- **极简**：不要 EventBus、不要 Provider 协议、不要 ContextBuilder

---

## 四、设计：ai workstation agent

### 4.1 核心理念

```
人 = 掌握者（给方向、做决策）
AI = 操作者（执行、规划、记忆）
```

AI 不只是回答问题。它是这台机器的操作者。它可以运行命令、读写文件、记住信息、规划每日任务。

### 4.2 状态模型

```
~/.ai/
├── GOALS.md           用户手写的目标（周/月/长期）
├── MEMORY.md          AI 自动积累的持久记忆
└── 2026-04-02.md      每日记录（AI 生成 + 追加）
```

全是 Markdown。人随时可读、可改。没有数据库，没有二进制状态。

### 4.3 三种模式

| 模式 | 触发 | AI 做什么 |
|------|------|----------|
| `morning` | 早上手动 or cron | 读目标 + 昨天记录 → 生成今日计划 → 主动提醒 |
| `chat` | 随时 | 带着目标/记忆/今日上下文对话，随时执行操作 |
| `evening` | 晚上手动 or cron | 总结今天 → 提取记忆 → 写入日志 |

### 4.4 工具集

只有 5 个工具，覆盖"操作这台机器"的核心能力：

| 工具 | 做什么 |
|------|-------|
| `shell` | 执行任意命令（带安全过滤） |
| `read_file` | 读取文件 |
| `write_file` | 写入文件 |
| `memory_save` | 追加事实到 MEMORY.md |
| `update_today` | 追加内容到今日文件 |

有了 shell，理论上可以做任何事。其他工具只是让常用操作更明确。

### 4.5 System Prompt 设计

prompt 在每次对话开始时动态构建：

```
基础角色设定（你是操作者，用户是掌握者）
+ 当前时间 / 工作目录 / 状态目录
+ GOALS.md 内容
+ MEMORY.md 内容
+ 今日文件内容
+ 模式特定指令（morning/evening/chat）
```

没有复杂的 ContextBuilder。一次字符串拼接。够用就行。

### 4.6 推理循环

```python
for _ in range(max_iterations):
    response = ollama.chat(model, messages, tools)
    if no tool_calls:
        return response.text
    for each tool_call:
        execute tool → append result
```

20 行代码。没有 QueryState / QueryDeps / async generator。
因为这不是框架，这是工具。

---

## 五、代码结构

```
ai/
├── __init__.py        包标记
├── __main__.py        入口 + 交互循环 + morning/evening 分发
├── engine.py          推理循环（ollama.chat + tool call 处理）
├── prompts.py         system prompt 构建
├── tools.py           工具定义 + 执行逻辑
└── run.py             独立入口（symlink 到 ~/.local/bin/ai）
```

总计约 250 行。无外部框架依赖（仅 `ollama` SDK）。

---

## 六、与 edge_agent 的关系

| 维度 | edge_agent | ai/ |
|------|-----------|-----|
| 定位 | 通用框架 | 个人工具 |
| 用户 | 所有人 | 我自己 |
| 代码量 | ~3200 行 | ~250 行 |
| 依赖 | EventBus, StateMachine, Provider, ToolPool... | ollama SDK |
| 上下文 | ContextBuilder → Snapshot → Renderer | 字符串拼接 |
| 推理 | query_loop async generator + QueryDeps | 20 行 for 循环 |
| 状态 | MemoryStore + 内存 turns | ~/.ai/ 目录下的 Markdown |
| 价值 | 学到了架构设计 | 每天实际在用 |

edge_agent 不是白做的。做过 ContextBuilder 才知道"其实字符串拼接就够了"。做过 ToolPool 才知道"5 个工具比 40 个好用"。做过 Provider 协议才知道"直接调 ollama.chat 最简单"。

**过度设计是通往简洁的必经之路。**

---

## 七、下一步

### 验证阶段（本周）

1. 每天早上 `ai morning`，看计划有没有用
2. 白天随手 `ai` 对话，看交互是否自然
3. 晚上 `ai evening`，看总结是否准确
4. 三天后评估：这个习惯是否成立？

### 如果习惯成立

- 加 cron 自动化 morning/evening
- 加桌面通知（早上弹出今日计划）
- 逐步扩展工具（git 操作、项目管理）
- 考虑 agent 主动性（检测到异常时提醒）

### 如果习惯不成立

- 停下来想：是 AI 能力不够，还是我根本不需要这个？
- 不要继续写代码来"修复"一个不存在的需求

---

## 八、核心教训

1. **先找问题，再写代码。** 不是反过来。
2. **先验证习惯，再自动化。** 手动做三天，再决定要不要写脚本。
3. **给自己用的工具不需要架构。** 200 行 > 3200 行，如果 200 行解决了问题。
4. **过度设计的价值在于你走过了。** 不走过，你不会知道简单方案为什么好。
5. **数据主权是真正的差异化。** 当 AI 要记住你的一切时，"数据在谁手里"不是功能，是底线。
