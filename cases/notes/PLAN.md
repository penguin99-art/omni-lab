# Edge Agent 执行计划

> 最后更新：2026-04-02

---

## 代码现状

### 框架核心

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| EdgeAgent 编排器 | `__init__.py` | 274 | 完整 |
| EventBus | `events.py` | 168 | 完整，12 种事件类型 |
| StateMachine | `state.py` | 55 | 完整 |
| GPUScheduler | `scheduler.py` | 42 | 完整 |
| IntentRouter | `router.py` | 38 | 完整，关键词路由 |
| MemoryStore | `memory.py` | 215 | 完整，语义检索 + 关键词回退 |
| ToolPool | `tools.py` | 286 | 完整，Tool ABC + 并行编排 |
| EdgeConfig | `config.py` | 60 | 完整，环境变量配置 |
| Errors | `errors.py` | 56 | 完整 |

### Agent Core

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| ConversationEngine | `agent/conversation.py` | 122 | 完整 |
| query_loop | `agent/query.py` | 226 | 完整 |

### Context Layer

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| TurnInput, ContextSnapshot, RenderedContext | `context/types.py` | 50 | 完整 |
| ContextBuilder | `context/builder.py` | 77 | 完整 |
| OllamaContextRenderer | `context/renderers.py` | 57 | 完整 |

### Providers

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| Protocol 定义 | `providers/__init__.py` | 64 | 完整 |
| OllamaProvider | `providers/ollama.py` | 142 | 完整 |
| MiniCPMProvider | `providers/minicpm.py` | 324 | 完整 |

### Channels

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| CLIChannel | `channels/cli.py` | 58 | 完整 |
| WebSocketChannel | `channels/websocket.py` | 307 | 完整 |

### 其他

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| API Gateway | `api.py` | 216 | 可用，独立路径（未接入 ConversationEngine） |
| 8 个内置工具 | `tools_builtin/` | 315 | 完整 |
| Web 前端 | `web/` | — | 完整 |

**总代码量：~3200 行 Python，49 个自动化测试。**

---

## 已完成的里程碑

### ✅ Phase 0：基础框架

- [x] EventBus + StateMachine + GPUScheduler + IntentRouter
- [x] MemoryStore（Markdown + 语义检索 + 关键词回退）
- [x] ToolPool（Tool ABC + FunctionTool + 并行编排 + deny list）
- [x] OllamaProvider（ReAct 循环 + 重试 + abort）
- [x] MiniCPMProvider（HTTP 对接 llama-server）
- [x] CLIChannel + WebSocketChannel
- [x] Web 前端（暗色 UI、麦克风、字幕、摄像头、状态指示）
- [x] API Gateway（OpenAI 兼容）
- [x] pytest 基础测试

### ✅ Phase 1：架构优化（v0.2.0 → v0.3.0）

- [x] 统一异步模型（消除多 event loop）
- [x] WebSocket 消除行为分叉（统一走 EventBus）
- [x] Protocol 签名统一 + 集中配置管理 + 优雅关闭
- [x] query_loop 状态机 + QueryDeps 依赖注入
- [x] ConversationEngine 会话管理
- [x] Abort/Cancel 支持
- [x] 结构化错误类型

### ✅ Phase 2：上下文层（v0.3.1）

- [x] Context Layer 三层解耦（Builder → Snapshot → Renderer）
- [x] MemoryStore 精简（移除 prompt 渲染逻辑）
- [x] ConversationEngine 接入 Context Pipeline
- [x] ReasoningProvider 签名改为 RenderedContext
- [x] EdgeAgent 集成（System 1 bootstrap + System 2 reasoning）
- [x] 测试更新（49 passed，0 linter errors）
- [x] CLI 三轮对话验证通过

---

## 下一步

### Phase 3：统一 Runtime（收口）

**目标**：所有路径（CLI、WebSocket、API Gateway）都走同一套 ConversationEngine + ContextBuilder。

| 任务 | 说明 | 优先级 |
|------|------|--------|
| API Gateway 切换到 ConversationEngine | `api.py` 目前直接调 Ollama，应复用统一 runtime | P0 |
| WebSocket 路径确认 | 确保 Second Brain 的 System 2 请求也走 ConversationEngine | P0 |
| 新增 Provider 时的 Renderer 扩展点 | 验证加一个新 Renderer 的开发体验 | P1 |

### Phase 4：硬件验证

| 任务 | 说明 |
|------|------|
| Qwen3.5 27B tok/s 基准 | prompt eval rate + eval rate + 显存 |
| MiniCPM-o 语音延迟 | 说完话到开始回复的延迟（目标 < 2 秒） |
| 双模型显存分布 | MiniCPM-o + Qwen3.5 + embedding 模型 |
| 30 分钟连续对话稳定性 | 不崩、不 OOM、不劣化 |

### Phase 5：稳定性 + 部署

| 任务 | 说明 |
|------|------|
| 24/7 运行测试 | health check + cron + 长时间监控 |
| 错误恢复 | Provider 自动重连、进程级 watchdog |
| systemd service | 设备开机自启 |
| Docker Compose | 容器化部署（可选） |

### Phase 6：体验打磨

| 任务 | 说明 |
|------|------|
| 记忆压缩 / 每日合并 | 记忆量大后的上下文管理 |
| 主动提醒 | 基于 cron + 记忆触发主动通知 |
| Auto-compact | 消息超 token 限制时自动压缩历史 |
| 语音交互优化 | 流式 TTS、打断响应速度 |

---

## 不做的事（v1 排除）

| 不做 | 原因 |
|------|------|
| Skills 框架 / 第三方技能市场 | 还没有用户，做生态为时过早 |
| 端云协同路由 | 与"数据不出设备"矛盾 |
| Telegram / 飞书频道 | 分散精力，v1 只做 Web + CLI |
| NanobotProvider | 文档设计了但没有实际需求 |
| LLMRouter V2 | 关键词路由够用 |
| JARVIS 式复杂 UI | 开发成本高，日常不看 |

---

## 主力模型

**Qwen3.5 27B（qwen3.5:27b）**，不再做模型选型对比。

---

## 可借用的现有工具

| 工具 | 用途 | 状态 |
|------|------|------|
| **Ollama** | 模型推理 + OpenAI 兼容 API | 已在用 |
| **llama.cpp (omni)** | MiniCPM-o 推理 + 全双工语音 | 已在用 |
| **sentence-transformers** | 记忆向量化 | 已在用 |
| **systemd** | 进程管理 | 待部署 |
| **Docker Compose** | 容器化 | 待评估 |
