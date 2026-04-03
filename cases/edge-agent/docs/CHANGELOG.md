# Changelog

## v0.3.1 — 上下文构建器 + 主推理链重构

> 引入独立的 Context Layer，将上下文装配、快照、渲染拆分为三步，
> 解耦 MemoryStore / Provider / ConversationEngine 的职责边界。
> 主推理链路已验证：CLI 三轮对话（闲聊 → 记忆写入 → 记忆召回）全部通过。

---

### 核心改动

#### 1. Context Layer（新增 `edge_agent/context/` 包）

**背景**：上下文组装逻辑散落在 `MemoryStore.build_context()`、`MemoryStore.build_system_prompt()` 和各 Provider 内部，职责混乱。

**After**：三层解耦：
- **`TurnInput`**：用户一次输入的结构化表示（text, channel, sender, visual_context, metadata）
- **`ContextBuilder`**：从 MemoryStore + ToolPool + TurnInput 装配 `ContextSnapshot`
- **`ContextSnapshot`**：provider 无关的 frozen 快照（identity, user_profile, recent_turns, relevant_memories, visual_context, tool_names, runtime_notes）
- **`OllamaContextRenderer`**：将 ContextSnapshot 渲染为 Ollama chat 格式的 `RenderedContext`（system_prompt + messages）

**新增文件**：
- `edge_agent/context/__init__.py`
- `edge_agent/context/types.py`
- `edge_agent/context/builder.py`
- `edge_agent/context/renderers.py`

#### 2. MemoryStore 精简

**Before**：`memory.py` 包含 `Context` dataclass、`build_context()`、`build_system_prompt()`，既是存储层又是渲染层。

**After**：移除所有上下文组装逻辑，`MemoryStore` 回归纯存储+检索：
- 移除 `Context` dataclass
- 移除 `build_context()` 和 `build_system_prompt()`
- 新增 `add_user()` / `add_assistant()` 便捷别名
- 公开 `parse_facts()` 供 ContextBuilder 调用

#### 3. ConversationEngine 接入 Context Pipeline

**Before**：`submit_message()` 直接拼字符串传给 provider。

**After**：完整管道：
```
TurnInput → ContextBuilder.build() → ContextSnapshot
         → OllamaContextRenderer.render() → RenderedContext
         → reason_fn(message, context, tools) → TurnResult
```

`ConversationEngine.__init__()` 新增 `context_builder` 和 `context_renderer` 参数。

#### 4. ReasoningProvider 签名变更

`reason()` 的 `context` 参数类型从旧的 `Context` 改为 `RenderedContext`。Provider 直接消费渲染后的 messages，不再自行组装。

#### 5. EdgeAgent 集成

- `__init__` 中创建 `ContextBuilder` 和 `OllamaContextRenderer`
- `run()` 中将它们传递给 `ConversationEngine`
- System 1 的初始 prompt 通过 `context_builder.bootstrap_prompt()` 生成
- 快路径（System 1 fast）使用 `memory.add_user()` 记录
- 慢路径 fallback 也使用 Context Pipeline

#### 6. ToolPool 兼容

新增 `register()` 别名，保持与旧 `ToolRegistry.register()` 调用兼容。

#### 7. 测试更新

- **新增** `tests/test_context.py`：覆盖 ContextBuilder、OllamaContextRenderer、ConversationEngine pipeline
- **更新** `tests/test_memory.py`：移除 `build_context` / `build_system_prompt` 测试，新增 `add_user` / `add_assistant` / `parse_facts` 测试
- **更新** `tests/test_ollama.py`：mock 改用 `RenderedContext`，修复 assertion 匹配

**测试结果**：49 passed，0 linter errors。

---

### 影响文件

| 文件 | 变更类型 |
|------|----------|
| `edge_agent/context/__init__.py` | 新增 |
| `edge_agent/context/types.py` | 新增 |
| `edge_agent/context/builder.py` | 新增 |
| `edge_agent/context/renderers.py` | 新增 |
| `edge_agent/memory.py` | 修改（精简） |
| `edge_agent/providers/__init__.py` | 修改（签名） |
| `edge_agent/providers/ollama.py` | 修改（消费 RenderedContext） |
| `edge_agent/agent/conversation.py` | 修改（接入 pipeline） |
| `edge_agent/__init__.py` | 修改（初始化 + 集成） |
| `edge_agent/tools.py` | 修改（register 别名） |
| `tests/test_context.py` | 新增 |
| `tests/test_memory.py` | 修改 |
| `tests/test_ollama.py` | 修改 |

---

## v0.3.0 — 基于 claude-code 模式的核心重构

> 参考 claude-code (v2.1.88) 的 `query.ts`、`Tool.ts`、`QueryEngine.ts` 设计模式，
> 对 edge-agent 的工具系统、推理循环、会话管理做了结构性重构。
> 向后兼容：`ToolRegistry` 别名保留，`@tool` 装饰器不变，apps 入口无需改动。

---

### 核心改动

#### 1. Tool 协议升级（Tool ABC + ToolPool）

**对标**: claude-code `Tool.ts` + `tools.ts` + `assembleToolPool`

**Before**: `ToolRegistry` 是一个扁平的 `{name → func}` 字典 + `@tool` 装饰器，无验证、无权限、无并行标记。

**After**: 三层设计：
- **`Tool` ABC**: `name`, `description`, `parameters_schema()`, `validate_input()`, `is_read_only()`, `is_parallel_safe()`, `is_enabled()`, `execute()`, `openai_schema()`
- **`FunctionTool`**: 将 `@tool` 装饰的函数自动包装为 `Tool` 实例
- **`ToolPool`**: 组装 + 编排，支持 deny list、智能批次执行

新增能力：
- **结果截断**: `MAX_RESULT_CHARS = 30_000`，防止巨大工具输出撑爆上下文
- **输入验证**: `validate_input()` 在执行前检查参数
- **并行安全标记**: `read_only` / `parallel_safe` 标志
- **Deny list**: `ToolPool(tools, deny=["shell"])` 可按名禁用危险工具
- **`ToolRegistry` 别名**: 向后兼容

**影响文件**: `edge_agent/tools.py`（重写）

#### 2. 工具编排（partition + 并行执行）

**对标**: claude-code `toolOrchestration.ts` — `partitionToolCalls` + `runToolsConcurrently`

**实现**:
- `ToolPool._partition(calls)`: 连续的 `parallel_safe` 工具分到同一批，遇到非安全工具断开
- `ToolPool.execute_batch(calls, max_concurrency=5)`: 安全批次用 `asyncio.gather` + `Semaphore` 并行，其余串行
- 结果按原始顺序返回

#### 3. Query Loop 状态机 + 依赖注入

**对标**: claude-code `query.ts` `queryLoop` + `query/deps.ts` `QueryDeps`

**Before**: `OllamaProvider.reason()` 一个 200 行方法，混合了模型调用、工具执行、重试逻辑、消息组装。

**After**: 拆为三部分：
- **`QueryState`**: 持有 `messages`, `tools`, `max_iterations`, 累计 `tools_used`
- **`QueryDeps`**: Protocol，注入 `call_model()` + `execute_tool()` — 测试时替换
- **`query_loop(state, deps, abort)`**: async generator，yield 事件流

#### 4. ConversationEngine: 会话状态分离

**对标**: claude-code `QueryEngine.ts` — 持有 `mutableMessages`，与传输层解耦

**After**: `ConversationEngine` 封装一次对话：
- `submit_message(text) -> TurnResult`
- 内部处理: memory 读写 → context 构建 → tool 注入 → 推理调用 → 结果保存
- `abort()` 取消当前推理

#### 5. Abort/Cancel 支持

**对标**: claude-code `QueryEngine.interrupt()` via `AbortController`

**实现**: `asyncio.Event` 驱动的中断机制，贯穿 OllamaProvider → ConversationEngine → EdgeAgent。

#### 6. OllamaProvider 瘦身

~100 行，职责清晰：实现 `QueryDeps` 协议 + 管理 Ollama 连接 + 重试。

#### 7. Builtin Tools 标注

所有只读工具标注了 `read_only=True, parallel_safe=True`。

---

## v0.2.0 — 架构优化（参考 claude-code 设计模式）

> 对照 `claude-code` (v2.1.88) 的架构设计，对 edge-agent 框架做了结构性改进。
> 所有改动向后兼容，不影响现有 WebSocket 协议和前端。

### 已完成的改进

- P0: 统一异步模型（消除多 event loop）
- P0: WebSocket 消除行为分叉（统一走 EventBus）
- P1: Protocol 签名统一 + 集中配置管理 + 优雅关闭
- P2: 状态转换日志 + 结构化错误类型

---

## 遗留 TODO

### 高优先级

- [ ] API Gateway 切换到 ConversationEngine + ContextBuilder 统一 runtime
- [ ] 集成测试: WebSocket 协议端到端
- [ ] 新增 Provider 时的 ContextRenderer 扩展点验证

### 中优先级

- [ ] **迁移到 Starlette + uvicorn**: 消除 Flask + 线程桥接
- [ ] **Auto-compact**: 当消息超过 token 限制时自动压缩历史
- [ ] **Streaming tool execution**: 当 Ollama 支持流式 tool_use 时适配
- [ ] **分层 Bootstrap**: config → health check → provider init → accept connections
- [ ] **Telemetry / Observability**: 结构化日志 + 指标

### 低优先级

- [ ] `api.py` MiniCPM 路由实现真正的多模态代理
- [ ] `@tool` 装饰器增加参数描述
- [ ] Plugin / hooks 体系
- [ ] 记忆压缩 / 主动提醒
