# omni-lab

Edge-native private AI agent framework with dual-system cognitive architecture.

Turn any GPU-equipped device into a fully private intelligent terminal: it can hear, see, speak, operate your computer, remember you, and proactively help you. All data stays on-device, forever.

## Architecture: Fast-in-Slow Dual System

```
System 1 (Fast)                     System 2 (Slow)
MiniCPM-o 4.5  ~5GB                Qwen3.5 27B  ~16GB
────────────────────                ────────────────────
Real-time voice (full-duplex)       Deep reasoning + planning
Camera/screen understanding         Tool calling (search/files/shell)
TTS voice synthesis                 Computer control (mouse/keyboard)
Always-on, <1s latency             On-demand, 3-10s latency
Handles 80% of interactions         Handles 20% action requests
```

Both models share the GPU via serial scheduling (GPUScheduler).

## Project Structure

```
omni-lab/
├── edge_agent/                      # Framework core
│   ├── __init__.py                  # EdgeAgent orchestrator (175 lines)
│   ├── events.py                    # Event types + EventBus (169 lines)
│   ├── state.py                     # StateMachine (57 lines)
│   ├── scheduler.py                 # GPUScheduler (43 lines)
│   ├── router.py                    # IntentRouter + KeywordRouter (39 lines)
│   ├── memory.py                    # MemoryStore + Context (119 lines)
│   ├── tools.py                     # ToolRegistry + @tool decorator (90 lines)
│   │
│   ├── providers/
│   │   ├── __init__.py              # PerceptionProvider + ReasoningProvider protocols
│   │   ├── minicpm.py               # System 1: MiniCPM-o via llama-server (295 lines)
│   │   └── ollama.py                # System 2: Qwen via Ollama + ReAct loop (119 lines)
│   │
│   ├── channels/
│   │   ├── __init__.py              # Channel protocol
│   │   ├── cli.py                   # Terminal channel (57 lines)
│   │   └── websocket.py             # Browser channel: voice + camera + agent (548 lines)
│   │
│   └── tools_builtin/
│       ├── __init__.py              # ALL_TOOLS export
│       ├── web.py                   # web_search (DuckDuckGo), web_fetch
│       ├── filesystem.py            # read_file, write_file, edit_file, list_dir
│       ├── shell.py                 # shell (with safety filter)
│       ├── computer.py              # screenshot, click, type_text, scroll, hotkey
│       └── system.py                # memory_save
│
├── apps/
│   ├── assistant.py                 # CLI assistant: System 2 only (37 lines)
│   └── second_brain.py              # Full dual-system app (77 lines)
│
├── memory/                          # Agent memory (plain-text Markdown)
│   ├── SOUL.md                      # AI identity definition
│   ├── USER.md                      # User profile (auto-populated)
│   └── MEMORY.md                    # Long-term facts (AI writes)
│
├── docs/
│   ├── ARCHITECTURE.md              # Detailed architecture design (Chinese, 1254 lines)
│   └── SCENARIOS.md                 # "Private Second Brain" scenario (Chinese, 671 lines)
│
├── llama.cpp-omni/                  # Submodule: llama-server for MiniCPM-o
└── pyproject.toml                   # Python package definition
```

## Key Components

| Component | Description |
|-----------|-------------|
| **EdgeAgent** | Top-level orchestrator. Wires EventBus, GPUScheduler, MemoryStore, ToolRegistry. Handles event routing and System 2 delegation. |
| **EventBus** | Async pub/sub for decoupled component communication. 12 event types defined. |
| **GPUScheduler** | `use_reasoning()` context manager: pauses System 1 before System 2 runs, resumes after. Both models stay in VRAM. |
| **IntentRouter** | Decides fast (System 1) vs slow (System 2). V1 = `KeywordRouter` with Chinese trigger words. |
| **MemoryStore** | Three-tier: `SOUL.md` (identity), `USER.md` (user profile), `MEMORY.md` (AI-written facts). All plain-text Markdown. |
| **ToolRegistry** | Converts `@tool`-decorated functions to OpenAI function-calling schemas. 12 built-in tools. |
| **MiniCPMProvider** | System 1: talks to `llama-server` HTTP API (omni_init, prefill, decode, break, reset, session config, TTS). |
| **OllamaProvider** | System 2: Ollama async client with multi-iteration ReAct loop and tool execution. |
| **WebSocketChannel** | Flask + flask-sock, embedded HTML/JS frontend, camera + mic + speech recognition, HTTPS support. |
| **CLIChannel** | Background thread reads stdin, emits `ChannelMessage`. For development/debugging. |

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (GB10 / RTX 4090 / Jetson recommended)
- Ollama installed: https://ollama.com

### 1. Install

```bash
cd /home/pineapi/gy
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Pull the reasoning model

```bash
ollama pull qwen3.5:27b
# Or smaller model for less VRAM:
# ollama pull qwen3.5:14b
# ollama pull qwen3.5:4b
```

### 3a. Run CLI Assistant (System 2 only)

Minimal setup — just Ollama, no llama-server needed.

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start the assistant
cd /home/pineapi/gy
source .venv/bin/activate
python apps/assistant.py
```

Type messages in the terminal. The AI can:
- Search the web (`web_search`, `web_fetch`)
- Read/write/edit files
- Run shell commands
- Save facts to long-term memory

### 3b. Run Full Second Brain (System 1 + System 2)

Requires both Ollama and llama-server with MiniCPM-o model.

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start llama-server with MiniCPM-o 4.5
cd llama.cpp-omni/build/bin
./llama-server \
    --model /path/to/MiniCPM-o-4_5-Q4_K_M.gguf \
    --port 9060 \
    --gpu-layers 99

# Terminal 3: Start Second Brain
cd /home/pineapi/gy
source .venv/bin/activate
python apps/second_brain.py
```

Open `https://localhost:8080` in your browser (or `http://` if no SSL certs).

Features:
- Real-time voice conversation (full-duplex via MiniCPM-o)
- Camera feed for visual understanding
- Speech recognition → intent routing → System 2 tool calls
- Web search, file operations, shell commands
- Long-term memory in Markdown

### Optional: HTTPS for browser microphone

Browsers require HTTPS for microphone access on non-localhost URLs. Generate self-signed certs:

```bash
openssl req -x509 -newkey rsa:2048 -keyout ssl_key.pem -out ssl_cert.pem \
    -days 365 -nodes -subj '/CN=localhost'
```

### Optional: Computer-use tools

```bash
pip install -e ".[computer]"
```

## Hardware Compatibility

| Hardware | VRAM | System 1 | System 2 |
|----------|------|----------|----------|
| GB10/GB200 | 128 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 27B Q4 (16GB) |
| RTX 4090 | 24 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 9B Q4 (6GB) |
| RTX 3060 | 12 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 4B Q4 (3GB) |
| Jetson Orin | 32-64 GB | MiniCPM-o Q4 (5GB) | Qwen3.5 14B Q4 (9GB) |

## Privacy

- All model inference is 100% local
- All data stored in local filesystem (plain-text Markdown)
- `web_search` / `web_fetch` are the only network exit points (can be disabled)
- Memory is fully auditable: user can review, edit, delete at any time

## Implementation Status

### Done (v0.1)

- [x] Framework core: EdgeAgent, EventBus, StateMachine, GPUScheduler
- [x] IntentRouter with keyword matching (Chinese + English triggers)
- [x] MemoryStore: three-tier Markdown memory (SOUL/USER/MEMORY)
- [x] ToolRegistry with 12 built-in tools
- [x] OllamaProvider: ReAct loop with multi-iteration tool calling
- [x] MiniCPMProvider: full llama-server API integration (init/prefill/decode/break/reset/TTS)
- [x] CLIChannel: terminal-based assistant
- [x] WebSocketChannel: browser UI with camera + mic + speech recognition
- [x] GPU serial scheduling (pause System 1 → run System 2 → resume)
- [x] Context injection (System 2 results → System 1 awareness)
- [x] Auto-reset on context overflow

### TODO

- [ ] **StateMachine integration**: defined but `transition()` is never called from EdgeAgent
- [ ] **VisualScene pipeline**: handler exists but nothing emits `VisualScene` events yet
- [ ] **Passive capture loop**: `CaptureEvent`/`DigestRequest`/`ProactiveHint` events defined but no emitters — the automatic screen-capture → digest → memory_save pipeline is not wired
- [ ] **WebSocketChannel.send()**: currently a no-op (`pass`)
- [ ] **NanobotProvider**: documented in ARCHITECTURE.md but not implemented
- [ ] **TelegramChannel / FeishuChannel**: documented but not implemented
- [ ] **LLMRouter (V2)**: use System 2 for intent classification instead of keyword matching
- [ ] **Memory compression**: daily merge of similar/redundant memories
- [ ] **Memory vector search**: embedding-based retrieval (currently keyword only)
- [ ] **Cron / heartbeat**: proactive scheduled tasks
- [ ] **Scripts**: `install.sh`, `start.sh`, `download_models.sh`
- [ ] **Docker Compose**: containerized deployment
- [ ] **Tests**: no automated tests yet
- [ ] **Sample apps**: `hypeman.py`, `narrator.py` from architecture doc

## License

MIT
