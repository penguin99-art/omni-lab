# omni-lab

基于 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) 的 Omni 多模态模型实验室 — 端侧部署、深度优化、爆款场景探索。

当前已实现：在本地 GPU 上运行 MiniCPM-o 4.5 的全双工语音+视觉对话系统，含多场景演示和性能分析工具。

## 特性

- **全双工语音对话** — 边听边说，模型自主决定何时回应
- **摄像头视觉理解** — 实时捕获画面，支持图像问答
- **场景演示系统** — 5 种预设场景，一键切换：
  - 💬 **自由对话** — 通用全双工语音对话
  - 👀 **主动提醒助手** — AI 主动观察并提醒，类似科幻电影中的 AI 管家
  - 🌐 **实时翻译官** — 对着外文菜单/路牌实时翻译，或语音口译
  - 📐 **AI 数学家教** — 拍照识别数学题，语音讲解解题过程
  - 🎬 **AI 看片解说员** — 实时描述和点评画面，支持互动提问
- **延迟 Profiling 仪表盘** — 实时显示 prefill/decode/端到端延迟，含 avg/p95/max 统计和趋势图
- **性能优化** — inotify 事件驱动 TTS 推送 + tmpfs (ramdisk) 减少磁盘 I/O
- **WebSocket 流式通信** — 低延迟双向通信，参考 [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 官方架构
- **TTS 语音合成** — 模型生成文字后通过 Token2Wav 合成语音，gapless 无缝播放
- **浏览器语音识别** — 利用 Web Speech API 实时显示用户语音转文字
- **量化评测** — Q4_K_M / Q8_0 量化损失对比 (MMStar 视觉 benchmark)
- **GGUF 量化推理** — Q4_K_M 量化，约 5GB 显存即可运行

## 架构

```
浏览器 (摄像头 + 麦克风 + Web Speech API)
   ↕  WebSocket (wss://)
Flask 中间件 (omni_web_demo.py, HTTPS:8080)
   ↕  HTTP JSON + 文件系统
llama-server (llama.cpp-omni, port 9060)
   ├── LLM 推理 (MiniCPM-o 4.5 Q4_K_M)
   ├── APM 音频编码 (Whisper)
   ├── VPM 视觉编码 (SigLip2)
   └── TTS 语音合成 (Token2Wav)
```

## 目录结构

```
.
├── README.md                 # 本文件
├── omni_web_demo.py          # Web 全双工 Demo（场景选择 + Profiling）
├── omni_duplex_demo.py       # 终端全双工 Demo（命令行交互）
├── run_omni_demo.sh          # 终端 Demo 启动脚本
├── download_models.sh        # 模型下载脚本
├── eval_vision.py            # MMStar 视觉 benchmark 评测脚本
├── run_q8_eval.sh            # Q8_0 一键评测脚本
├── official_ref_audio.wav    # 官方 TTS 参考音频（声音克隆）
├── llama_omni_zh_prompt.patch # 中文提示词补丁（编译前 apply）
├── llama.cpp-omni/           # 推理引擎（C++, 需编译, git submodule）
│   ├── build/bin/
│   │   ├── llama-server      # HTTP 推理服务
│   │   └── llama-omni-cli    # CLI 测试工具
│   └── tools/omni/           # 多模态推理核心代码
├── eval_results/             # 量化评测结果
│   ├── REPORT.md             # 量化对比评测报告
│   ├── mmstar_results_Q4_K_M.jsonl
│   ├── mmstar_summary_Q4_K_M.json
│   └── ...
└── models/                   # 模型文件（不含在代码库中）
    └── MiniCPM-o-4_5-gguf/
        ├── MiniCPM-o-4_5-Q4_K_M.gguf      # 主模型 (~4.8GB)
        ├── MiniCPM-o-4_5-Q8_0.gguf         # 高精度模型 (~8.7GB, 可选)
        ├── audio/                           # 音频编码器
        ├── vision/                          # 视觉编码器
        ├── tts/                             # TTS 模型
        └── token2wav-gguf/                  # 语音合成 vocoder
```

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU, >= 8GB VRAM (推荐 16GB+) |
| CUDA | >= 12.0 |
| OS | Linux (已测试 aarch64 / x86_64) |
| Python | >= 3.10 |
| CMake | >= 3.16 |
| 浏览器 | Chrome / Edge (需 HTTPS 才能使用麦克风和摄像头) |

## 快速开始

### 1. 克隆项目

```bash
git clone --recursive https://github.com/<your-username>/omni-lab.git
cd omni-lab
```

### 2. 下载模型

```bash
bash download_models.sh
```

模型文件约 10GB，来自 [HuggingFace openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)。下载完成后需要创建 projector 软链接：

```bash
ln -s models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
      models/MiniCPM-o-4_5-gguf/token2wav-gguf/MiniCPM-o-4_5-projector-F16.gguf
```

### 3. 编译 llama.cpp-omni

```bash
cd llama.cpp-omni

# 应用中文提示词补丁（使模型以普通话回答，而非英文/粤语）
git apply ../llama_omni_zh_prompt.patch

cmake -B build \
  -DLLAMA_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-omni-cli -j$(nproc)
cd ..
```

### 4. 生成 SSL 证书

浏览器的麦克风/摄像头 API 要求 HTTPS。生成自签名证书：

```bash
openssl req -x509 -newkey rsa:2048 -keyout ssl_key.pem -out ssl_cert.pem \
  -days 365 -nodes -subj "/CN=localhost"
```

### 5. 安装 Python 依赖

```bash
pip install flask flask-sock numpy soundfile requests inotify
```

### 6. 启动服务

**终端 1 — 启动推理服务：**

```bash
./llama.cpp-omni/build/bin/llama-server \
  --host 0.0.0.0 --port 9060 \
  --model ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  -ngl 99 --ctx-size 4096 --repeat-penalty 1.05 --temp 0.7
```

等待出现 `"status":"ok"` 后继续。

**终端 2 — 启动 Web Demo：**

```bash
python3 omni_web_demo.py --port 8080 --llama-port 9060
```

### 7. 打开浏览器

访问 `https://<your-ip>:8080`，接受自签名证书警告，点击「开始对话」。

## 使用说明

### Web Demo (`omni_web_demo.py`)

- **场景选择** — 顶部栏选择场景（自由对话/主动提醒/翻译官/数学家教/解说员），需要在停止状态下切换
- **语音对话** — 点击「开始对话」，对着麦克风说话。模型会自动回应并播放语音。
- **文本对话** — 切换到「文本对话」标签，直接输入文字测试模型。
- **摄像头** — 勾选「摄像头」后，模型可以看到你的画面并结合视觉回答。
- **延迟监控** — 右侧面板实时显示 prefill/decode/端到端延迟统计和趋势图
- **重置** — 点击「重置」清除对话历史和模型状态。

### 终端 Demo (`omni_duplex_demo.py`)

无需浏览器，直接在终端通过麦克风和扬声器交互：

```bash
bash run_omni_demo.sh
# 或
python3 omni_duplex_demo.py --llama-port 9060
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 8080 | Web Demo 端口 |
| `--llama-port` | 9060 | llama-server 端口 |
| `--llama-host` | 127.0.0.1 | llama-server 地址 |
| `--no-ssl` | - | 禁用 HTTPS（不推荐） |
| `--ssl-cert` | ssl_cert.pem | SSL 证书路径 |
| `--ssl-key` | ssl_key.pem | SSL 密钥路径 |

## 模型文件清单

从 HuggingFace 下载，共约 10GB：

| 文件 | 大小 | 说明 |
|------|------|------|
| `MiniCPM-o-4_5-Q4_K_M.gguf` | ~4.8 GB | 主 LLM (Q4_K_M 量化) |
| `vision/MiniCPM-o-4_5-vision-F16.gguf` | ~2.5 GB | 视觉编码器 (SigLip2) |
| `tts/MiniCPM-o-4_5-tts-F16.gguf` | ~1.1 GB | TTS 模型 |
| `tts/MiniCPM-o-4_5-projector-F16.gguf` | ~15 MB | TTS Projector |
| `audio/MiniCPM-o-4_5-audio-F16.gguf` | ~629 MB | 音频编码器 (Whisper) |
| `token2wav-gguf/flow_matching.gguf` | ~437 MB | Flow Matching 模型 |
| `token2wav-gguf/prompt_cache.gguf` | ~211 MB | Prompt 缓存 |
| `token2wav-gguf/encoder.gguf` | ~144 MB | Token2Wav 编码器 |
| `token2wav-gguf/hifigan2.gguf` | ~79 MB | HiFi-GAN 声码器 |
| `token2wav-gguf/flow_extra.gguf` | ~13 MB | Flow 辅助模型 |

## 技术细节

### 全双工通信协议 (WebSocket)

```
Client → {"type":"prepare", "media_type":2, "duplex":true}
Server → {"type":"prepared"}

Client → {"type":"audio_chunk", "audio":"<base64 WAV>", "frame":"<base64 JPEG>"}
Server → {"type":"result", "text":"...", "is_listen":true/false}
Server → {"type":"audio", "chunks":[{"pcm":"<base64 float32>", "sr":24000}]}

Client → {"type":"stop"}
Server → {"type":"stopped"}
```

### 推理流程

每 1 秒为一个周期：
1. 浏览器采集 1s 音频（16kHz mono WAV）+ 摄像头帧（JPEG）
2. 通过 WebSocket 发送到 Flask 中间件
3. 中间件调用 `llama-server` 的 `/v1/stream/prefill` 和 `/v1/stream/decode`
4. 模型返回 `is_listen`（继续听）或文本内容（开始说话）
5. TTS 后台线程自动检测新生成的 WAV 文件并推送到浏览器
6. 浏览器使用 Web Audio API 进行 gapless 无缝播放

## 量化评测

使用 [MMStar](https://mmstar-benchmark.github.io/) 视觉 benchmark 评估量化损失（300 题子集, seed=42）：

| 模型版本 | MMStar 准确率 | 量化损失 | 平均延迟 | 模型大小 |
|---|---|---|---|---|
| bf16 (官方) | **73.1%** | -- | -- | ~18 GB |
| Q8_0 | **69.67%** | -3.43 pp | 3.97s | ~8.7 GB |
| Q4_K_M | **70.33%** | -2.77 pp | 2.50s | ~4.8 GB |

**结论**: Q4_K_M 是最佳性价比选择 — 精度与 Q8_0 相当，延迟更低，显存仅需一半。

详见 [`eval_results/REPORT.md`](eval_results/REPORT.md)。

运行评测：

```bash
# 运行 Q4_K_M 或 Q8_0 评测
python3 eval_vision.py --samples 300 --tag Q4_K_M
python3 eval_vision.py --samples 300 --tag Q8_0
```

## 致谢

- [MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o) — 面壁智能 & OpenBMB
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) — GGUF 多模态推理引擎
- [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) — 官方 PyTorch Demo（架构参考）

## License

本项目 Demo 代码以 MIT 协议开源。模型权重请遵循 [MiniCPM-o 4.5 许可协议](https://github.com/OpenBMB/MiniCPM-o)。
