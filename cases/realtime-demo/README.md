# Realtime Voice + Vision Demo

A **Parlor-style** local real-time AI assistant: browser microphone + camera,
browser-side VAD (Silero), server-side ASR (faster-whisper), local LLM
inference (Ollama), and sentence-level TTS streaming — with full Chinese
support.

## Features

| Feature | Implementation |
|---|---|
| **Voice Activity Detection** | Silero VAD (browser, `@ricky0123/vad-web@0.0.29`) with energy-based fallback |
| **Speech Recognition** | `faster-whisper` (server, Simplified Chinese by default) |
| **LLM Inference** | Ollama streaming API (`gemma4:e2b` primary, auto-fallback) |
| **Text-to-Speech** | Kokoro ONNX → Edge TTS (中/英自动切换) → ffmpeg-flite fallback |
| **Audio Playback** | Gapless PCM scheduling via Web AudioContext (Parlor protocol) |
| **Barge-in** | Echo-suppressed interrupt with 800ms grace period |
| **State Machine** | `loading → listening → processing → speaking` with visual glow |
| **Waveform** | Real-time frequency bar visualization |
| **Camera** | Mirrored selfie view, frame captured at speech-end |
| **Proactive Vision** | Periodic frame diff → auto-analyse camera changes → voice report |
| **Auto-reconnect** | WebSocket reconnects on disconnect (2s backoff) |
| **Conversation Memory** | Server maintains per-session chat history (last 20 turns) |
| **Language Mirroring** | Model replies in the same language the user speaks |
| **Text Input** | Bypass ASR with typed text + Enter |

## Quick Start

```bash
cd cases/realtime-demo
pip install -r requirements.txt

# Start (defaults to Ollama on localhost:11436, model gemma4:e2b)
python3 server.py

# Open in browser
# http://127.0.0.1:8010
```

Grant microphone and camera permissions. Speak naturally — the assistant
responds with voice. Speak again while it's talking to interrupt (barge-in).
Or type text and press Enter to bypass ASR.

Click **"Watch Off"** to toggle **Proactive Vision** — the assistant will
periodically monitor the camera, detect visual changes, and proactively
describe what it sees via TTS.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11436` | Ollama API base URL |
| `REALTIME_DEMO_MODEL` | `gemma4:e2b` | Primary model |
| `REALTIME_DEMO_FALLBACK_MODELS` | `gemma4:e4b,gemma4:26b` | Comma-separated fallback models |
| `REALTIME_DEMO_PORT` | `8010` | Server port |
| `REALTIME_DEMO_HOST` | `127.0.0.1` | Server bind address |
| `REALTIME_DEMO_TTS` | `kokoro` | TTS backend (`kokoro`, `edge`, `flite`) |
| `REALTIME_DEMO_ASR` | `faster-whisper` | ASR backend |
| `REALTIME_DEMO_WHISPER_MODEL` | `base` | Whisper model size |
| `REALTIME_DEMO_WHISPER_LANGUAGE` | `zh` | ASR language lock (`auto` for any) |
| `REALTIME_DEMO_PROACTIVE_VISION` | `false` | Enable proactive vision on startup |
| `REALTIME_DEMO_PROACTIVE_INTERVAL` | `5` | Frame capture interval (seconds) |
| `REALTIME_DEMO_PROACTIVE_DIFF_THRESHOLD` | `0.06` | Frame diff threshold (0~1) |
| `REALTIME_DEMO_PROACTIVE_COOLDOWN` | `15` | Min seconds between proactive inferences |

## Architecture

```
Browser                          Server (Flask + WebSocket)
───────                          ─────────────────────────
Mic → Silero VAD ──────────────→ faster-whisper ASR
Camera → JPEG capture            │  (Simplified Chinese bias)
  │                              ├─→ Ollama streaming LLM
  │                              │     (gemma4:e2b + vision)
  │                              │
  └─ [Watch] frame diff ───────→ proactive vision (cooldown + __SKIP__)
                                 │
AudioContext ←── PCM chunks ←──── TTS (Kokoro / Edge / flite)
  (gapless scheduling)            (auto Chinese/English voice)
```

### TTS Fallback Chain

1. **Kokoro ONNX** — local, fast, multi-language (requires model download)
2. **Edge TTS** — Microsoft cloud, high quality Chinese (`zh-CN-XiaoxiaoNeural`) and English (`en-US-AriaNeural`), auto-detects language
3. **ffmpeg-flite** — local, English only, lowest quality

### WebSocket Protocol (Parlor-compatible)

**Client → Server:**
- `{ audio: "wav_b64", image: "jpeg_b64" }` — voice utterance with camera frame
- `{ type: "interrupt" }` — barge-in (stop current response)
- `{ type: "text_input", text: "..." }` — typed text input
- `{ type: "vision_watch", image: "jpeg_b64" }` — proactive vision frame
- `{ type: "proactive_toggle", enabled: true }` — enable/disable proactive vision

**Server → Client:**
- `{ type: "ready", config: {...}, asr: {...}, tts: {...} }`
- `{ type: "text", transcription: "user said..." }` — ASR result
- `{ type: "assistant_token", text: "chunk" }` — streaming LLM token
- `{ type: "audio_start", sample_rate: 24000 }` — begin PCM stream
- `{ type: "audio_chunk", audio: "pcm_int16_b64", index: 0 }` — TTS audio
- `{ type: "audio_end", tts_time: 1.23 }` — end PCM stream
- `{ type: "text", text: "full response", llm_time: 1.97 }` — final response
- `{ type: "proactive_observation", text: "...", llm_time: 2.1 }` — vision observation
- `{ type: "proactive_status", enabled: true }` — proactive toggle ack

## Dependencies

- **Python**: Flask, flask-sock, httpx, numpy, faster-whisper, edge-tts, kokoro-onnx (optional)
- **System**: ffmpeg (for Kokoro/flite/edge audio decoding)
- **Browser**: Modern browser with WebRTC (mic/camera) and Web Audio API
- **Ollama**: v0.20.4+ with `gemma4:e2b` (or other supported model)

## Known Limitations

- Silero VAD requires `onnxruntime-web@1.22.0` from CDN; falls back to
  energy-based VAD if WASM loading fails
- `faster-whisper` downloads the Whisper model on first use (~150MB)
- Edge TTS requires internet access; for fully offline use, install Kokoro ONNX
- Camera frames are sent only with voice utterances unless proactive vision is enabled
- Proactive vision uses client-side pixel diff to filter unchanged frames; model may still return `__SKIP__` for uninteresting scenes
