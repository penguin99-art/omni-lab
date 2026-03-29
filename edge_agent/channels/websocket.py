"""WebSocket channel: browser voice + camera + dual-system agent integration.

This is the primary channel that connects both System 1 (perception) and System 2 (reasoning).
Uses Flask + flask_sock for WebSocket, reuses the proven protocol from omni_web_demo.py.

Protocol:
  Client -> {type:"prepare"}
  Server -> {type:"prepared"}
  Client -> {type:"audio_chunk", audio:base64, frame:base64}
  Server -> {type:"result", text:"...", is_listen:bool}
  Server -> {type:"audio", chunks:[...]}
  Client -> {type:"user_text", text:"..."}           <- triggers IntentRouter
  Server -> {type:"agent_status", status:"thinking"}  <- System 2 working
  Server -> {type:"agent_result", text:"..."}          <- System 2 done
  Client -> {type:"stop"}
  Server -> {type:"stopped"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from ..events import (
    EventBus,
    UserSpeech,
    ThinkingStarted,
    ReasoningDone,
    ChannelMessage,
)

if TYPE_CHECKING:
    from ..providers.minicpm import MiniCPMProvider
    from .. import EdgeAgent

log = logging.getLogger(__name__)


class WebSocketChannel:
    """Full-duplex WebSocket channel with voice + camera + agent support."""

    name = "websocket"

    def __init__(self, agent: "EdgeAgent", host: str = "0.0.0.0", port: int = 8080) -> None:
        self._agent = agent
        self._host = host
        self._port = port
        self._bus: Optional[EventBus] = None
        self._app = None

    async def start(self, bus: EventBus) -> None:
        self._bus = bus
        self._setup_flask()
        thread = threading.Thread(target=self._run_flask, daemon=True)
        thread.start()
        log.info("WebSocket channel started on %s:%d", self._host, self._port)

    async def send(self, text: str, media: Optional[bytes] = None) -> None:
        pass

    def _setup_flask(self) -> None:
        from flask import Flask, jsonify
        from flask_sock import Sock

        app = Flask(__name__)
        sock = Sock(app)
        self._app = app
        agent = self._agent

        @app.route("/")
        def index():
            return _get_html()

        @app.route("/api/status")
        def api_status():
            s1_ok = False
            s2_ok = False
            if agent.perception:
                try:
                    loop = asyncio.new_event_loop()
                    s1_ok = loop.run_until_complete(agent.perception.health())
                    loop.close()
                except Exception:
                    pass
            if agent.reasoning:
                try:
                    loop = asyncio.new_event_loop()
                    s2_ok = loop.run_until_complete(agent.reasoning.health())
                    loop.close()
                except Exception:
                    pass
            return jsonify({"system1": s1_ok, "system2": s2_ok})

        @sock.route("/ws/duplex")
        def duplex_ws(ws):
            self._handle_ws(ws)

    def _run_flask(self) -> None:
        import ssl
        from pathlib import Path

        cert = Path("ssl_cert.pem")
        key = Path("ssl_key.pem")
        ssl_ctx = None
        if cert.exists() and key.exists():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert), str(key))

        self._app.run(
            host=self._host,
            port=self._port,
            ssl_context=ssl_ctx,
            debug=False,
            use_reloader=False,
        )

    def _handle_ws(self, ws) -> None:
        """Handle a single WebSocket connection."""
        agent = self._agent
        perception = agent.perception
        send_lock = threading.Lock()
        ws_closed = [False]
        loop = asyncio.new_event_loop()

        def safe_send(data):
            if ws_closed[0]:
                return False
            try:
                with send_lock:
                    ws.send(data)
                return True
            except Exception:
                ws_closed[0] = True
                return False

        def on_thinking(event: ThinkingStarted):
            safe_send(json.dumps({
                "type": "agent_status",
                "status": "thinking",
                "query": event.query,
            }, ensure_ascii=False))

        def on_reasoning_done(event: ReasoningDone):
            safe_send(json.dumps({
                "type": "agent_result",
                "text": event.text,
                "tools_used": [t.get("tool", "") for t in event.tools_used],
            }, ensure_ascii=False))

        agent.bus.on(ThinkingStarted, on_thinking)
        agent.bus.on(ReasoningDone, on_reasoning_done)

        try:
            while True:
                try:
                    raw = ws.receive(timeout=600)
                except Exception:
                    break
                if raw is None:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    safe_send(json.dumps({"type": "error", "error": "invalid json"}))
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "prepare":
                    if perception is None:
                        safe_send(json.dumps({"type": "error", "error": "System 1 not configured"}))
                        continue
                    try:
                        system_prompt = agent.memory.build_system_prompt()
                        loop.run_until_complete(perception.start(system_prompt))
                        safe_send(json.dumps({"type": "prepared"}))
                    except Exception as e:
                        safe_send(json.dumps({"type": "error", "error": str(e)}))

                elif msg_type == "audio_chunk":
                    if perception is None:
                        continue
                    audio_b64 = msg.get("audio", "")
                    frame_b64 = msg.get("frame", "")
                    if not audio_b64:
                        continue
                    try:
                        result = loop.run_until_complete(perception.feed(audio_b64, frame_b64))
                        safe_send(json.dumps({
                            "type": "result",
                            "text": result.text,
                            "is_listen": result.is_listening,
                        }, ensure_ascii=False))
                        if result.audio_chunks:
                            safe_send(json.dumps({
                                "type": "audio",
                                "chunks": result.audio_chunks,
                            }))
                    except Exception as e:
                        log.error("audio_chunk error: %s", e)
                        safe_send(json.dumps({"type": "error", "error": str(e)}))

                elif msg_type == "user_text":
                    user_text = msg.get("text", "").strip()
                    if not user_text:
                        continue
                    agent.memory.append_turn("user", user_text)
                    intent = agent.router.classify(user_text)
                    log.info("user_text: '%s' -> %s", user_text[:40], intent)

                    if intent == "slow":
                        safe_send(json.dumps({
                            "type": "agent_status",
                            "status": "thinking",
                        }))
                        try:
                            loop.run_until_complete(
                                agent._delegate_to_system2(user_text, reply_channel=None)
                            )
                        except Exception as e:
                            safe_send(json.dumps({
                                "type": "agent_result",
                                "text": "Error: {}".format(e),
                                "tools_used": [],
                            }))

                elif msg_type == "stop":
                    if perception:
                        try:
                            loop.run_until_complete(perception.pause())
                        except Exception:
                            pass
                    safe_send(json.dumps({"type": "stopped"}))
                    break

                elif msg_type == "reset":
                    if perception:
                        try:
                            loop.run_until_complete(perception.reset())
                        except Exception:
                            pass
                    safe_send(json.dumps({"type": "reset_done"}))

        finally:
            agent.bus.off(ThinkingStarted, on_thinking)
            agent.bus.off(ReasoningDone, on_reasoning_done)
            if perception:
                try:
                    loop.run_until_complete(perception.pause())
                except Exception:
                    pass
            loop.close()
            log.info("WebSocket session ended")


def _get_html() -> str:
    """Return the frontend HTML page."""
    return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edge Agent - Private AI Assistant</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0f;color:#e0e0e8;min-height:100vh;display:flex;flex-direction:column}
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:14px 24px;border-bottom:1px solid #2a2a4a;display:flex;align-items:center;gap:16px}
.hdr h1{font-size:17px;color:#a0c4ff}.hdr .sub{font-size:11px;color:#666;margin-top:2px}
.status-dots{margin-left:auto;display:flex;gap:10px;align-items:center}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot-ok{background:#51cf66}.dot-off{background:#666}.dot-busy{background:#ffd43b;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.3}}
.status-label{font-size:10px;color:#888}
.main{display:flex;gap:12px;padding:12px;flex:1;max-width:1400px;margin:0 auto;width:100%}
.col{background:#12121f;border:1px solid #2a2a4a;border-radius:10px;padding:14px;display:flex;flex-direction:column}
.col-left{flex:0 0 320px}.col-center{flex:1;min-width:0}.col-right{flex:0 0 280px}
.col h2{font-size:11px;font-weight:600;color:#7b8cde;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
video{width:100%;border-radius:8px;background:#000;margin-bottom:8px}
.ctrls{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.btn{padding:7px 16px;border:none;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;transition:.2s}
.btn-p{background:#3b5bdb;color:#fff}.btn-p:hover{background:#4c6ef5}
.btn-d{background:#c92a2a;color:#fff}.btn-d:hover{background:#e03131}
.btn-s{background:#2a2a4a;color:#ccc}.btn-s:hover{background:#3a3a5a}
.btn:disabled{opacity:.4;cursor:not-allowed}
.chat{flex:1;overflow-y:auto;padding:6px;display:flex;flex-direction:column;gap:6px}
.msg{padding:8px 12px;border-radius:9px;max-width:88%;font-size:13px;line-height:1.5;word-break:break-word}
.msg-user{background:#1e3a5f;color:#a0c4ff;align-self:flex-end}
.msg-bot{background:#1a2a1a;color:#a0ffb0;align-self:flex-start}
.msg-agent{background:#2a1a2a;color:#e0a0ff;align-self:flex-start;border-left:3px solid #9775fa}
.msg-sys{background:#2a2a3a;color:#888;align-self:center;font-size:11px;padding:4px 10px}
.si{padding:7px 12px;border-radius:7px;text-align:center;font-size:13px;font-weight:600;margin-bottom:8px;transition:.3s}
.si-listen{background:#1a2a1a;color:#51cf66;border:1px solid #2b8a3e}
.si-think{background:#2a2a1a;color:#ffd43b;border:1px solid #5c4813}
.si-speak{background:#1a1a3a;color:#748ffc;border:1px solid #3b5bdb}
.si-idle{background:#1a1a1a;color:#666;border:1px solid #333}
.memory-card{background:#0f0f1a;border-radius:8px;padding:10px;margin-bottom:8px;font-size:11px;color:#888;max-height:200px;overflow-y:auto}
.memory-card h3{font-size:11px;color:#7b8cde;margin-bottom:6px}
.tool-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;background:#2a1a3a;color:#c084fc;margin:2px}
#log{font-family:monospace;font-size:10px;color:#555;max-height:80px;overflow-y:auto;padding:5px;background:#0a0a14;border-radius:5px;margin-top:6px}
@media(max-width:900px){.main{flex-direction:column}.col-left,.col-right{flex:0 0 auto}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>Edge Agent</h1>
    <div class="sub">Private AI Assistant | System 1 + System 2</div>
  </div>
  <div class="status-dots">
    <span class="dot dot-off" id="dotS1"></span><span class="status-label">System 1</span>
    <span class="dot dot-off" id="dotS2"></span><span class="status-label">System 2</span>
  </div>
</div>
<div class="main">
  <div class="col col-left">
    <h2>Camera & Controls</h2>
    <video id="cam" autoplay playsinline muted></video>
    <div class="ctrls">
      <button class="btn btn-p" id="btnStart" onclick="start()">Start</button>
      <button class="btn btn-d" id="btnStop" onclick="stop()" disabled>Stop</button>
      <button class="btn btn-s" onclick="resetCtx()">Reset</button>
    </div>
    <div class="si si-idle" id="stateInd">IDLE</div>
    <div id="log"></div>
  </div>
  <div class="col col-center">
    <h2>Conversation</h2>
    <div class="chat" id="chat"></div>
  </div>
  <div class="col col-right">
    <h2>Agent Status</h2>
    <div class="memory-card" id="agentInfo">
      <h3>System 2 (Reasoning)</h3>
      <p>Waiting for activation...</p>
    </div>
    <h2>Tools Used</h2>
    <div id="toolsUsed" style="min-height:40px"></div>
    <h2 style="margin-top:12px">Recent Memory</h2>
    <div class="memory-card" id="memoryView">
      <p>No memories yet.</p>
    </div>
  </div>
</div>
<script>
const PROTO = location.protocol === 'https:' ? 'wss' : 'ws';
let ws, mediaStream, audioCtx, recorder, camVideo;
let sending = false;
const FRAME_EVERY_N = 5;
let frameCounter = 0;

function lg(t) {
  const el = document.getElementById('log');
  const now = new Date().toLocaleTimeString();
  el.innerHTML += `[${now}] ${t}\n`;
  el.scrollTop = el.scrollHeight;
}

function addMsg(cls, text) {
  const chat = document.getElementById('chat');
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function setState(s) {
  const el = document.getElementById('stateInd');
  el.textContent = s;
  el.className = 'si si-' + s.toLowerCase();
}

async function start() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({audio: true, video: true});
    camVideo = document.getElementById('cam');
    camVideo.srcObject = mediaStream;

    audioCtx = new AudioContext({sampleRate: 16000});
    const src = audioCtx.createMediaStreamSource(mediaStream);
    const proc = audioCtx.createScriptProcessor(4096, 1, 1);
    let buf = [];
    proc.onaudioprocess = e => {
      if (!sending) return;
      const d = e.inputBuffer.getChannelData(0);
      buf.push(new Float32Array(d));
      if (buf.length >= 4) {
        const total = buf.reduce((a, b) => a + b.length, 0);
        const merged = new Float32Array(total);
        let off = 0;
        for (const b2 of buf) { merged.set(b2, off); off += b2.length; }
        buf = [];
        sendChunk(merged);
      }
    };
    src.connect(proc);
    proc.connect(audioCtx.destination);

    ws = new WebSocket(`${PROTO}://${location.host}/ws/duplex`);
    ws.onopen = () => { lg('WS connected'); ws.send(JSON.stringify({type:'prepare'})); };
    ws.onmessage = e => handleMsg(JSON.parse(e.data));
    ws.onerror = () => lg('WS error');
    ws.onclose = () => { lg('WS closed'); sending = false; setState('IDLE'); };

    // Speech recognition for text
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      const sr = new SR();
      sr.continuous = true; sr.interimResults = false; sr.lang = 'zh-CN';
      sr.onresult = e => {
        for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) {
            const t = e.results[i][0].transcript.trim();
            if (t) {
              addMsg('msg-user', t);
              ws.send(JSON.stringify({type:'user_text', text:t}));
            }
          }
        }
      };
      sr.onerror = () => {};
      sr.onend = () => { if (sending) sr.start(); };
      sr.start();
      window._sr = sr;
    }

    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnStop').disabled = false;
  } catch(e) { lg('Error: ' + e.message); }
}

function sendChunk(pcm) {
  if (!ws || ws.readyState !== 1) return;
  const wav = encodeWav(pcm, 16000);
  const b64audio = btoa(String.fromCharCode(...new Uint8Array(wav)));

  let b64frame = '';
  frameCounter++;
  if (frameCounter % FRAME_EVERY_N === 0 && camVideo && camVideo.videoWidth > 0) {
    const c = document.createElement('canvas');
    c.width = 320; c.height = 240;
    c.getContext('2d').drawImage(camVideo, 0, 0, 320, 240);
    b64frame = c.toDataURL('image/jpeg', 0.6).split(',')[1];
  }
  ws.send(JSON.stringify({type:'audio_chunk', audio:b64audio, frame:b64frame}));
}

function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const s = (o, str) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
  s(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); s(8, 'WAVE');
  s(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    let s16 = Math.max(-1, Math.min(1, samples[i])) * 0x7FFF;
    v.setInt16(44 + i * 2, s16, true);
  }
  return buf;
}

function handleMsg(m) {
  if (m.type === 'prepared') {
    sending = true;
    setState('LISTEN');
    lg('System 1 ready');
    document.getElementById('dotS1').className = 'dot dot-ok';
  } else if (m.type === 'result') {
    if (m.text && m.text.trim()) {
      addMsg('msg-bot', m.text);
    }
    setState(m.is_listen ? 'LISTEN' : 'SPEAK');
  } else if (m.type === 'audio') {
    playAudio(m.chunks);
  } else if (m.type === 'agent_status') {
    setState('THINK');
    document.getElementById('dotS2').className = 'dot dot-busy';
    document.getElementById('agentInfo').innerHTML = '<h3>System 2</h3><p>Thinking...</p>';
  } else if (m.type === 'agent_result') {
    setState('LISTEN');
    document.getElementById('dotS2').className = 'dot dot-ok';
    addMsg('msg-agent', m.text);
    // Show tools used
    const toolsDiv = document.getElementById('toolsUsed');
    if (m.tools_used && m.tools_used.length > 0) {
      toolsDiv.innerHTML = m.tools_used.map(t => `<span class="tool-badge">${t}</span>`).join('');
    }
    document.getElementById('agentInfo').innerHTML = '<h3>System 2</h3><p>Done.</p>';
    // TTS via browser
    if (m.text && 'speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(m.text);
      u.lang = 'zh-CN'; u.rate = 1.1;
      speechSynthesis.speak(u);
    }
  } else if (m.type === 'error') {
    lg('Error: ' + m.error);
  }
}

function playAudio(chunks) {
  if (!audioCtx) return;
  for (const c of chunks) {
    try {
      const obj = typeof c === 'string' ? JSON.parse(c) : c;
      const raw = atob(obj.pcm);
      const arr = new Float32Array(raw.length / 4);
      const dv = new DataView(new Uint8Array([...raw].map(c2 => c2.charCodeAt(0))).buffer);
      for (let i = 0; i < arr.length; i++) arr[i] = dv.getFloat32(i * 4, true);
      const buf = audioCtx.createBuffer(1, arr.length, obj.sr || 16000);
      buf.getChannelData(0).set(arr);
      const src2 = audioCtx.createBufferSource();
      src2.buffer = buf;
      src2.connect(audioCtx.destination);
      src2.start();
    } catch(e) {}
  }
}

function stop() {
  sending = false;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:'stop'}));
  if (window._sr) try { window._sr.stop(); } catch(e) {}
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  setState('IDLE');
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled = true;
}

function resetCtx() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:'reset'}));
  addMsg('msg-sys', 'Context reset');
}

// Check status on load
fetch('/api/status').then(r=>r.json()).then(d=>{
  document.getElementById('dotS1').className = 'dot ' + (d.system1 ? 'dot-ok' : 'dot-off');
  document.getElementById('dotS2').className = 'dot ' + (d.system2 ? 'dot-ok' : 'dot-off');
}).catch(()=>{});
</script>
</body>
</html>"""
