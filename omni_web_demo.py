#!/usr/bin/env python3
"""
MiniCPM-o 4.5 全双工语音+摄像头 Web Demo
(参考 OpenBMB/MiniCPM-o-Demo 官方架构重写)

架构:
  浏览器 (摄像头+麦克风)
    ↕ WebSocket (单连接，全双工)
  Flask 中间件 (port 8080, HTTPS)
    ↕ 文件磁盘 + HTTP JSON
  llama-server (port 9060)

协议:
  Client → {"type":"prepare"}
  Server → {"type":"prepared"}
  Client → {"type":"audio_chunk", "audio":"base64 WAV", "frame":"base64 JPEG"}
  Server → {"type":"result", "text":"...", "is_listen":bool}  (立即返回，不等TTS)
  Server → {"type":"audio", "chunks":[...]}                   (后台线程推送TTS音频)
  Client → {"type":"stop"}
  Server → {"type":"stopped"}
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from flask import Flask, jsonify, request
from flask_sock import Sock

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "MiniCPM-o-4_5-gguf"

WORK_DIR = Path(tempfile.mkdtemp(prefix="omni_web_"))
AUDIO_DIR = WORK_DIR / "audio"
FRAME_DIR = WORK_DIR / "frames"
OUTPUT_DIR = WORK_DIR / "output"
for d in [AUDIO_DIR, FRAME_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
sock = Sock(app)

state = {
    "llama_host": "127.0.0.1",
    "llama_port": 9060,
    "initialized": False,
    "prefill_cnt": 1,
    "audio_idx": 0,
    "frame_idx": 0,
    "round_idx": 0,
}
state_lock = threading.Lock()


def llama_url(path: str) -> str:
    return f"http://{state['llama_host']}:{state['llama_port']}{path}"


def _do_init(media_type=2, duplex=True):
    ref_audio = str(BASE_DIR / "official_ref_audio.wav")
   
    body = {
        "media_type": media_type,
        "use_tts": True,
        "duplex_mode": duplex,
        "model_dir": str(MODEL_DIR) + "/",
        "tts_bin_dir": str(MODEL_DIR / "token2wav-gguf"),
        "tts_gpu_layers": 99,
        "token2wav_device": "gpu:0",
        "output_dir": str(OUTPUT_DIR),
        "n_predict": 2048,
        "voice_audio": ref_audio,
    }
    r = requests.post(llama_url("/v1/stream/omni_init"), json=body, timeout=120)
    r.raise_for_status()
    resp = r.json()
    if not resp.get("success"):
        raise RuntimeError(f"omni_init failed: {resp}")
    with state_lock:
        state["initialized"] = True
        state["prefill_cnt"] = 1
        state["round_idx"] = 0
    return resp


def _do_prefill(audio_b64, frame_b64=None):
    audio_bytes = base64.b64decode(audio_b64)
    with state_lock:
        cnt = state["prefill_cnt"]
        idx = state["audio_idx"]
        state["audio_idx"] += 1
    audio_path = str(AUDIO_DIR / f"chunk_{idx:06d}.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    img_path = ""
    if frame_b64:
        frame_bytes = base64.b64decode(frame_b64)
        with state_lock:
            fidx = state["frame_idx"]
            state["frame_idx"] += 1
        img_path = str(FRAME_DIR / f"frame_{fidx:06d}.jpg")
        with open(img_path, "wb") as f:
            f.write(frame_bytes)

    body = {"audio_path_prefix": audio_path, "img_path_prefix": img_path, "cnt": cnt}
    r = requests.post(llama_url("/v1/stream/prefill"), json=body, timeout=30)
    r.raise_for_status()
    with state_lock:
        state["prefill_cnt"] += 1
    return True


def _do_decode():
    with state_lock:
        rid = state["round_idx"]

    body = {"stream": True, "round_idx": rid}
    r = requests.post(llama_url("/v1/stream/decode"), json=body, stream=True, timeout=600)
    r.raise_for_status()
    r.encoding = "utf-8"

    full_text = ""
    is_listen = False
    is_end_of_turn = False

    for chunk in r.iter_content(chunk_size=None):
        if not chunk:
            continue
        text = chunk.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                ev = json.loads(payload)
                if ev.get("content"):
                    full_text += ev["content"]
                if ev.get("is_listen"):
                    is_listen = True
                if ev.get("end_of_turn") or ev.get("stop"):
                    is_end_of_turn = True
            except (json.JSONDecodeError, KeyError):
                pass

    with state_lock:
        state["round_idx"] += 1

    return full_text, is_listen, is_end_of_turn


def _collect_new_wavs(cursor):
    """Collect WAV files starting from cursor index. Returns (list, new_cursor)."""
    results = []
    tts_dir = OUTPUT_DIR / "tts_wav"
    if not tts_dir.exists():
        return results, cursor
    while True:
        wav_path = tts_dir / f"wav_{cursor}.wav"
        if not wav_path.exists():
            break
        try:
            if wav_path.stat().st_size < 100:
                break
            data, sr = sf.read(str(wav_path), dtype="float32")
            pcm_b64 = base64.b64encode(data.astype(np.float32).tobytes()).decode("ascii")
            results.append({"i": cursor, "pcm": pcm_b64, "sr": sr})
            cursor += 1
        except Exception:
            break
    return results, cursor


# ─── WebSocket Duplex ────────────────────────────────────────

@sock.route("/ws/duplex")
def duplex_ws(ws):
    """
    Full-duplex WebSocket. Decode returns immediately; TTS audio is pushed
    from a separate background thread that polls for new WAV files.
    """
    wav_cursor = [0]
    tts_stop = threading.Event()

    def tts_pusher():
        """Background thread: polls for TTS WAV files and pushes them via WS."""
        while not tts_stop.is_set():
            try:
                new_wavs, wav_cursor[0] = _collect_new_wavs(wav_cursor[0])
                if new_wavs:
                    ws.send(json.dumps({
                        "type": "audio",
                        "chunks": new_wavs,
                    }, ensure_ascii=False))
            except Exception:
                break
            tts_stop.wait(0.1)

    tts_thread = None

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
                ws.send(json.dumps({"type": "error", "error": "invalid json"}))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "prepare":
                try:
                    media_type = msg.get("media_type", 2)
                    duplex = msg.get("duplex", True)

                    # Clean up old TTS WAV files before starting new session
                    tts_dir = OUTPUT_DIR / "tts_wav"
                    if tts_dir.exists():
                        import shutil
                        shutil.rmtree(tts_dir, ignore_errors=True)
                        tts_dir.mkdir(parents=True, exist_ok=True)
                        print("[prepare] cleaned old TTS WAV files", flush=True)

                    _do_init(media_type, duplex)
                    wav_cursor[0] = 0

                    if tts_thread is None or not tts_thread.is_alive():
                        tts_thread = threading.Thread(target=tts_pusher, daemon=True)
                        tts_thread.start()

                    ws.send(json.dumps({"type": "prepared"}))
                except Exception as e:
                    ws.send(json.dumps({"type": "error", "error": str(e)}))

            elif msg_type == "audio_chunk":
                audio_b64 = msg.get("audio")
                frame_b64 = msg.get("frame")
                if not audio_b64:
                    ws.send(json.dumps({"type": "error", "error": "no audio"}))
                    continue

                try:
                    t0 = time.monotonic()
                    _do_prefill(audio_b64, frame_b64)
                    t1 = time.monotonic()
                    text, is_listen, is_end = _do_decode()
                    t2 = time.monotonic()

                    prefill_ms = (t1 - t0) * 1000
                    decode_ms = (t2 - t1) * 1000
                    total_ms = (t2 - t0) * 1000

                    status = "LISTEN" if is_listen else "SPEAK"
                    print(f"[{status}] prefill={prefill_ms:.0f}ms decode={decode_ms:.0f}ms "
                          f"total={total_ms:.0f}ms text='{text[:40]}'",
                          flush=True)

                    ws.send(json.dumps({
                        "type": "result",
                        "text": text,
                        "is_listen": is_listen,
                        "end_of_turn": is_end,
                        "timing": {"prefill": round(prefill_ms),
                                   "decode": round(decode_ms),
                                   "total": round(total_ms)},
                    }, ensure_ascii=False))

                except Exception as e:
                    print(f"[ERROR] audio_chunk: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        ws.send(json.dumps({"type": "error", "error": str(e)}))
                    except Exception:
                        break

            elif msg_type == "stop":
                try:
                    requests.post(llama_url("/v1/stream/break"), json={}, timeout=5)
                except Exception:
                    pass
                ws.send(json.dumps({"type": "stopped"}))
                break

            elif msg_type == "reset":
                try:
                    requests.post(llama_url("/v1/stream/reset"), json={}, timeout=10)
                    with state_lock:
                        state["round_idx"] = 0
                        state["prefill_cnt"] = 1
                    wav_cursor[0] = 0
                    ws.send(json.dumps({"type": "reset_done"}))
                except Exception as e:
                    ws.send(json.dumps({"type": "error", "error": str(e)}))

    finally:
        tts_stop.set()
        if tts_thread and tts_thread.is_alive():
            tts_thread.join(timeout=2)
        print("[WS] duplex session ended", flush=True)


# ─── HTTP ─────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    try:
        r = requests.get(llama_url("/health"), timeout=3)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return jsonify({"server_ok": ok, "initialized": state["initialized"],
                    "round_idx": state["round_idx"]})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Text chat via llama-server's /v1/chat/completions"""
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    history = data.get("history", [])
    messages = [{"role": "system", "content": "你是一个友好的中文助手，请用普通话回答。"}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})
    try:
        r = requests.post(llama_url("/v1/chat/completions"), json={
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
        }, timeout=60)
        r.raise_for_status()
        resp = r.json()
        text = resp["choices"][0]["message"]["content"]
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return HTML_PAGE


# ─── 前端 ────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniCPM-o 4.5 全双工对话</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0f;color:#e0e0e8;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:16px 24px;border-bottom:1px solid #2a2a4a}
.hdr h1{font-size:18px;color:#a0c4ff}.hdr p{font-size:12px;color:#888;margin-top:2px}
.main{display:flex;gap:12px;padding:12px;max-width:1400px;margin:0 auto;height:calc(100vh - 70px)}
.pnl{background:#12121f;border:1px solid #2a2a4a;border-radius:10px;padding:14px;display:flex;flex-direction:column}
.left{flex:0 0 360px}.right{flex:1;min-width:0}
.pnl h2{font-size:13px;font-weight:600;color:#7b8cde;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
video{width:100%;border-radius:8px;background:#000;margin-bottom:10px}
.ctrls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.btn{padding:8px 18px;border:none;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;transition:.2s}
.btn-p{background:#3b5bdb;color:#fff}.btn-p:hover{background:#4c6ef5}
.btn-d{background:#c92a2a;color:#fff}.btn-d:hover{background:#e03131}
.btn-s{background:#2a2a4a;color:#ccc}.btn-s:hover{background:#3a3a5a}
.btn:disabled{opacity:.4;cursor:not-allowed}
.sbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.badge{padding:3px 9px;border-radius:16px;font-size:11px;font-weight:600}
.b-ok{background:#0b7285;color:#e3fafc}.b-err{background:#862e2e;color:#ffc9c9}
.b-info{background:#1a1a3a;color:#aaa}
.chat{flex:1;overflow-y:auto;padding:6px;display:flex;flex-direction:column;gap:8px}
.msg{padding:9px 13px;border-radius:9px;max-width:88%;font-size:14px;line-height:1.5;word-break:break-word}
.msg-user{background:#1e3a5f;color:#a0c4ff;align-self:flex-end}
.msg-bot{background:#1a2a1a;color:#a0ffb0;align-self:flex-start}
.msg-sys{background:#2a2a3a;color:#888;align-self:center;font-size:11px;padding:5px 10px}
.tabBtn.active{background:#3b5bdb!important;color:#fff!important}
.meter{height:5px;background:#1a1a2e;border-radius:3px;margin:6px 0;overflow:hidden}
.meter-bar{height:100%;transition:width .1s;width:0%}
.viz{display:flex;align-items:flex-end;gap:2px;height:36px;margin:6px 0}
.viz .bar{width:3px;background:#3b5bdb;border-radius:2px;transition:height 50ms}
.si{padding:8px 14px;border-radius:7px;text-align:center;font-size:14px;font-weight:600;margin-bottom:10px;transition:.3s}
.si-listen{background:#1a2a1a;color:#51cf66;border:1px solid #2b8a3e}
.si-think{background:#2a2a1a;color:#ffd43b;border:1px solid #5c4813}
.si-speak{background:#1a1a3a;color:#748ffc;border:1px solid #3b5bdb}
.si-idle{background:#1a1a1a;color:#666;border:1px solid #333}
.timing{font-size:10px;color:#555;margin-top:2px}
#log{font-family:'Cascadia Code','Fira Code',monospace;font-size:10px;color:#555;max-height:120px;overflow-y:auto;padding:6px;background:#0a0a14;border-radius:5px;margin-top:8px}
@media(max-width:900px){.main{flex-direction:column}.left{flex:0 0 auto}}
</style>
</head>
<body>

<div class="hdr">
  <h1>MiniCPM-o 4.5 Omni 全双工对话</h1>
  <p>WebSocket 全双工 · PCM 流式音频 · 参考官方 MiniCPM-o-Demo</p>
</div>

<div class="main">
  <div class="pnl left">
    <h2>输入</h2>
    <div class="si si-idle" id="si">待机</div>
    <video id="cam" autoplay muted playsinline></video>
    <div class="viz" id="viz"></div>
    <div class="meter"><div class="meter-bar" id="vol"></div></div>
    <div class="ctrls">
      <button class="btn btn-p" id="btnGo" onclick="go()">开始对话</button>
      <button class="btn btn-d" id="btnStop" onclick="stop()" disabled>停止</button>
      <button class="btn btn-s" onclick="doReset()">重置</button>
    </div>
    <div class="ctrls">
      <label style="font-size:12px;color:#888;display:flex;align-items:center;gap:5px">
        <input type="checkbox" id="chkCam" checked> 摄像头
      </label>
    </div>
    <div class="sbar">
      <span class="badge b-err" id="bSrv">Server: --</span>
      <span class="badge b-info" id="bRnd">轮次: 0</span>
      <span class="badge b-info" id="bTim">--</span>
    </div>
    <div id="log"></div>
  </div>
  <div class="pnl right">
    <div style="display:flex;gap:8px;margin-bottom:10px">
      <button class="btn btn-s tabBtn active" onclick="switchTab('voice')" id="tabVoice">语音对话</button>
      <button class="btn btn-s tabBtn" onclick="switchTab('text')" id="tabText">文本对话</button>
    </div>
    <div id="voiceTab" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2>语音对话</h2>
      <div class="chat" id="chat"></div>
    </div>
    <div id="textTab" style="flex:1;display:none;flex-direction:column;min-height:0">
      <h2>文本对话</h2>
      <div class="chat" id="textChat"></div>
      <div style="display:flex;gap:8px;margin-top:8px">
        <input id="txtInput" type="text" placeholder="输入消息..." 
               style="flex:1;padding:8px 12px;border-radius:7px;border:1px solid #3a3a5a;background:#1a1a2e;color:#e0e0e8;font-size:14px"
               onkeydown="if(event.key==='Enter')sendText()">
        <button class="btn btn-p" onclick="sendText()">发送</button>
      </div>
    </div>
  </div>
</div>

<script>
const CHUNK_MS=1000, PLAYBACK_SR=24000, PLAYBACK_DELAY_MS=250;
let ws=null, mediaStream=null, audioCtx=null, analyser=null;
let active=false, timer=null, micChunks=[];
let playCtx=null, nextPlayTime=0, pendingBufs=0;
let inflight=0;
let curBotEl=null;  // current bot message element (for merging text)
let recognition=null; // speech recognition
let curUserEl=null;  // current user speech element

function log(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString('zh',{hour12:false});e.textContent+=`[${t}] ${m}\n`;e.scrollTop=e.scrollHeight}
function addMsg(r,t){const c=document.getElementById('chat'),d=document.createElement('div');d.className=`msg msg-${r}`;d.textContent=t;c.appendChild(d);c.scrollTop=c.scrollHeight;return d}
function appendBot(t){
  if(!t)return;
  const c=document.getElementById('chat');
  if(!curBotEl){curBotEl=document.createElement('div');curBotEl.className='msg msg-bot';curBotEl.textContent='';c.appendChild(curBotEl)}
  curBotEl.textContent+=t;
  c.scrollTop=c.scrollHeight;
}
function finishBot(){curBotEl=null}
function setSt(s,t){const e=document.getElementById('si');e.className=`si si-${s}`;e.textContent=t}

async function poll(){try{const r=await fetch('/api/status'),d=await r.json();document.getElementById('bSrv').textContent=`Server: ${d.server_ok?'OK':'OFF'}`;document.getElementById('bSrv').className=`badge ${d.server_ok?'b-ok':'b-err'}`;document.getElementById('bRnd').textContent=`轮次: ${d.round_idx}`}catch(_){}}
setInterval(poll,5000);poll();

// ─── audio capture ───
function collectChunk(){
  if(!micChunks.length)return null;
  const c=micChunks; micChunks=[];
  let n=0; for(const x of c)n+=x.length;
  const m=new Float32Array(n); let o=0;
  for(const x of c){m.set(x,o);o+=x.length}
  return m;
}

function f32toWavB64(s,sr){
  const n=s.length,b=new ArrayBuffer(44+n*2),v=new DataView(b);
  function w(o,t){for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i))}
  w(0,'RIFF');v.setUint32(4,36+n*2,true);w(8,'WAVE');w(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);
  v.setUint16(34,16,true);w(36,'data');v.setUint32(40,n*2,true);
  for(let i=0;i<n;i++){let x=Math.max(-1,Math.min(1,s[i]));v.setInt16(44+i*2,x<0?x*0x8000:x*0x7FFF,true)}
  return btoa(String.fromCharCode(...new Uint8Array(b)));
}

function captureFrame(){
  const v=document.getElementById('cam');
  if(!v.srcObject||!v.videoWidth)return null;
  const c=document.createElement('canvas');
  c.width=Math.min(v.videoWidth,640);
  c.height=Math.round(c.width*v.videoHeight/v.videoWidth);
  c.getContext('2d').drawImage(v,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',0.6).split(',')[1];
}

// ─── gapless PCM playback ───
function scheduleAudio(pcmB64,sr){
  if(!playCtx)return;
  const raw=atob(pcmB64);
  const bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  const arr=new Float32Array(bytes.buffer);

  const buf=playCtx.createBuffer(1,arr.length,sr||PLAYBACK_SR);
  buf.getChannelData(0).set(arr);
  const src=playCtx.createBufferSource();
  src.buffer=buf; src.connect(playCtx.destination);

  const now=playCtx.currentTime;
  if(nextPlayTime<now) nextPlayTime=now+PLAYBACK_DELAY_MS/1000;
  src.start(nextPlayTime);
  nextPlayTime+=buf.duration;
  pendingBufs++;
  src.onended=()=>{
    pendingBufs--;
    if(pendingBufs<=0) setSt('listen','正在听...');
  };
}

function playChunks(chunks){
  if(!chunks||!chunks.length)return;
  setSt('speak','回复中...');
  for(const c of chunks)scheduleAudio(c.pcm,c.sr);
}

// ─── WebSocket ───
function connectWs(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/ws/duplex`);
  ws.onopen=()=>{
    log('WS 已连接');
    const useCam=document.getElementById('chkCam').checked;
    ws.send(JSON.stringify({type:'prepare',media_type:useCam?2:1,duplex:true}));
  };
  ws.onclose=()=>{log('WS 断开');ws=null};
  ws.onerror=()=>log('WS 错误');
  ws.onmessage=e=>{
    let d;
    try{d=JSON.parse(e.data)}catch(_){return}

    if(d.type==='prepared'){
      addMsg('sys','模型就绪，请说话');
      setSt('listen','正在听...');
      startLoop();
    } else if(d.type==='result'){
      inflight=Math.max(0,inflight-1);
      if(d.text && d.text.length>0){
        appendBot(d.text);
      }
      if(d.is_listen){
        finishBot();
        if(pendingBufs<=0) setSt('listen','正在听...');
      }
      if(d.timing){
        document.getElementById('bTim').textContent=
          `P:${d.timing.prefill}ms D:${d.timing.decode}ms`;
      }
    } else if(d.type==='audio'){
      playChunks(d.chunks);
      log(`收到 ${d.chunks.length} 段音频`);
    } else if(d.type==='error'){
      log('错误: '+d.error);
      inflight=Math.max(0,inflight-1);
    } else if(d.type==='stopped'){
      log('会话已停止');
    } else if(d.type==='reset_done'){
      log('已重置');
    }
  };
}

function startLoop(){
  timer=setInterval(()=>{
    if(!active||!ws||ws.readyState!==1)return;
    // Allow max 1 in-flight request to keep latency low but prevent overlap
    if(inflight>=1)return;
    const samples=collectChunk();
    if(!samples||samples.length<100)return;
    const useCam=document.getElementById('chkCam').checked;
    const audioB64=f32toWavB64(samples,16000);
    const msg={type:'audio_chunk',audio:audioB64};
    if(useCam){const f=captureFrame();if(f)msg.frame=f}
    inflight++;
    ws.send(JSON.stringify(msg));
  },CHUNK_MS);
}

// ─── session lifecycle ───
async function go(){
  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
    addMsg('sys',location.protocol==='http:'?'需要 HTTPS 访问':'浏览器不支持');return
  }
  const useCam=document.getElementById('chkCam').checked;
  try{
    const c={audio:{sampleRate:16000,channelCount:1,echoCancellation:true,noiseSuppression:true}};
    if(useCam)c.video={width:{ideal:640},height:{ideal:480}};
    mediaStream=await navigator.mediaDevices.getUserMedia(c);
  }catch(e){addMsg('sys','无法访问设备: '+e.message);return}
  if(useCam)document.getElementById('cam').srcObject=mediaStream;

  audioCtx=new AudioContext({sampleRate:16000});
  const src=audioCtx.createMediaStreamSource(mediaStream);
  analyser=audioCtx.createAnalyser();analyser.fftSize=256;
  src.connect(analyser);
  const proc=audioCtx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=e=>{if(active)micChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))};
  const g=audioCtx.createGain();g.gain.value=0;g.connect(audioCtx.destination);
  src.connect(proc);proc.connect(g);
  setupViz();

  playCtx=new AudioContext({sampleRate:PLAYBACK_SR});
  nextPlayTime=0;pendingBufs=0;inflight=0;

  active=true;
  document.getElementById('btnGo').disabled=true;
  document.getElementById('btnStop').disabled=false;
  setSt('think','正在初始化...');
  addMsg('sys','正在连接...');

  // Speech recognition for showing user's words
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(SR){
    recognition=new SR();
    recognition.lang='zh-CN';
    recognition.continuous=true;
    recognition.interimResults=true;
    recognition.onresult=ev=>{
      for(let i=ev.resultIndex;i<ev.results.length;i++){
        const t=ev.results[i][0].transcript.trim();
        if(!t)continue;
        if(ev.results[i].isFinal){
          // Finalize: replace interim element or create new one
          if(curUserEl){curUserEl.textContent=t;curUserEl=null}
          else addMsg('user',t);
          finishBot();
        } else {
          // Interim: show live text in a temporary bubble
          const c=document.getElementById('chat');
          if(!curUserEl){curUserEl=document.createElement('div');curUserEl.className='msg msg-user';curUserEl.style.opacity='0.6';c.appendChild(curUserEl)}
          curUserEl.textContent=t;
          c.scrollTop=c.scrollHeight;
        }
      }
    };
    recognition.onend=()=>{if(active)try{recognition.start()}catch(_){}};
    recognition.onerror=ev=>{if(ev.error!=='no-speech'&&ev.error!=='aborted')log('语音识别: '+ev.error)};
    try{recognition.start()}catch(_){}
  }

  connectWs();
}

function stop(){
  active=false;
  if(timer){clearInterval(timer);timer=null}
  if(recognition){try{recognition.abort()}catch(_){};recognition=null}
  curBotEl=null;curUserEl=null;
  try{if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'stop'}))}catch(_){}
  try{if(ws)ws.close()}catch(_){}
  ws=null;
  if(mediaStream){mediaStream.getTracks().forEach(t=>t.stop());mediaStream=null}
  if(audioCtx){audioCtx.close();audioCtx=null;analyser=null}
  document.getElementById('cam').srcObject=null;
  document.getElementById('btnGo').disabled=false;
  document.getElementById('btnStop').disabled=true;
  setSt('idle','已停止');
  addMsg('sys','对话已停止');
}

async function doReset(){
  stop();
  document.getElementById('chat').innerHTML='';
  setSt('idle','待机');
  addMsg('sys','已重置');
}

// ─── text chat ───
let textHistory=[];
function switchTab(tab){
  document.getElementById('voiceTab').style.display=tab==='voice'?'flex':'none';
  document.getElementById('textTab').style.display=tab==='text'?'flex':'none';
  document.getElementById('tabVoice').className='btn btn-s tabBtn'+(tab==='voice'?' active':'');
  document.getElementById('tabText').className='btn btn-s tabBtn'+(tab==='text'?' active':'');
}
function addTextMsg(role,text){
  const c=document.getElementById('textChat'),d=document.createElement('div');
  d.className=`msg msg-${role==='user'?'user':'bot'}`;
  d.textContent=text;c.appendChild(d);c.scrollTop=c.scrollHeight;
}
async function sendText(){
  const inp=document.getElementById('txtInput'),msg=inp.value.trim();
  if(!msg)return;
  inp.value='';inp.disabled=true;
  addTextMsg('user',msg);
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,history:textHistory})});
    const d=await r.json();
    if(d.error){addTextMsg('sys','错误: '+d.error)}
    else{addTextMsg('bot',d.text);textHistory.push({role:'user',content:msg},{role:'assistant',content:d.text})}
  }catch(e){addTextMsg('sys','请求失败: '+e.message)}
  inp.disabled=false;inp.focus();
}

// ─── viz ───
function setupViz(){
  const c=document.getElementById('viz');c.innerHTML='';
  for(let i=0;i<32;i++){const b=document.createElement('div');b.className='bar';b.style.height='2px';c.appendChild(b)}
  const ad=new Uint8Array(128);
  !function draw(){
    if(!analyser)return;
    analyser.getByteFrequencyData(ad);
    const bs=c.children, st=Math.floor(ad.length/bs.length);
    let s=0;
    for(let i=0;i<bs.length;i++){const v=ad[i*st]||0;bs[i].style.height=Math.max(2,v/4)+'px';s+=v}
    const vol=Math.min(100,(s/bs.length/255)*300);
    const m=document.getElementById('vol');
    m.style.width=vol+'%';
    m.style.background=vol>8?'linear-gradient(90deg,#2b8a3e,#51cf66)':'linear-gradient(90deg,#3b5bdb,#22b8cf)';
    if(active)requestAnimationFrame(draw);
  }()
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 全双工 Web Demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=9060)
    parser.add_argument("--no-ssl", action="store_true")
    parser.add_argument("--ssl-cert", default=str(BASE_DIR / "ssl_cert.pem"))
    parser.add_argument("--ssl-key", default=str(BASE_DIR / "ssl_key.pem"))
    args = parser.parse_args()

    state["llama_host"] = args.llama_host
    state["llama_port"] = args.llama_port

    use_ssl = not args.no_ssl and os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key)
    proto = "https" if use_ssl else "http"

    print("=" * 60)
    print("  MiniCPM-o 4.5 全双工 Web Demo (WebSocket)")
    print(f"  {proto}://0.0.0.0:{args.port}")
    print(f"  llama-server: {args.llama_host}:{args.llama_port}")
    print(f"  工作目录: {WORK_DIR}")
    print("=" * 60, flush=True)

    ssl_ctx = None
    if use_ssl:
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(args.ssl_cert, args.ssl_key)

    app.run(host=args.host, port=args.port, ssl_context=ssl_ctx,
            threaded=True, debug=False)


if __name__ == "__main__":
    main()
