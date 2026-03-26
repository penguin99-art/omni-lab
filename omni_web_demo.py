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

SHM_BASE = Path("/dev/shm")
_tmpdir_base = SHM_BASE if SHM_BASE.is_dir() else Path(tempfile.gettempdir())
WORK_DIR = Path(tempfile.mkdtemp(prefix="omni_web_", dir=str(_tmpdir_base)))
AUDIO_DIR = WORK_DIR / "audio"
FRAME_DIR = WORK_DIR / "frames"
OUTPUT_DIR = WORK_DIR / "output"
for d in [AUDIO_DIR, FRAME_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
sock = Sock(app)

SCENARIOS = {
    "default": {
        "name": "自由对话",
        "icon": "💬",
        "system_prompt": "你是一个友好的中文助手，请用普通话回答。",
        "description": "通用全双工语音对话模式",
    },
    "danmaku": {
        "name": "AI 弹幕",
        "icon": "📺",
        "system_prompt": (
            "你是一个实时弹幕评论员，风格类似B站弹幕。"
            "你通过摄像头观察画面，不断生成简短、有趣、犀利的弹幕评论。"
            "每条弹幕不超过15个字，要幽默、要有梗、要接地气。"
            "可以吐槽、可以夸、可以发问、可以玩谐音梗。"
            "像网友在直播间发弹幕一样自然随意。用普通话。"
        ),
        "description": "B站弹幕风格实时评论 — 摄像头画面上飘过AI弹幕",
    },
    "pictionary": {
        "name": "你画我猜",
        "icon": "🎨",
        "system_prompt": (
            "你在玩你画我猜游戏。用户会在纸上画画或展示物体给你看。"
            "你的任务是尽快猜出他画的/展示的是什么。"
            "只回答你的猜测，用简短的词语，比如'猫''汽车''太阳'。"
            "如果不确定就继续猜，每次给出一个新的猜测。"
            "猜对时说'我猜是XXX！'。用普通话。"
        ),
        "description": "画画或展示物体，AI实时猜你画了什么",
    },
    "expression": {
        "name": "表情挑战",
        "icon": "🎭",
        "system_prompt": (
            "你是表情挑战的裁判。游戏规则："
            "1. 你先给用户一个表情指令，如'请做出惊讶的表情'、'请做出生气的样子'等。"
            "2. 用户对着摄像头做表情。"
            "3. 你观察用户的表情，给出1-10分的评分和简短点评。"
            "4. 然后给出下一个表情挑战。"
            "表情种类：开心、惊讶、生气、悲伤、害怕、厌恶、困惑、骄傲、害羞、兴奋、"
            "鬼脸、wink、无辜、高冷、卖萌。"
            "评分要有趣，点评要毒舌但友好。用普通话。"
        ),
        "description": "AI出题你来演 — 做表情、拿评分、PK表情包",
    },
}

state = {
    "llama_host": "127.0.0.1",
    "llama_port": 9060,
    "initialized": False,
    "prefill_cnt": 1,
    "audio_idx": 0,
    "frame_idx": 0,
    "round_idx": 0,
    "scenario": "default",
    "timing_history": [],
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


def _read_wav_b64(wav_path: Path):
    """Read a WAV file and return dict with base64-encoded PCM + sample rate."""
    try:
        if wav_path.stat().st_size < 100:
            return None
        data, sr = sf.read(str(wav_path), dtype="float32")
        pcm_b64 = base64.b64encode(data.astype(np.float32).tobytes()).decode("ascii")
        return {"pcm": pcm_b64, "sr": sr}
    except Exception:
        return None


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
        entry = _read_wav_b64(wav_path)
        if entry is None:
            break
        entry["i"] = cursor
        results.append(entry)
        cursor += 1
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
    empty_listen_count = [0]
    chunk_since_reset = [0]
    MAX_CHUNKS_BEFORE_RESET = 500

    send_lock = threading.Lock()
    ws_closed = [False]

    conversation_history = []
    cur_bot_text = [""]
    MAX_HISTORY_TURNS = 8

    def safe_ws_send(data):
        if ws_closed[0]:
            return False
        try:
            with send_lock:
                ws.send(data)
            return True
        except Exception:
            ws_closed[0] = True
            return False

    def tts_pusher():
        """Background thread: polls for TTS WAV files and pushes them via WS."""
        print("[TTS] pusher thread started", flush=True)
        while not tts_stop.is_set() and not ws_closed[0]:
            new_wavs, wav_cursor[0] = _collect_new_wavs(wav_cursor[0])
            if new_wavs:
                ok = safe_ws_send(json.dumps({"type": "audio", "chunks": new_wavs}, ensure_ascii=False))
                if not ok:
                    break
                print(f"[TTS] pushed {len(new_wavs)} chunk(s), cursor={wav_cursor[0]}", flush=True)
            tts_stop.wait(0.08)
        print("[TTS] pusher thread exiting", flush=True)

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
                safe_ws_send(json.dumps({"type": "error", "error": "invalid json"}))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "prepare":
                try:
                    media_type = msg.get("media_type", 2)
                    duplex = msg.get("duplex", True)
                    scenario_id = msg.get("scenario", "default")
                    with state_lock:
                        state["scenario"] = scenario_id
                        state["timing_history"] = []

                    import shutil
                    tts_dir = OUTPUT_DIR / "tts_wav"
                    if tts_dir.exists():
                        shutil.rmtree(tts_dir, ignore_errors=True)
                    tts_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[prepare] scenario={scenario_id}, cleaned TTS WAV files", flush=True)

                    _do_init(media_type, duplex)
                    wav_cursor[0] = 0

                    if tts_thread is None or not tts_thread.is_alive():
                        tts_stop.clear()
                        tts_thread = threading.Thread(target=tts_pusher, daemon=True)
                        tts_thread.start()

                    safe_ws_send(json.dumps({"type": "prepared", "scenario": scenario_id}))
                except Exception as e:
                    safe_ws_send(json.dumps({"type": "error", "error": str(e)}))

            elif msg_type == "audio_chunk":
                audio_b64 = msg.get("audio")
                frame_b64 = msg.get("frame")
                if not audio_b64:
                    safe_ws_send(json.dumps({"type": "error", "error": "no audio"}))
                    continue

                def do_auto_reset(reason):
                    print(f"[AUTO-RESET] {reason}", flush=True)
                    if cur_bot_text[0].strip():
                        conversation_history.append({"role": "assistant", "text": cur_bot_text[0].strip()})
                        cur_bot_text[0] = ""
                    empty_listen_count[0] = 0
                    chunk_since_reset[0] = 0
                    try:
                        requests.post(llama_url("/v1/stream/reset"), json={}, timeout=10)
                        import shutil
                        tts_dir = OUTPUT_DIR / "tts_wav"
                        if tts_dir.exists():
                            shutil.rmtree(tts_dir, ignore_errors=True)
                        tts_dir.mkdir(parents=True, exist_ok=True)

                        _do_init(2, True)
                        wav_cursor[0] = 0

                        recent = conversation_history[-MAX_HISTORY_TURNS:]
                        if recent:
                            with state_lock:
                                scenario_id = state.get("scenario", "default")
                            base_prompt = SCENARIOS.get(scenario_id, SCENARIOS["default"])["system_prompt"]
                            history_lines = "\n".join(
                                f"{'用户' if t['role']=='user' else '助手'}：{t['text'][:80]}"
                                for t in recent
                            )
                            custom_prompt = (
                                f"流式全双工对话。{base_prompt}\n\n"
                                f"[之前的对话记录，请延续话题]\n{history_lines}"
                            )
                            ref_audio = str(BASE_DIR / "official_ref_audio.wav")
                            try:
                                r = requests.post(llama_url("/v1/stream/update_session_config"), json={
                                    "system_prompt": custom_prompt,
                                    "voice_audio": ref_audio,
                                }, timeout=30)
                                print(f"[AUTO-RESET] injected {len(recent)} turns into system prompt ({len(custom_prompt)} chars)", flush=True)
                            except Exception as e:
                                print(f"[AUTO-RESET] history injection failed (non-fatal): {e}", flush=True)

                        safe_ws_send(json.dumps({
                            "type": "result",
                            "text": "[上下文已自动刷新，继续对话]",
                            "is_listen": True,
                            "auto_reset": True,
                        }, ensure_ascii=False))
                        print("[AUTO-RESET] session re-initialized successfully", flush=True)
                    except Exception as re_err:
                        print(f"[AUTO-RESET] failed: {re_err}", flush=True)

                try:
                    chunk_since_reset[0] += 1

                    if chunk_since_reset[0] >= MAX_CHUNKS_BEFORE_RESET:
                        do_auto_reset(f"proactive reset at {chunk_since_reset[0]} chunks (context approaching limit)")
                        continue

                    t0 = time.monotonic()
                    _do_prefill(audio_b64, frame_b64)
                    t1 = time.monotonic()
                    text, is_listen, is_end = _do_decode()
                    t2 = time.monotonic()

                    prefill_ms = (t1 - t0) * 1000
                    decode_ms = (t2 - t1) * 1000
                    total_ms = (t2 - t0) * 1000

                    if not is_listen and text and text.strip():
                        cur_bot_text[0] += text
                    if is_listen and cur_bot_text[0].strip():
                        conversation_history.append({"role": "assistant", "text": cur_bot_text[0].strip()})
                        if len(conversation_history) > MAX_HISTORY_TURNS * 2:
                            conversation_history[:] = conversation_history[-MAX_HISTORY_TURNS * 2:]
                        cur_bot_text[0] = ""

                    if is_listen and (not text or not text.strip()):
                        empty_listen_count[0] += 1
                    else:
                        empty_listen_count[0] = 0

                    status = "LISTEN" if is_listen else "SPEAK"
                    print(f"[{status}] prefill={prefill_ms:.0f}ms decode={decode_ms:.0f}ms "
                          f"total={total_ms:.0f}ms text='{text[:40]}'"
                          f" chunks={chunk_since_reset[0]}/{MAX_CHUNKS_BEFORE_RESET}"
                          f" empty={empty_listen_count[0]}",
                          flush=True)

                    timing = {
                        "prefill": round(prefill_ms),
                        "decode": round(decode_ms),
                        "total": round(total_ms),
                        "ts": round(time.time() * 1000),
                    }
                    with state_lock:
                        state["timing_history"].append(timing)
                        if len(state["timing_history"]) > 500:
                            state["timing_history"] = state["timing_history"][-200:]

                    safe_ws_send(json.dumps({
                        "type": "result",
                        "text": text,
                        "is_listen": is_listen,
                        "end_of_turn": is_end,
                        "timing": timing,
                    }, ensure_ascii=False))

                    if empty_listen_count[0] >= 30:
                        do_auto_reset("30 consecutive empty LISTEN — model stuck")

                except Exception as e:
                    print(f"[ERROR] audio_chunk: {e}", flush=True)
                    if ws_closed[0]:
                        break
                    safe_ws_send(json.dumps({"type": "error", "error": str(e)}))

            elif msg_type == "user_text":
                user_t = msg.get("text", "").strip()
                if user_t:
                    conversation_history.append({"role": "user", "text": user_t})
                    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
                        conversation_history[:] = conversation_history[-MAX_HISTORY_TURNS * 2:]

            elif msg_type == "stop":
                try:
                    requests.post(llama_url("/v1/stream/break"), json={}, timeout=5)
                except Exception:
                    pass
                safe_ws_send(json.dumps({"type": "stopped"}))
                break

            elif msg_type == "reset":
                try:
                    requests.post(llama_url("/v1/stream/reset"), json={}, timeout=10)
                    with state_lock:
                        state["round_idx"] = 0
                        state["prefill_cnt"] = 1
                    wav_cursor[0] = 0
                    safe_ws_send(json.dumps({"type": "reset_done"}))
                except Exception as e:
                    safe_ws_send(json.dumps({"type": "error", "error": str(e)}))

    finally:
        tts_stop.set()
        if tts_thread and tts_thread.is_alive():
            tts_thread.join(timeout=2)
        try:
            requests.post(llama_url("/v1/stream/break"), json={}, timeout=3)
        except Exception:
            pass
        print("[WS] duplex session ended (stream break sent)", flush=True)


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


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify({k: {"name": v["name"], "icon": v["icon"], "description": v["description"]}
                    for k, v in SCENARIOS.items()})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Text chat via llama-server's /v1/chat/completions"""
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    history = data.get("history", [])
    scenario_id = data.get("scenario", state.get("scenario", "default"))
    sys_prompt = SCENARIOS.get(scenario_id, SCENARIOS["default"])["system_prompt"]
    messages = [{"role": "system", "content": sys_prompt}]
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


@app.route("/api/timing_history")
def api_timing_history():
    """Return recent timing data for the profiling dashboard."""
    with state_lock:
        return jsonify(state["timing_history"][-200:])


@app.route("/")
def index():
    return HTML_PAGE


# ─── 前端 ────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>omni-lab · MiniCPM-o 4.5</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0f;color:#e0e0e8;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:14px 24px;border-bottom:1px solid #2a2a4a;display:flex;align-items:center;gap:16px}
.hdr h1{font-size:17px;color:#a0c4ff;white-space:nowrap}.hdr p{font-size:11px;color:#888;margin-top:1px}
.scenario-sel{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
.sc-btn{padding:6px 12px;border:1px solid #2a2a4a;border-radius:20px;font-size:12px;cursor:pointer;background:#12121f;color:#aaa;transition:.2s;white-space:nowrap}
.sc-btn:hover{border-color:#4c6ef5;color:#fff}.sc-btn.active{background:#3b5bdb;color:#fff;border-color:#3b5bdb}
.main{display:flex;gap:12px;padding:12px;max-width:1600px;margin:0 auto;height:calc(100vh - 64px)}
.pnl{background:#12121f;border:1px solid #2a2a4a;border-radius:10px;padding:14px;display:flex;flex-direction:column}
.left{flex:0 0 340px}.center{flex:1;min-width:0}.right-prof{flex:0 0 280px}
.pnl h2{font-size:12px;font-weight:600;color:#7b8cde;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
video{width:100%;border-radius:8px;background:#000;margin-bottom:8px}
.ctrls{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.btn{padding:7px 16px;border:none;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;transition:.2s}
.btn-p{background:#3b5bdb;color:#fff}.btn-p:hover{background:#4c6ef5}
.btn-d{background:#c92a2a;color:#fff}.btn-d:hover{background:#e03131}
.btn-s{background:#2a2a4a;color:#ccc}.btn-s:hover{background:#3a3a5a}
.btn:disabled{opacity:.4;cursor:not-allowed}
.sbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.badge{padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600}
.b-ok{background:#0b7285;color:#e3fafc}.b-err{background:#862e2e;color:#ffc9c9}
.b-info{background:#1a1a3a;color:#aaa}
.chat{flex:1;overflow-y:auto;padding:6px;display:flex;flex-direction:column;gap:6px}
.msg{padding:8px 12px;border-radius:9px;max-width:88%;font-size:13px;line-height:1.5;word-break:break-word}
.msg-user{background:#1e3a5f;color:#a0c4ff;align-self:flex-end}
.msg-bot{background:#1a2a1a;color:#a0ffb0;align-self:flex-start}
.msg-sys{background:#2a2a3a;color:#888;align-self:center;font-size:11px;padding:4px 10px}
.tabBtn.active{background:#3b5bdb!important;color:#fff!important}
.meter{height:4px;background:#1a1a2e;border-radius:3px;margin:4px 0;overflow:hidden}
.meter-bar{height:100%;transition:width .1s;width:0%}
.viz{display:flex;align-items:flex-end;gap:2px;height:30px;margin:4px 0}
.viz .bar{width:3px;background:#3b5bdb;border-radius:2px;transition:height 50ms}
.si{padding:7px 12px;border-radius:7px;text-align:center;font-size:13px;font-weight:600;margin-bottom:8px;transition:.3s}
.si-listen{background:#1a2a1a;color:#51cf66;border:1px solid #2b8a3e}
.si-think{background:#2a2a1a;color:#ffd43b;border:1px solid #5c4813}
.si-speak{background:#1a1a3a;color:#748ffc;border:1px solid #3b5bdb}
.si-idle{background:#1a1a1a;color:#666;border:1px solid #333}
#log{font-family:'Cascadia Code','Fira Code',monospace;font-size:10px;color:#555;max-height:100px;overflow-y:auto;padding:5px;background:#0a0a14;border-radius:5px;margin-top:6px}
.scenario-desc{font-size:11px;color:#888;padding:6px 10px;background:#0f0f1a;border-radius:6px;margin-bottom:8px;line-height:1.4}
.prof-card{background:#0f0f1a;border-radius:8px;padding:10px;margin-bottom:8px}
.prof-card h3{font-size:11px;color:#7b8cde;margin-bottom:6px;text-transform:uppercase}
.prof-val{font-size:22px;font-weight:700;color:#a0c4ff}
.prof-unit{font-size:11px;color:#666;margin-left:4px}
.prof-sub{font-size:10px;color:#555;margin-top:2px}
canvas.chart{width:100%;height:80px;border-radius:6px;background:#0a0a14;margin-top:4px}
.live-sub{padding:8px 14px;background:#1e3a5f;color:#a0c4ff;border-radius:8px;font-size:13px;margin-top:6px;opacity:0.7;min-height:0;transition:opacity .2s;line-height:1.5}
.cam-wrap{position:relative;overflow:hidden;border-radius:8px;margin-bottom:8px}
.cam-wrap video{margin-bottom:0;border-radius:0}
.danmaku-layer{position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;overflow:hidden;z-index:10;display:none}
.danmaku{position:absolute;white-space:nowrap;font-weight:700;font-size:20px;text-shadow:2px 2px 4px rgba(0,0,0,.9),0 0 8px rgba(0,0,0,.5);pointer-events:none;will-change:transform}
@keyframes dm-scroll{from{transform:translateX(0)}to{transform:translateX(calc(-100% - 100vw))}}
.game-panel{display:none;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:12px;padding:20px}
.game-top{display:flex;gap:32px;align-items:center;margin-bottom:8px}
.game-timer{font-size:56px;font-weight:900;color:#ffd43b;font-variant-numeric:tabular-nums;min-width:80px;text-align:center}
.game-timer.warn{color:#e03131;animation:pulse .5s infinite alternate}
@keyframes pulse{to{opacity:.5}}
.game-score-box{text-align:center}
.game-score-box .label{font-size:11px;color:#666;text-transform:uppercase}
.game-score-box .val{font-size:32px;font-weight:800;color:#51cf66}
.game-round-box{text-align:center}
.game-round-box .label{font-size:11px;color:#666;text-transform:uppercase}
.game-round-box .val{font-size:24px;font-weight:700;color:#a0c4ff}
.game-guess{font-size:48px;font-weight:900;color:#e0e0e8;text-align:center;padding:32px 24px;background:linear-gradient(135deg,#1a1a3a,#12122a);border-radius:16px;min-height:120px;display:flex;align-items:center;justify-content:center;width:100%;max-width:500px;border:2px solid #2a2a4a;transition:border-color .3s}
.game-guess.hot{border-color:#ffd43b;box-shadow:0 0 20px rgba(255,212,59,.2)}
.game-btns{display:flex;gap:12px;margin-top:8px}
.game-btns .btn{padding:10px 28px;font-size:14px;border-radius:10px}
.game-history{width:100%;max-width:500px;max-height:140px;overflow-y:auto;font-size:12px;color:#555;margin-top:12px}
.game-history div{padding:6px 10px;border-bottom:1px solid #1a1a2e;display:flex;justify-content:space-between}
.expr-panel{display:none;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:16px;padding:20px}
.expr-card{background:linear-gradient(135deg,#1a1a3a,#12122a);border-radius:20px;padding:32px 40px;text-align:center;min-width:300px;border:2px solid #2a2a4a}
.expr-emoji{font-size:72px;display:block;margin-bottom:12px}
.expr-target{font-size:28px;font-weight:800;color:#ffd43b}
.expr-hint{font-size:13px;color:#666;margin-top:8px}
.expr-result{text-align:center;min-height:120px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}
.expr-score-big{font-size:80px;font-weight:900;background:linear-gradient(135deg,#51cf66,#22b8cf);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}
.expr-comment{font-size:15px;color:#ccc;max-width:320px;line-height:1.5}
.expr-stats{display:flex;gap:32px;margin-top:8px}
.expr-stats .stat{text-align:center}
.expr-stats .stat .label{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px}
.expr-stats .stat .val{font-size:28px;font-weight:800;color:#a0c4ff}
.expr-stats .stat .val.combo{color:#ffd43b}
.expr-progress{width:100%;max-width:300px;height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden}
.expr-progress-bar{height:100%;background:linear-gradient(90deg,#3b5bdb,#51cf66);transition:width .3s;border-radius:3px}
@media(max-width:1100px){.main{flex-direction:column}.left,.right-prof{flex:0 0 auto}}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <h1>omni-lab · MiniCPM-o 4.5</h1>
    <p>全双工多模态 · WebSocket · 场景演示</p>
  </div>
  <div class="scenario-sel" id="scenarioSel"></div>
</div>

<div class="main">
  <div class="pnl left">
    <h2>输入</h2>
    <div class="si si-idle" id="si">待机</div>
    <div class="scenario-desc" id="scenarioDesc">通用全双工语音对话模式</div>
    <div class="cam-wrap">
      <video id="cam" autoplay muted playsinline></video>
      <div class="danmaku-layer" id="danmakuLayer"></div>
    </div>
    <div class="viz" id="viz"></div>
    <div class="meter"><div class="meter-bar" id="vol"></div></div>
    <div class="ctrls">
      <button class="btn btn-p" id="btnGo" onclick="go()">开始对话</button>
      <button class="btn btn-d" id="btnStop" onclick="stop()" disabled>停止</button>
      <button class="btn btn-s" onclick="doReset()">重置</button>
    </div>
    <div class="ctrls">
      <label style="font-size:11px;color:#888;display:flex;align-items:center;gap:4px">
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

  <div class="pnl center">
    <div id="tabBar" style="display:flex;gap:8px;margin-bottom:8px">
      <button class="btn btn-s tabBtn active" onclick="switchTab('voice')" id="tabVoice">语音对话</button>
      <button class="btn btn-s tabBtn" onclick="switchTab('text')" id="tabText">文本对话</button>
    </div>
    <div id="voiceTab" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2 id="voiceTitle">语音对话</h2>
      <div class="chat" id="chat"></div>
      <div id="liveSubtitle" class="live-sub" style="display:none"></div>
    </div>
    <div id="textTab" style="flex:1;display:none;flex-direction:column;min-height:0">
      <h2>文本对话</h2>
      <div class="chat" id="textChat"></div>
      <div style="display:flex;gap:8px;margin-top:8px">
        <input id="txtInput" type="text" placeholder="输入消息..."
               style="flex:1;padding:8px 12px;border-radius:7px;border:1px solid #3a3a5a;background:#1a1a2e;color:#e0e0e8;font-size:13px"
               onkeydown="if(event.key==='Enter')sendText()">
        <button class="btn btn-p" onclick="sendText()">发送</button>
      </div>
    </div>

    <!-- Pictionary Game UI -->
    <div id="gameTab" class="game-panel">
      <div class="game-top">
        <div class="game-round-box"><div class="label">Round</div><div class="val" id="gameRound">1</div></div>
        <div class="game-timer" id="gameTimer">30</div>
        <div class="game-score-box"><div class="label">Score</div><div class="val" id="gameScore">0</div></div>
      </div>
      <div style="font-size:13px;color:#888;margin-bottom:4px">AI 正在猜...</div>
      <div class="game-guess" id="gameGuess">🎨 展示给我看</div>
      <div class="game-btns">
        <button class="btn btn-p" onclick="gameCorrect()" id="btnCorrect">✅ 猜对了!</button>
        <button class="btn btn-s" onclick="gameSkip()">⏭ 跳过</button>
        <button class="btn btn-s" onclick="gameRestart()">🔄 重新开始</button>
      </div>
      <div class="game-history" id="gameHistory"></div>
    </div>

    <!-- Expression Challenge UI -->
    <div id="exprTab" class="expr-panel">
      <div class="expr-card" id="exprCard">
        <span class="expr-emoji" id="exprEmoji">🎭</span>
        <div class="expr-target" id="exprTarget">等待开始...</div>
        <div class="expr-hint" id="exprHint">AI 会给你表情挑战</div>
      </div>
      <div class="expr-result" id="exprResult">
        <div class="expr-score-big" id="exprScoreBig" style="display:none">--</div>
        <div class="expr-comment" id="exprComment"></div>
      </div>
      <div class="expr-stats">
        <div class="stat"><div class="label">总分</div><div class="val" id="exprTotalScore">0</div></div>
        <div class="stat"><div class="label">连击</div><div class="val combo" id="exprCombo">0</div></div>
        <div class="stat"><div class="label">轮次</div><div class="val" id="exprRounds">0</div></div>
      </div>
      <div class="expr-progress"><div class="expr-progress-bar" id="exprProgress" style="width:0%"></div></div>
      <button class="btn btn-p" onclick="exprNext()" id="btnExprNext" style="margin-top:8px">⏭ 下一个挑战</button>
    </div>
  </div>

  <div class="pnl right-prof">
    <h2>延迟 Profiling</h2>
    <div class="prof-card">
      <h3>端到端延迟</h3>
      <div><span class="prof-val" id="profTotal">--</span><span class="prof-unit">ms</span></div>
      <div class="prof-sub" id="profTotalSub">avg / p95 / max</div>
    </div>
    <div class="prof-card">
      <h3>Prefill</h3>
      <div><span class="prof-val" id="profPrefill">--</span><span class="prof-unit">ms</span></div>
      <div class="prof-sub" id="profPrefillSub">--</div>
    </div>
    <div class="prof-card">
      <h3>Decode</h3>
      <div><span class="prof-val" id="profDecode">--</span><span class="prof-unit">ms</span></div>
      <div class="prof-sub" id="profDecodeSub">--</div>
    </div>
    <div class="prof-card">
      <h3>延迟趋势 (最近 50 轮)</h3>
      <canvas class="chart" id="chartCanvas" width="260" height="80"></canvas>
    </div>
    <div class="prof-card">
      <h3>统计</h3>
      <div class="prof-sub" id="profStats">总轮次: 0</div>
    </div>
  </div>
</div>

<script>
const CHUNK_MS=1000, PLAYBACK_SR=24000, PLAYBACK_DELAY_MS=250;
const FRAME_EVERY_N=5;
let ws=null, mediaStream=null, audioCtx=null, analyser=null;
let active=false, timer=null, micChunks=[];
let playCtx=null, nextPlayTime=0, pendingBufs=0;
let inflight=0, chunkCount=0;
let curBotEl=null;
let recognition=null;
let currentScenario='default';
let scenarios={};
let timingData=[];
let phase='idle';

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

function setPhase(p){
  if(phase===p)return;
  phase=p;
  if(p==='listen') setSt('listen','正在听...');
  else if(p==='speak') setSt('speak','回复中...');
  else setSt('idle','待机');
}

// ─── live subtitle (interim speech) ───
function showLiveSub(text){
  const el=document.getElementById('liveSubtitle');
  el.textContent=text;
  el.style.display='block';
}
function hideLiveSub(){
  const el=document.getElementById('liveSubtitle');
  el.style.display='none';
  el.textContent='';
}

async function loadScenarios(){
  try{
    const r=await fetch('/api/scenarios');
    scenarios=await r.json();
    const sel=document.getElementById('scenarioSel');
    sel.innerHTML='';
    for(const[k,v]of Object.entries(scenarios)){
      const b=document.createElement('button');
      b.className='sc-btn'+(k===currentScenario?' active':'');
      b.textContent=v.icon+' '+v.name;
      b.onclick=()=>selectScenario(k);
      b.dataset.id=k;
      sel.appendChild(b);
    }
  }catch(_){}
}

function selectScenario(id){
  if(active){log('请先停止当前对话再切换场景');return}
  currentScenario=id;
  document.querySelectorAll('.sc-btn').forEach(b=>{
    b.classList.toggle('active',b.dataset.id===id);
  });
  const sc=scenarios[id]||{};
  document.getElementById('scenarioDesc').textContent=sc.description||'';
  document.getElementById('voiceTitle').textContent=(sc.icon||'')+' '+(sc.name||'语音对话');
  log('切换场景: '+(sc.name||id));

  const voiceTab=document.getElementById('voiceTab');
  const textTab=document.getElementById('textTab');
  const gameTab=document.getElementById('gameTab');
  const exprTab=document.getElementById('exprTab');
  const danmakuLayer=document.getElementById('danmakuLayer');
  const tabBar=document.getElementById('tabBar');

  voiceTab.style.display='none';
  textTab.style.display='none';
  gameTab.style.display='none';
  exprTab.style.display='none';
  danmakuLayer.style.display='none';
  tabBar.style.display='flex';

  if(id==='danmaku'){
    voiceTab.style.display='flex';
    danmakuLayer.style.display='block';
  } else if(id==='pictionary'){
    gameTab.style.display='flex';
    tabBar.style.display='none';
    gameReset();
  } else if(id==='expression'){
    exprTab.style.display='flex';
    tabBar.style.display='none';
    exprReset();
  } else {
    voiceTab.style.display='flex';
  }
}

async function poll(){try{const r=await fetch('/api/status'),d=await r.json();document.getElementById('bSrv').textContent=`Server: ${d.server_ok?'OK':'OFF'}`;document.getElementById('bSrv').className=`badge ${d.server_ok?'b-ok':'b-err'}`;document.getElementById('bRnd').textContent=`轮次: ${d.round_idx}`}catch(_){}}
setInterval(poll,5000);poll();
loadScenarios();

// ─── profiling ───
function updateProfiling(t){
  timingData.push(t);
  if(timingData.length>200)timingData=timingData.slice(-100);
  const n=timingData.length;
  const totals=timingData.map(x=>x.total), prefills=timingData.map(x=>x.prefill), decodes=timingData.map(x=>x.decode);
  const avg=a=>Math.round(a.reduce((s,x)=>s+x,0)/a.length);
  const p95=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length*0.95)]};
  const mx=a=>Math.max(...a);

  document.getElementById('profTotal').textContent=avg(totals);
  document.getElementById('profTotalSub').textContent=`avg ${avg(totals)} / p95 ${p95(totals)} / max ${mx(totals)}`;
  document.getElementById('profPrefill').textContent=avg(prefills);
  document.getElementById('profPrefillSub').textContent=`avg ${avg(prefills)} / p95 ${p95(prefills)} / max ${mx(prefills)}`;
  document.getElementById('profDecode').textContent=avg(decodes);
  document.getElementById('profDecodeSub').textContent=`avg ${avg(decodes)} / p95 ${p95(decodes)} / max ${mx(decodes)}`;
  document.getElementById('profStats').textContent=`总轮次: ${n} · 最近avg: ${avg(totals.slice(-10))}ms`;
  drawChart();
}

function drawChart(){
  const canvas=document.getElementById('chartCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const recent=timingData.slice(-50);
  if(recent.length<2)return;
  const maxV=Math.max(...recent.map(x=>x.total),100);

  ctx.strokeStyle='#1a1a3a'; ctx.lineWidth=0.5;
  for(let i=0;i<4;i++){const y=H*(i/4);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}

  const drawLine=(data,color)=>{
    ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath();
    data.forEach((v,i)=>{
      const x=i/(data.length-1)*W, y=H-v/maxV*H*0.9;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
  };

  drawLine(recent.map(x=>x.prefill),'#51cf66');
  drawLine(recent.map(x=>x.decode),'#748ffc');
  drawLine(recent.map(x=>x.total),'#ffd43b');

  ctx.font='9px sans-serif';
  ctx.fillStyle='#51cf66';ctx.fillText('prefill',4,10);
  ctx.fillStyle='#748ffc';ctx.fillText('decode',50,10);
  ctx.fillStyle='#ffd43b';ctx.fillText('total',96,10);
  ctx.fillStyle='#444';ctx.fillText(maxV+'ms',W-35,10);
}

// ═══════════════════════════════════════════════════════════
// ─── Danmaku (弹幕) ───
// ═══════════════════════════════════════════════════════════
const DM_COLORS=['#fff','#ffd43b','#51cf66','#ff6b6b','#748ffc','#22b8cf','#f783ac','#ff922b','#da77f2'];
let dmTrackUsed=new Array(12).fill(0);

function addDanmaku(text){
  if(!text||!text.trim())return;
  const layer=document.getElementById('danmakuLayer');
  if(layer.style.display==='none')return;
  const el=document.createElement('div');
  el.className='danmaku';
  el.textContent=text.trim();
  el.style.color=DM_COLORS[Math.floor(Math.random()*DM_COLORS.length)];
  const fontSize=16+Math.floor(Math.random()*10);
  el.style.fontSize=fontSize+'px';

  const now=Date.now();
  let track=0, minTime=Infinity;
  for(let i=0;i<dmTrackUsed.length;i++){
    if(dmTrackUsed[i]<now){track=i;break}
    if(dmTrackUsed[i]<minTime){minTime=dmTrackUsed[i];track=i}
  }
  const topPct=4+track*(88/dmTrackUsed.length);
  el.style.top=topPct+'%';
  dmTrackUsed[track]=now+4000;

  const layerW=layer.offsetWidth||600;
  el.style.right='-50%';
  el.style.animation=`dm-scroll ${5+Math.random()*2}s linear forwards`;

  layer.appendChild(el);
  el.addEventListener('animationend',()=>el.remove());
  setTimeout(()=>{if(el.parentNode)el.remove()},10000);
}

// ═══════════════════════════════════════════════════════════
// ─── Pictionary (你画我猜) ───
// ═══════════════════════════════════════════════════════════
let gameState={round:1,score:0,timer:30,interval:null,history:[],currentGuess:'',guessAccum:''};

function gameReset(){
  gameState={round:1,score:0,timer:30,interval:null,history:[],currentGuess:'',guessAccum:''};
  document.getElementById('gameRound').textContent='1';
  document.getElementById('gameScore').textContent='0';
  document.getElementById('gameTimer').textContent='30';
  document.getElementById('gameTimer').classList.remove('warn');
  document.getElementById('gameGuess').textContent='🎨 展示给我看';
  document.getElementById('gameGuess').classList.remove('hot');
  document.getElementById('gameHistory').innerHTML='';
}

function gameStartTimer(){
  if(gameState.interval)clearInterval(gameState.interval);
  gameState.timer=30;
  document.getElementById('gameTimer').textContent='30';
  document.getElementById('gameTimer').classList.remove('warn');
  gameState.interval=setInterval(()=>{
    gameState.timer--;
    const el=document.getElementById('gameTimer');
    el.textContent=gameState.timer;
    if(gameState.timer<=10)el.classList.add('warn');
    if(gameState.timer<=0){
      clearInterval(gameState.interval);
      gameState.interval=null;
      gameAddHistory('⏰ 时间到','--');
      gameNextRound();
    }
  },1000);
}

function gameUpdateGuess(text){
  if(!text)return;
  gameState.guessAccum+=text;
  gameState.currentGuess=gameState.guessAccum.trim();
  const el=document.getElementById('gameGuess');
  el.textContent=gameState.currentGuess||'🤔 思考中...';
  el.classList.add('hot');
  setTimeout(()=>el.classList.remove('hot'),500);
  if(!gameState.interval)gameStartTimer();
}

function gameFinishGuess(){
  gameState.guessAccum='';
}

function gameCorrect(){
  gameState.score++;
  document.getElementById('gameScore').textContent=gameState.score;
  const timeUsed=30-gameState.timer;
  gameAddHistory('✅ '+gameState.currentGuess, timeUsed+'s');
  if(gameState.interval){clearInterval(gameState.interval);gameState.interval=null}
  gameNextRound();
}

function gameSkip(){
  gameAddHistory('⏭ 跳过','--');
  if(gameState.interval){clearInterval(gameState.interval);gameState.interval=null}
  gameNextRound();
}

function gameNextRound(){
  gameState.round++;
  gameState.currentGuess='';
  gameState.guessAccum='';
  document.getElementById('gameRound').textContent=gameState.round;
  document.getElementById('gameGuess').textContent='🎨 展示下一个';
  document.getElementById('gameGuess').classList.remove('hot');
  document.getElementById('gameTimer').textContent='30';
  document.getElementById('gameTimer').classList.remove('warn');
}

function gameRestart(){
  if(gameState.interval){clearInterval(gameState.interval);gameState.interval=null}
  gameReset();
}

function gameAddHistory(what,time){
  const el=document.getElementById('gameHistory');
  const d=document.createElement('div');
  d.innerHTML=`<span>R${gameState.round}: ${what}</span><span>${time}</span>`;
  el.insertBefore(d,el.firstChild);
}

// ═══════════════════════════════════════════════════════════
// ─── Expression Challenge (表情挑战) ───
// ═══════════════════════════════════════════════════════════
const EXPR_EMOJIS={'开心':'😄','惊讶':'😲','生气':'😠','悲伤':'😢','害怕':'😨','厌恶':'🤢',
  '困惑':'🤔','骄傲':'😏','害羞':'😳','兴奋':'🤩','鬼脸':'🤪','wink':'😜','无辜':'🥺','高冷':'😎','卖萌':'🥰'};
let exprState={totalScore:0,combo:0,rounds:0,maxRounds:10,currentTarget:'',lastScore:0,textAccum:''};

function exprReset(){
  exprState={totalScore:0,combo:0,rounds:0,maxRounds:10,currentTarget:'',lastScore:0,textAccum:''};
  document.getElementById('exprTotalScore').textContent='0';
  document.getElementById('exprCombo').textContent='0';
  document.getElementById('exprRounds').textContent='0';
  document.getElementById('exprProgress').style.width='0%';
  document.getElementById('exprEmoji').textContent='🎭';
  document.getElementById('exprTarget').textContent='等待AI出题...';
  document.getElementById('exprHint').textContent='开始对话后AI会给你表情挑战';
  document.getElementById('exprScoreBig').style.display='none';
  document.getElementById('exprComment').textContent='';
}

function exprUpdateJudgment(text){
  if(!text)return;
  exprState.textAccum+=text;
  const full=exprState.textAccum;

  const scoreMatch=full.match(/(\d+)\s*[\/／分]/);
  if(scoreMatch){
    const s=parseInt(scoreMatch[1]);
    if(s>=1&&s<=10){
      exprState.lastScore=s;
      document.getElementById('exprScoreBig').textContent=s;
      document.getElementById('exprScoreBig').style.display='block';
      const hue=s>=7?120:s>=4?60:0;
      document.getElementById('exprScoreBig').style.background=`linear-gradient(135deg,hsl(${hue},70%,60%),hsl(${hue+30},70%,50%))`;
      document.getElementById('exprScoreBig').style['-webkit-background-clip']='text';
    }
  }

  for(const[name,emoji]of Object.entries(EXPR_EMOJIS)){
    if(full.includes(name)){
      exprState.currentTarget=name;
      document.getElementById('exprEmoji').textContent=emoji;
      document.getElementById('exprTarget').textContent='请做: '+name;
      document.getElementById('exprHint').textContent='对着摄像头做出这个表情!';
      break;
    }
  }

  document.getElementById('exprComment').textContent=full.slice(-80);
}

function exprFinishJudgment(){
  const full=exprState.textAccum;
  if(exprState.lastScore>0){
    exprState.totalScore+=exprState.lastScore;
    exprState.rounds++;
    if(exprState.lastScore>=7) exprState.combo++;
    else exprState.combo=0;
    document.getElementById('exprTotalScore').textContent=exprState.totalScore;
    document.getElementById('exprCombo').textContent=exprState.combo;
    document.getElementById('exprRounds').textContent=exprState.rounds;
    document.getElementById('exprProgress').style.width=(exprState.rounds/exprState.maxRounds*100)+'%';
    exprState.lastScore=0;
  }
  document.getElementById('exprComment').textContent=full.slice(-120);
  exprState.textAccum='';
}

function exprNext(){
  exprState.textAccum='';
  exprState.lastScore=0;
  document.getElementById('exprScoreBig').style.display='none';
  document.getElementById('exprComment').textContent='等待AI出下一题...';
  document.getElementById('exprEmoji').textContent='🎭';
  document.getElementById('exprTarget').textContent='准备好了吗?';
  document.getElementById('exprHint').textContent='AI正在想下一个挑战...';
}

// ═══════════════════════════════════════════════════════════
// ─── Scenario-aware message routing ───
// ═══════════════════════════════════════════════════════════
function routeBotText(text, isFinish){
  if(currentScenario==='danmaku'){
    if(isFinish&&curBotEl){
      const full=curBotEl.textContent||'';
      full.split(/[，。！？、；\n]+/).filter(s=>s.trim()).forEach(s=>addDanmaku(s));
      finishBot();
    } else {
      appendBot(text);
    }
  } else if(currentScenario==='pictionary'){
    if(isFinish){gameFinishGuess();finishBot()}
    else{gameUpdateGuess(text);appendBot(text)}
  } else if(currentScenario==='expression'){
    if(isFinish){exprFinishJudgment();finishBot()}
    else{exprUpdateJudgment(text);appendBot(text)}
  } else {
    if(isFinish) finishBot();
    else appendBot(text);
  }
}

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
  if(phase!=='speak') setPhase('speak');
  src.onended=()=>{
    pendingBufs--;
    if(pendingBufs<=0) setPhase('listen');
  };
}

function playChunks(chunks){
  if(!chunks||!chunks.length)return;
  for(const c of chunks)scheduleAudio(c.pcm,c.sr);
}

// ─── WebSocket ───
function connectWs(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/ws/duplex`);
  ws.onopen=()=>{
    log('WS 已连接');
    const useCam=document.getElementById('chkCam').checked;
    ws.send(JSON.stringify({type:'prepare',media_type:useCam?2:1,duplex:true,scenario:currentScenario}));
  };
  ws.onclose=()=>{log('WS 断开');ws=null};
  ws.onerror=()=>log('WS 错误');
  ws.onmessage=e=>{
    let d;
    try{d=JSON.parse(e.data)}catch(_){return}

    if(d.type==='prepared'){
      const sc=scenarios[d.scenario||currentScenario]||{};
      addMsg('sys',(sc.icon||'')+' '+(sc.name||'模型')+'就绪，请说话');
      setPhase('listen');
      startLoop();
    } else if(d.type==='result'){
      inflight=Math.max(0,inflight-1);
      if(d.auto_reset){
        addMsg('sys','⟳ 上下文已刷新（对话记录保留在页面上）');
        finishBot();
      } else {
        if(d.text&&d.text.length>0) routeBotText(d.text, false);
        if(d.is_listen){
          routeBotText('', true);
          if(pendingBufs<=0) setPhase('listen');
        }
      }
      if(d.timing){
        document.getElementById('bTim').textContent=`P:${d.timing.prefill}ms D:${d.timing.decode}ms`;
        updateProfiling(d.timing);
      }
    } else if(d.type==='audio'){
      playChunks(d.chunks);
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
    if(inflight>=1)return;
    const samples=collectChunk();
    if(!samples||samples.length<100)return;
    const useCam=document.getElementById('chkCam').checked;
    chunkCount++;
    const audioB64=f32toWavB64(samples,16000);
    const msg={type:'audio_chunk',audio:audioB64};
    if(useCam&&chunkCount%FRAME_EVERY_N===0){const f=captureFrame();if(f)msg.frame=f}
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
    const c={audio:{sampleRate:16000,channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}};
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
  nextPlayTime=0;pendingBufs=0;inflight=0;timingData=[];

  active=true;
  document.getElementById('btnGo').disabled=true;
  document.getElementById('btnStop').disabled=false;
  setSt('think','正在初始化...');
  addMsg('sys','正在连接...');

  if(currentScenario==='pictionary')gameStartTimer();

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
          hideLiveSub();
          addMsg('user',t);
          finishBot();
          if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'user_text',text:t}));
        } else {
          showLiveSub(t);
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
  setPhase('idle');
  if(timer){clearInterval(timer);timer=null}
  if(gameState.interval){clearInterval(gameState.interval);gameState.interval=null}
  if(recognition){try{recognition.abort()}catch(_){};recognition=null}
  curBotEl=null;
  hideLiveSub();
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
  timingData=[];
  drawChart();
  if(currentScenario==='pictionary')gameReset();
  if(currentScenario==='expression')exprReset();
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
      body:JSON.stringify({message:msg,history:textHistory,scenario:currentScenario})});
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
