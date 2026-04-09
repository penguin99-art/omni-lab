"""Realtime voice+vision demo — Flask WebSocket server.

Protocol matches Parlor (audio_start / audio_chunk / audio_end) but adds
server-side ASR and streaming LLM tokens.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx
import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_sock import Sock
from simple_websocket.errors import ConnectionClosed

from asr import ASRUnavailableError, create_asr
from config import CONFIG
from tts import TTSUnavailableError, create_tts_backend

BASE_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")
sock = Sock(app)

ASR_ENGINE = create_asr(CONFIG)
TTS_ENGINE = create_tts_backend(CONFIG)

SENTENCE_RE = re.compile(r"(.+?[.!?。！？\n]+)(?=\s|$)", re.S)


class GenerationCancelled(Exception):
    pass


def _preview(text: str, limit: int = 120) -> str:
    normalized = " ".join((text or "").split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def log_line(stage: str, *, session: str = "", turn: str = "", **kw):
    parts = [f"stage={stage}"]
    if session:
        parts.append(f"session={session}")
    if turn:
        parts.append(f"turn={turn}")
    for k, v in kw.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v!r}")
    print("[realtime-demo] " + " ".join(parts), flush=True)


def extract_sentences(buf: str, flush_chars: int) -> tuple[str, list[str]]:
    sentences: list[str] = []
    cursor = 0
    for m in SENTENCE_RE.finditer(buf):
        sentences.append(m.group(1).strip())
        cursor = m.end()
    rest = buf[cursor:].lstrip()
    if not sentences and len(buf) >= flush_chars:
        cut = buf.rfind(" ", 0, flush_chars)
        if cut <= 0:
            cut = flush_chars
        sentences.append(buf[:cut].strip())
        rest = buf[cut:].lstrip()
    return rest, [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    active_cancel: threading.Event | None = None
    active_thread: threading.Thread | None = None
    ws_closed: bool = False
    conversation: list[dict] = field(default_factory=list)

    def cancel_current(self) -> bool:
        if self.active_cancel and not self.active_cancel.is_set():
            self.active_cancel.set()
            return True
        return False


# ---------------------------------------------------------------------------
# Ollama bridge with conversation history
# ---------------------------------------------------------------------------

MAX_HISTORY = 20  # keep last N user+assistant pairs


class OllamaBridge:
    def __init__(self, base_url: str, primary_model: str, fallback_models: list[str], system_prompt: str):
        self.base_url = base_url.rstrip("/")
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.system_prompt = system_prompt

    def _candidates(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for m in [self.primary_model, *self.fallback_models]:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _messages(self, text: str, image_b64: str, history: list[dict]) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(history[-MAX_HISTORY * 2 :])
        user_msg: dict = {"role": "user", "content": text}
        if image_b64:
            user_msg["images"] = [image_b64]
        msgs.append(user_msg)
        return msgs

    def _stream_one(
        self, model: str, text: str, image_b64: str, history: list[dict], cancel: threading.Event,
    ) -> Iterable[dict]:
        payload = {
            "model": model,
            "messages": self._messages(text, image_b64, history),
            "stream": True,
            "options": {"num_predict": 512},
        }
        with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if cancel.is_set():
                    raise GenerationCancelled()
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield {"model": model, "content": chunk}
                if data.get("done"):
                    break

    def stream_reply(
        self, text: str, image_b64: str, history: list[dict], cancel: threading.Event,
    ) -> Iterable[dict]:
        last_err: Exception | None = None
        for model in self._candidates():
            try:
                yield from self._stream_one(model, text, image_b64, history, cancel)
                return
            except GenerationCancelled:
                raise
            except Exception as exc:
                last_err = exc
                if image_b64:
                    continue
                raise
        if last_err:
            raise RuntimeError(str(last_err))


BRIDGE = OllamaBridge(CONFIG.ollama_host, CONFIG.model_name, CONFIG.fallback_models, CONFIG.system_prompt)


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

def send_event(ws, state: SessionState, payload: dict) -> bool:
    if state.ws_closed:
        return False
    with state.send_lock:
        try:
            ws.send(json.dumps(payload))
            return True
        except ConnectionClosed:
            state.ws_closed = True
            log_line("ws_closed", session=state.session_id)
            return False


def _strip_data_url(s: str) -> str:
    return s.split(",", 1)[1] if "," in s else s


# ---------------------------------------------------------------------------
# Turn processing
# ---------------------------------------------------------------------------

def start_turn(ws, state: SessionState, *, audio_b64: str | None = None,
               image_b64: str | None = None, text_input: str | None = None):
    if state.active_thread and state.active_thread.is_alive():
        state.cancel_current()
        state.active_thread.join(timeout=1.0)

    cancel = threading.Event()
    state.active_cancel = cancel
    t = threading.Thread(target=run_turn, args=(ws, state, cancel, audio_b64, image_b64, text_input), daemon=True)
    state.active_thread = t
    t.start()


def run_turn(ws, state: SessionState, cancel: threading.Event,
             audio_b64: str | None, image_b64: str | None, text_input: str | None):
    turn = uuid.uuid4().hex[:12]
    sid = state.session_id
    assistant_text = ""
    sentence_buf = ""
    audio_started = False
    tts_idx = 0
    tts_t0 = None

    try:
        # --- 1. Resolve user text ---
        if text_input is not None:
            user_text = text_input.strip()
            log_line("input_text", session=sid, turn=turn, text=_preview(user_text, 200))
        else:
            send_event(ws, state, {"type": "status", "phase": "transcribing"})
            wav_bytes = base64.b64decode(audio_b64 or "")
            asr_result = ASR_ENGINE.transcribe_wav_bytes(wav_bytes)
            user_text = asr_result.text.strip()
            log_line("asr_result", session=sid, turn=turn, text=_preview(user_text, 200),
                     reason=getattr(asr_result, "reason", ""), lang=getattr(asr_result, "language", ""))

        if not user_text:
            send_event(ws, state, {"type": "error", "message": "No speech detected."})
            return

        # Send transcript to client (Parlor-style: type=text with transcription)
        if not send_event(ws, state, {"type": "text", "transcription": user_text}):
            return

        # --- 2. LLM streaming ---
        send_event(ws, state, {"type": "status", "phase": "thinking"})
        img = image_b64 or ""
        active_model = CONFIG.model_name
        t0 = time.time()
        log_line("model_start", session=sid, turn=turn, model=active_model, has_image=bool(img))

        for ev in BRIDGE.stream_reply(user_text, img, state.conversation, cancel):
            if cancel.is_set():
                raise GenerationCancelled()
            active_model = ev["model"]
            chunk = ev["content"]
            assistant_text += chunk
            sentence_buf += chunk

            if not send_event(ws, state, {"type": "assistant_token", "text": chunk}):
                return

            sentence_buf, sentences = extract_sentences(sentence_buf, CONFIG.sentence_flush_chars)
            for sentence in sentences:
                if cancel.is_set():
                    raise GenerationCancelled()
                if tts_t0 is None:
                    tts_t0 = time.time()
                audio_started = _tts_sentence(ws, state, cancel, sentence, turn, audio_started, tts_idx)
                tts_idx += 1

        llm_time = time.time() - t0

        # Flush remaining
        if sentence_buf.strip() and not cancel.is_set():
            if tts_t0 is None:
                tts_t0 = time.time()
            audio_started = _tts_sentence(ws, state, cancel, sentence_buf.strip(), turn, audio_started, tts_idx)
            tts_idx += 1

        # Audio stream end
        if audio_started and not cancel.is_set():
            tts_time = round(time.time() - (tts_t0 or t0), 2)
            send_event(ws, state, {"type": "audio_end", "tts_time": tts_time})

        # Final text event
        log_line("model_done", session=sid, turn=turn, model=active_model,
                 chars=len(assistant_text), llm_s=f"{llm_time:.2f}")
        send_event(ws, state, {
            "type": "text", "text": assistant_text, "llm_time": round(llm_time, 2),
        })

        # Update conversation history
        state.conversation.append({"role": "user", "content": user_text})
        state.conversation.append({"role": "assistant", "content": assistant_text})

    except GenerationCancelled:
        log_line("cancelled", session=sid, turn=turn)
        if audio_started:
            send_event(ws, state, {"type": "audio_end"})
    except ASRUnavailableError as exc:
        log_line("asr_unavailable", session=sid, turn=turn, err=str(exc))
        send_event(ws, state, {"type": "error", "message": str(exc)})
    except Exception as exc:
        log_line("turn_error", session=sid, turn=turn, err=str(exc))
        send_event(ws, state, {"type": "error", "message": str(exc)})
    finally:
        state.active_cancel = None
        state.active_thread = None
        log_line("turn_end", session=sid, turn=turn)


def _tts_sentence(ws, state: SessionState, cancel: threading.Event,
                  text: str, turn: str, audio_started: bool, idx: int) -> bool:
    """Synthesize one sentence → PCM, stream to client. Returns updated audio_started."""
    if cancel.is_set():
        return audio_started
    try:
        pcm, sr = TTS_ENGINE.generate_pcm(text)
    except TTSUnavailableError as exc:
        log_line("tts_unavail", session=state.session_id, turn=turn, err=str(exc))
        return audio_started

    log_line("tts_chunk", session=state.session_id, turn=turn, idx=idx, sentence=_preview(text, 80))

    if not audio_started:
        if not send_event(ws, state, {"type": "audio_start", "sample_rate": sr}):
            return audio_started

    pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    send_event(ws, state, {
        "type": "audio_chunk",
        "audio": base64.b64encode(pcm_i16.tobytes()).decode("ascii"),
        "index": idx,
    })
    return True


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "config": CONFIG.public,
        "asr": ASR_ENGINE.status,
        "tts": TTS_ENGINE.status,
    })


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@sock.route("/ws")
def websocket(ws):
    state = SessionState()
    log_line("ws_connected", session=state.session_id, model=CONFIG.model_name)
    send_event(ws, state, {
        "type": "ready",
        "config": CONFIG.public,
        "asr": ASR_ENGINE.status,
        "tts": TTS_ENGINE.status,
    })

    while True:
        raw = ws.receive()
        if raw is None:
            state.ws_closed = True
            state.cancel_current()
            log_line("ws_disconnected", session=state.session_id)
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type", "")

        # Parlor-style: utterance is {audio, image} without a type field
        if "audio" in data and "type" not in data:
            start_turn(ws, state, audio_b64=data["audio"],
                       image_b64=data.get("image"))

        elif msg_type == "interrupt":
            if state.cancel_current():
                log_line("barge_in", session=state.session_id)
                send_event(ws, state, {"type": "audio_end"})

        elif msg_type == "text_input":
            start_turn(ws, state, text_input=data.get("text", ""),
                       image_b64=data.get("image"))

        # Legacy message types for backward compat
        elif msg_type == "utterance":
            start_turn(ws, state, audio_b64=data.get("wav_b64"),
                       image_b64=data.get("image"))
        elif msg_type == "speech_start":
            if state.cancel_current():
                log_line("barge_in", session=state.session_id)
                send_event(ws, state, {"type": "audio_end"})
        elif msg_type == "frame":
            pass  # no longer needed; image comes with utterance
        elif msg_type == "ping":
            send_event(ws, state, {"type": "pong"})


if __name__ == "__main__":
    app.run(host=CONFIG.host, port=CONFIG.port, debug=CONFIG.debug)
