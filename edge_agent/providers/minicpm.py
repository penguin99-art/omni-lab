"""MiniCPMProvider: System 1 (Perception) implementation via llama-server HTTP API.

Refactored from omni_web_demo.py. Talks to llama-server running MiniCPM-o 4.5.

llama-server endpoints used:
  POST /v1/stream/omni_init         - Initialize model
  POST /v1/stream/prefill           - Feed audio + image frame
  POST /v1/stream/decode            - Get model response (streaming SSE)
  POST /v1/stream/break             - Pause/interrupt inference
  POST /v1/stream/reset             - Reset context
  POST /v1/stream/update_session_config - Inject system prompt
  GET  /health                      - Health check
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from . import PerceptionResult

log = logging.getLogger(__name__)


class MiniCPMProvider:
    """
    System 1 implementation: real-time multimodal perception via llama-server.
    Handles audio + video frame streaming, TTS output collection, and context management.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:9060",
        model_dir: str = "./models/MiniCPM-o-4_5-gguf",
        ref_audio: str = "./official_ref_audio.wav",
        work_dir: str = "",
        max_chunks_before_reset: int = 300,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_dir = Path(model_dir)
        self.ref_audio = ref_audio

        base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ea_minicpm_", dir=str(base)))
        self.audio_dir = self.work_dir / "audio"
        self.frame_dir = self.work_dir / "frames"
        self.output_dir = self.work_dir / "output"
        for d in [self.audio_dir, self.frame_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._prefill_cnt = 1
        self._audio_idx = 0
        self._frame_idx = 0
        self._round_idx = 0
        self._initialized = False
        self._paused = False
        self._lock = threading.Lock()
        self._max_chunks = max_chunks_before_reset
        self._chunks_since_reset = 0
        self._conversation_history: list[dict] = []
        self._cur_bot_text = ""
        self._system_prompt = ""
        self._wav_cursor = 0
        self._client = httpx.AsyncClient(timeout=120)

    # -- PerceptionProvider interface ------------------------------------------

    async def start(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        await self._init_model()
        await self._update_session_config(system_prompt)
        log.info("MiniCPMProvider started. Work dir: %s", self.work_dir)

    async def pause(self) -> None:
        if self._paused:
            return
        try:
            await self._post("/v1/stream/break", {})
            self._paused = True
            log.debug("MiniCPM paused (break)")
        except Exception:
            log.warning("Failed to pause MiniCPM")

    async def resume(self) -> None:
        self._paused = False
        log.debug("MiniCPM resumed")

    async def feed(self, audio_b64: str, frame_b64: str = "") -> PerceptionResult:
        if self._paused:
            return PerceptionResult(text="", is_listening=True)

        with self._lock:
            self._chunks_since_reset += 1

        if self._chunks_since_reset >= self._max_chunks:
            await self._auto_reset("context approaching limit")
            return PerceptionResult(
                text="[context auto-refreshed]",
                is_listening=True,
            )

        await self._prefill(audio_b64, frame_b64)
        text, is_listen, is_end = await self._decode()

        if not is_listen and text and text.strip():
            self._cur_bot_text += text
        if is_listen and self._cur_bot_text.strip():
            self._conversation_history.append({"role": "assistant", "text": self._cur_bot_text.strip()})
            if len(self._conversation_history) > 16:
                self._conversation_history = self._conversation_history[-16:]
            self._cur_bot_text = ""

        audio_chunks = self._collect_tts_audio()

        return PerceptionResult(
            text=text,
            is_listening=is_listen,
            audio_chunks=audio_chunks,
        )

    async def inject_context(self, text: str) -> None:
        prompt = self._system_prompt + "\n\n" + text if self._system_prompt else text
        await self._update_session_config(prompt)

    async def reset(self) -> None:
        await self._auto_reset("manual reset")

    # -- Internal methods ------------------------------------------------------

    async def _post(self, path: str, body: dict, timeout: float = 30) -> dict:
        url = self.server_url + path
        resp = await self._client.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    async def _init_model(self, media_type: int = 2, duplex: bool = True) -> None:
        body = {
            "media_type": media_type,
            "use_tts": True,
            "duplex_mode": duplex,
            "model_dir": str(self.model_dir) + "/",
            "tts_bin_dir": str(self.model_dir / "token2wav-gguf"),
            "tts_gpu_layers": 99,
            "token2wav_device": "gpu:0",
            "output_dir": str(self.output_dir),
            "n_predict": 2048,
            "voice_audio": self.ref_audio,
        }
        resp = await self._post("/v1/stream/omni_init", body, timeout=120)
        if not resp.get("success"):
            raise RuntimeError("omni_init failed: {}".format(resp))
        with self._lock:
            self._initialized = True
            self._prefill_cnt = 1
            self._round_idx = 0

    async def _update_session_config(self, system_prompt: str) -> None:
        try:
            body = {
                "system_prompt": system_prompt,
                "voice_audio": self.ref_audio,
            }
            await self._post("/v1/stream/update_session_config", body)
        except Exception as e:
            log.warning("update_session_config failed (non-fatal): %s", e)

    async def _prefill(self, audio_b64: str, frame_b64: str = "") -> None:
        audio_bytes = base64.b64decode(audio_b64)
        with self._lock:
            cnt = self._prefill_cnt
            idx = self._audio_idx
            self._audio_idx += 1

        audio_path = str(self.audio_dir / "chunk_{:06d}.wav".format(idx))
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        img_path = ""
        if frame_b64:
            frame_bytes = base64.b64decode(frame_b64)
            with self._lock:
                fidx = self._frame_idx
                self._frame_idx += 1
            img_path = str(self.frame_dir / "frame_{:06d}.jpg".format(fidx))
            with open(img_path, "wb") as f:
                f.write(frame_bytes)

        body = {"audio_path_prefix": audio_path, "img_path_prefix": img_path, "cnt": cnt}
        await self._post("/v1/stream/prefill", body, timeout=30)
        with self._lock:
            self._prefill_cnt += 1

    async def _decode(self) -> tuple:
        with self._lock:
            rid = self._round_idx

        body = {"stream": True, "round_idx": rid}
        url = self.server_url + "/v1/stream/decode"

        full_text = ""
        is_listen = False
        is_end = False

        async with self._client.stream("POST", url, json=body, timeout=600) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                for line in chunk.split("\n"):
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
                            is_end = True
                    except (json.JSONDecodeError, KeyError):
                        pass

        with self._lock:
            self._round_idx += 1

        return full_text, is_listen, is_end

    def _collect_tts_audio(self) -> list[str]:
        """Collect new TTS WAV files and return as base64 strings."""
        results = []
        tts_dir = self.output_dir / "tts_wav"
        if not tts_dir.exists():
            return results
        while True:
            wav_path = tts_dir / "wav_{}.wav".format(self._wav_cursor)
            if not wav_path.exists():
                break
            try:
                if wav_path.stat().st_size < 100:
                    break
                data, sr = sf.read(str(wav_path), dtype="float32")
                pcm_b64 = base64.b64encode(data.astype(np.float32).tobytes()).decode("ascii")
                results.append(json.dumps({"pcm": pcm_b64, "sr": sr, "i": self._wav_cursor}))
                self._wav_cursor += 1
            except Exception:
                break
        return results

    async def _auto_reset(self, reason: str) -> None:
        log.info("Auto-reset: %s", reason)
        if self._cur_bot_text.strip():
            self._conversation_history.append({"role": "assistant", "text": self._cur_bot_text.strip()})
            self._cur_bot_text = ""

        try:
            await self._post("/v1/stream/reset", {}, timeout=10)
        except Exception as e:
            log.warning("Reset failed: %s", e)

        tts_dir = self.output_dir / "tts_wav"
        if tts_dir.exists():
            shutil.rmtree(tts_dir, ignore_errors=True)
        tts_dir.mkdir(parents=True, exist_ok=True)

        await self._init_model()
        self._wav_cursor = 0
        self._chunks_since_reset = 0

        recent = self._conversation_history[-8:]
        if recent:
            history_lines = "\n".join(
                "{}: {}".format("User" if t["role"] == "user" else "Assistant", t["text"][:80])
                for t in recent
            )
            prompt = "{}\n\n[Previous conversation]\n{}".format(self._system_prompt, history_lines)
            await self._update_session_config(prompt)

    async def health(self) -> bool:
        try:
            resp = await self._client.get(self.server_url + "/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
