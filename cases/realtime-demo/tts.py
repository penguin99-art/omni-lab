from __future__ import annotations

import io
import shutil
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

from config import DemoConfig


class TTSUnavailableError(RuntimeError):
    pass


@dataclass
class SynthesisResult:
    audio_wav: bytes
    sample_rate: int
    backend: str


class BaseTTSBackend:
    sample_rate: int = 24000

    def generate_pcm(self, text: str) -> tuple[np.ndarray, int]:
        """Return (float32_samples, sample_rate). Primary method for streaming."""
        result = self.synthesize(text)
        buf = io.BytesIO(result.audio_wav)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            pcm_int16 = np.frombuffer(frames, dtype=np.int16)
            return pcm_int16.astype(np.float32) / 32768.0, wf.getframerate()

    def synthesize(self, text: str) -> SynthesisResult:
        raise NotImplementedError

    @property
    def status(self) -> dict:
        return {"available": True, "backend": "unknown", "reason": ""}


def _pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class KokoroTTSBackend(BaseTTSBackend):
    def __init__(self, voice: str, speed: float):
        try:
            from kokoro_onnx import Kokoro
        except Exception as exc:
            raise TTSUnavailableError(
                "Install kokoro-onnx to enable local Kokoro TTS: pip install kokoro-onnx onnxruntime"
            ) from exc

        self.voice = voice
        self.speed = speed
        self.backend_name = "kokoro"
        self.sample_rate = 24000

        model_path, voices_path = self._resolve_model_files()
        self._engine = Kokoro(model_path, voices_path)

    @staticmethod
    def _resolve_model_files() -> tuple[str, str]:
        """Find kokoro model files: local directory first, then HuggingFace Hub."""
        import os
        from pathlib import Path

        base = Path(__file__).parent
        local_model = base / "kokoro-v1.0.onnx"
        local_voices = base / "voices-v1.0.bin"
        if local_model.exists() and local_voices.exists():
            return str(local_model), str(local_voices)

        cache = Path(os.environ.get("KOKORO_CACHE", Path.home() / ".cache" / "kokoro"))
        cached_model = cache / "kokoro-v1.0.onnx"
        cached_voices = cache / "voices-v1.0.bin"
        if cached_model.exists() and cached_voices.exists():
            return str(cached_model), str(cached_voices)

        raise TTSUnavailableError(
            f"Kokoro model files not found. Download kokoro-v1.0.onnx and voices-v1.0.bin "
            f"to {base} or {cache}"
        )

    def generate_pcm(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._engine.create(text, voice=self.voice, speed=self.speed)
        return np.asarray(samples, dtype=np.float32), sr

    def synthesize(self, text: str) -> SynthesisResult:
        samples, sample_rate = self._engine.create(text, voice=self.voice, speed=self.speed)
        return SynthesisResult(
            audio_wav=_pcm_to_wav_bytes(np.asarray(samples), sample_rate),
            sample_rate=sample_rate,
            backend=self.backend_name,
        )

    @property
    def status(self) -> dict:
        return {"available": True, "backend": self.backend_name, "reason": "", "voice": self.voice}


class EdgeTTSBackend(BaseTTSBackend):
    """Microsoft Edge TTS — supports Chinese, English, and many other languages."""

    def __init__(self):
        try:
            import edge_tts as _  # noqa: F401
        except ImportError as exc:
            raise TTSUnavailableError("pip install edge-tts") from exc
        self.sample_rate = 24000

    def generate_pcm(self, text: str) -> tuple[np.ndarray, int]:
        import asyncio
        import edge_tts

        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        voice = "zh-CN-XiaoxiaoNeural" if has_cjk else "en-US-AriaNeural"

        async def _synth():
            comm = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    mp3_bytes = pool.submit(lambda: asyncio.run(_synth())).result(timeout=15)
            else:
                mp3_bytes = loop.run_until_complete(_synth())
        except RuntimeError:
            mp3_bytes = asyncio.run(_synth())

        if not mp3_bytes:
            raise TTSUnavailableError("Edge TTS returned empty audio")

        import subprocess
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1"],
            input=mp3_bytes, capture_output=True,
        )
        if proc.returncode != 0:
            raise TTSUnavailableError(f"ffmpeg decode failed: {proc.stderr[:200]}")

        pcm_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
        return pcm_i16.astype(np.float32) / 32768.0, 24000

    def synthesize(self, text: str) -> SynthesisResult:
        pcm, sr = self.generate_pcm(text)
        return SynthesisResult(audio_wav=_pcm_to_wav_bytes(pcm, sr), sample_rate=sr, backend="edge-tts")

    @property
    def status(self) -> dict:
        return {"available": True, "backend": "edge-tts", "reason": ""}


class FfmpegFliteBackend(BaseTTSBackend):
    def __init__(self):
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise TTSUnavailableError("ffmpeg is required for flite fallback TTS")
        self.sample_rate = 16000

    def synthesize(self, text: str) -> SynthesisResult:
        escaped = (
            text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace("\n", " ")
        )
        proc = subprocess.run(
            [
                self.ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"flite=text='{escaped}':voice=slt",
                "-f", "wav", "-",
            ],
            check=True,
            capture_output=True,
        )
        return SynthesisResult(audio_wav=proc.stdout, sample_rate=16000, backend="ffmpeg-flite")

    @property
    def status(self) -> dict:
        return {"available": True, "backend": "ffmpeg-flite", "reason": ""}


class UnavailableTTS(BaseTTSBackend):
    def __init__(self, reason: str):
        self.reason = reason

    def generate_pcm(self, text: str) -> tuple[np.ndarray, int]:
        raise TTSUnavailableError(self.reason)

    def synthesize(self, text: str) -> SynthesisResult:
        raise TTSUnavailableError(self.reason)

    @property
    def status(self) -> dict:
        return {"available": False, "backend": "unavailable", "reason": self.reason}


def create_tts_backend(config: DemoConfig) -> BaseTTSBackend:
    backend = config.tts_backend
    if backend == "kokoro":
        try:
            return KokoroTTSBackend(config.kokoro_voice, config.kokoro_speed)
        except Exception as exc:
            print(f"[tts] Kokoro unavailable: {exc}")
    elif backend == "edge":
        try:
            return EdgeTTSBackend()
        except TTSUnavailableError as exc:
            print(f"[tts] Edge TTS unavailable: {exc}")
    elif backend in {"system", "flite"}:
        try:
            return FfmpegFliteBackend()
        except TTSUnavailableError as exc:
            return UnavailableTTS(str(exc))

    # Fallback chain: edge-tts (Chinese support) → flite (English only)
    for cls in [EdgeTTSBackend, FfmpegFliteBackend]:
        try:
            be = cls()
            print(f"[tts] Using fallback: {be.status['backend']}")
            return be
        except TTSUnavailableError:
            continue
    return UnavailableTTS("No TTS backend available")
