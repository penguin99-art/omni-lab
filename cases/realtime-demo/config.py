from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class DemoConfig:
    host: str = os.environ.get("REALTIME_DEMO_HOST", "127.0.0.1")
    port: int = int(os.environ.get("REALTIME_DEMO_PORT", "8010"))
    debug: bool = os.environ.get("REALTIME_DEMO_DEBUG", "").lower() in {"1", "true", "yes"}

    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11436")
    model_name: str = os.environ.get("REALTIME_DEMO_MODEL", "gemma4:e2b")
    fallback_models: list[str] = field(
        default_factory=lambda: _parse_csv_env(
            "REALTIME_DEMO_FALLBACK_MODELS",
            "gemma4:e4b,gemma4:26b",
        )
    )
    system_prompt: str = os.environ.get(
        "REALTIME_DEMO_SYSTEM_PROMPT",
        (
            "You are a helpful local realtime multimodal assistant. "
            "Always reply in the same language the user speaks. "
            "If the user speaks Chinese, reply in Simplified Chinese (简体中文), never Traditional Chinese. "
            "Use the camera frame when relevant. Keep spoken responses concise (1-3 sentences)."
        ),
    )

    tts_backend: str = os.environ.get("REALTIME_DEMO_TTS", "kokoro").lower()
    kokoro_voice: str = os.environ.get("REALTIME_DEMO_KOKORO_VOICE", "af_heart")
    kokoro_speed: float = float(os.environ.get("REALTIME_DEMO_KOKORO_SPEED", "1.0"))

    asr_backend: str = os.environ.get("REALTIME_DEMO_ASR", "faster-whisper").lower()
    whisper_model: str = os.environ.get("REALTIME_DEMO_WHISPER_MODEL", "base")
    whisper_compute_type: str = os.environ.get("REALTIME_DEMO_WHISPER_COMPUTE", "int8")
    whisper_language: str = os.environ.get("REALTIME_DEMO_WHISPER_LANGUAGE", "zh").strip().lower()

    sample_rate: int = int(os.environ.get("REALTIME_DEMO_SAMPLE_RATE", "16000"))
    frame_interval_ms: int = int(os.environ.get("REALTIME_DEMO_FRAME_INTERVAL_MS", "900"))
    jpeg_quality: float = float(os.environ.get("REALTIME_DEMO_JPEG_QUALITY", "0.65"))
    sentence_flush_chars: int = int(os.environ.get("REALTIME_DEMO_SENTENCE_FLUSH_CHARS", "220"))

    # Proactive vision — periodically analyse camera for changes
    proactive_vision: bool = os.environ.get("REALTIME_DEMO_PROACTIVE_VISION", "").lower() in {"1", "true", "yes"}
    proactive_interval_s: int = int(os.environ.get("REALTIME_DEMO_PROACTIVE_INTERVAL", "5"))
    proactive_diff_threshold: float = float(os.environ.get("REALTIME_DEMO_PROACTIVE_DIFF_THRESHOLD", "0.06"))
    proactive_cooldown_s: int = int(os.environ.get("REALTIME_DEMO_PROACTIVE_COOLDOWN", "15"))
    proactive_prompt: str = os.environ.get(
        "REALTIME_DEMO_PROACTIVE_PROMPT",
        (
            "Look at this camera frame carefully. "
            "If you notice something interesting, notable, or changed "
            "(like a person appeared/left, an object was moved, a gesture, "
            "a new item on screen, etc.), briefly describe what you see "
            "in 1-2 concise sentences in the user's language. "
            "If the user speaks Chinese, reply in Simplified Chinese. "
            "If nothing notable or interesting is happening, respond with exactly: __SKIP__"
        ),
    )

    @property
    def public(self) -> dict:
        return {
            "model_name": self.model_name,
            "fallback_models": self.fallback_models,
            "sample_rate": self.sample_rate,
            "frame_interval_ms": self.frame_interval_ms,
            "jpeg_quality": self.jpeg_quality,
            "tts_backend": self.tts_backend,
            "asr_backend": self.asr_backend,
            "whisper_language": self.whisper_language,
            "proactive_vision": self.proactive_vision,
            "proactive_interval_s": self.proactive_interval_s,
            "proactive_diff_threshold": self.proactive_diff_threshold,
            "proactive_cooldown_s": self.proactive_cooldown_s,
        }


CONFIG = DemoConfig()
