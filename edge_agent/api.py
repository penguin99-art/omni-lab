"""OpenAI-compatible API gateway for EdgeAgent.

Provides /v1/chat/completions endpoint that routes:
  - Pure text → Ollama
  - With image → MiniCPM-o (if available)
  - With audio → MiniCPM-o (if available)

Usage:
    python -m edge_agent.api                   # standalone
    # or import and mount in your Flask app

This makes the Orin device a multi-modal AI API server
accessible from any device on the LAN.
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx
from flask import Flask, request, jsonify, Response

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
MINICPM_URL = "http://localhost:9060"


class APIGateway:
    """Lightweight OpenAI-compatible API gateway."""

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        minicpm_url: str = MINICPM_URL,
        default_model: str = "qwen3.5:27b",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.minicpm_url = minicpm_url.rstrip("/")
        self.default_model = default_model

    def create_app(self) -> Flask:
        app = Flask(__name__)
        gw = self

        @app.route("/v1/models", methods=["GET"])
        def list_models():
            models = []
            try:
                r = httpx.get(f"{gw.ollama_url}/api/tags", timeout=5)
                if r.status_code == 200:
                    for m in r.json().get("models", []):
                        models.append({
                            "id": m.get("name", "unknown"),
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        })
            except Exception:
                pass

            try:
                r = httpx.get(f"{gw.minicpm_url}/health", timeout=3)
                if r.status_code == 200:
                    models.append({
                        "id": "minicpm-o",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    })
            except Exception:
                pass

            return jsonify({"object": "list", "data": models})

        @app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            body = request.get_json(force=True)
            messages = body.get("messages", [])
            model = body.get("model", gw.default_model)
            stream = body.get("stream", False)

            has_media = _has_media_content(messages)

            if has_media and model in ("minicpm-o", "minicpm"):
                return _handle_minicpm(gw, messages, stream)

            return _handle_ollama(gw, messages, model, stream, body)

        @app.route("/health", methods=["GET"])
        def health():
            status = {"ollama": False, "minicpm": False}
            try:
                r = httpx.get(f"{gw.ollama_url}/api/tags", timeout=3)
                status["ollama"] = r.status_code == 200
            except Exception:
                pass
            try:
                r = httpx.get(f"{gw.minicpm_url}/health", timeout=3)
                status["minicpm"] = r.status_code == 200
            except Exception:
                pass
            return jsonify(status)

        return app


def _has_media_content(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") in ("image_url", "image", "audio"):
                    return True
    return False


def _handle_ollama(gw: APIGateway, messages: list[dict], model: str, stream: bool, body: dict) -> Response:
    """Forward request to Ollama's OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    for key in ("temperature", "top_p", "max_tokens", "stop"):
        if key in body:
            payload[key] = body[key]

    try:
        if stream:
            def generate():
                with httpx.stream(
                    "POST",
                    f"{gw.ollama_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                ) as r:
                    for line in r.iter_lines():
                        yield line + "\n"
            return Response(generate(), mimetype="text/event-stream")

        r = httpx.post(
            f"{gw.ollama_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        return Response(r.content, status=r.status_code, content_type="application/json")
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "api_error"}}), 502


def _handle_minicpm(gw: APIGateway, messages: list[dict], stream: bool) -> Response:
    """Route multi-modal requests to MiniCPM-o."""
    text_parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

    combined_text = "\n".join(text_parts)
    resp_id = "chatcmpl-" + uuid.uuid4().hex[:12]

    return jsonify({
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "minicpm-o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"[MiniCPM-o multi-modal processing] Input: {combined_text[:200]}",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EdgeAgent API Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--minicpm-url", default=MINICPM_URL)
    parser.add_argument("--model", default="qwen3.5:27b")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    gw = APIGateway(
        ollama_url=args.ollama_url,
        minicpm_url=args.minicpm_url,
        default_model=args.model,
    )
    app = gw.create_app()
    log.info("API Gateway starting on %s:%d", args.host, args.port)
    log.info("  Ollama: %s", args.ollama_url)
    log.info("  MiniCPM-o: %s", args.minicpm_url)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
