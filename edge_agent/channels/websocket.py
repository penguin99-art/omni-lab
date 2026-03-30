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
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ..events import (
    EventBus,
    VisualScene,
    ThinkingStarted,
    ReasoningDone,
)

if TYPE_CHECKING:
    from .. import EdgeAgent

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class WebSocketChannel:
    """Full-duplex WebSocket channel with voice + camera + agent support."""

    name = "websocket"

    def __init__(self, agent: "EdgeAgent", host: str = "0.0.0.0", port: int = 8080) -> None:
        self._agent = agent
        self._host = host
        self._port = port
        self._bus: Optional[EventBus] = None
        self._app = None
        self._active_senders: list[Callable] = []

    async def start(self, bus: EventBus) -> None:
        self._bus = bus
        self._setup_flask()
        thread = threading.Thread(target=self._run_flask, daemon=True)
        thread.start()
        log.info("WebSocket channel started on %s:%d", self._host, self._port)

    async def send(self, text: str, media: Optional[bytes] = None) -> None:
        """Push a message to all active WebSocket clients."""
        msg = json.dumps({"type": "agent_result", "text": text}, ensure_ascii=False)
        dead: list[Callable] = []
        for sender in list(self._active_senders):
            if not sender(msg):
                dead.append(sender)
        for s in dead:
            try:
                self._active_senders.remove(s)
            except ValueError:
                pass

    def _setup_flask(self) -> None:
        from flask import Flask, send_from_directory, jsonify
        from flask_sock import Sock

        app = Flask(__name__)
        sock = Sock(app)
        self._app = app
        agent = self._agent

        @app.route("/")
        def index():
            if WEB_DIR.exists() and (WEB_DIR / "index.html").exists():
                return send_from_directory(str(WEB_DIR), "index.html")
            return "<h1>Edge Agent</h1><p>Web UI not found. Place files in edge_agent/web/</p>"

        @app.route("/<path:filename>")
        def static_files(filename):
            return send_from_directory(str(WEB_DIR), filename)

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

        self._active_senders.append(safe_send)

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
                        if frame_b64 and result.text and result.text.strip():
                            try:
                                loop.run_until_complete(
                                    self._bus.emit(VisualScene(description=result.text))
                                )
                            except Exception:
                                log.debug("VisualScene emit failed")
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
            if safe_send in self._active_senders:
                self._active_senders.remove(safe_send)
            agent.bus.off(ThinkingStarted, on_thinking)
            agent.bus.off(ReasoningDone, on_reasoning_done)
            if perception:
                try:
                    loop.run_until_complete(perception.pause())
                except Exception:
                    pass
            loop.close()
            log.info("WebSocket session ended")
