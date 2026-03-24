#!/usr/bin/env python3
"""
MiniCPM-o 4.5 全双工语音+摄像头对话 Demo

架构:
  llama-server (HTTP API, port 9060)
    ↕
  本脚本 (Python client)
    - 麦克风录音 → 1s WAV chunks
    - 摄像头截帧 → JPEG
    - 调用 /v1/stream/prefill + /v1/stream/decode
    - 监听 TTS 输出 WAV 并播放

使用方式:
  1. 先启动 llama-server (本脚本可自动启动，或手动):
     ./llama.cpp-omni/build/bin/llama-server \\
       --host 0.0.0.0 --port 9060 \\
       --model models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \\
       -ngl 99 --ctx-size 4096

  2. 运行本脚本:
     python3 omni_duplex_demo.py

  按 Ctrl+C 退出。
"""

import argparse
import json
import os
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf


# ─── 配置 ──────────────────────────────────────────────────────────

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9060
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_CHUNK_SEC = 1.0
FRAME_INTERVAL_SEC = 1.0
TTS_SAMPLE_RATE = 24000

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "MiniCPM-o-4_5-gguf"
LLAMA_SERVER = BASE_DIR / "llama.cpp-omni" / "build" / "bin" / "llama-server"
LLM_MODEL = MODEL_DIR / "MiniCPM-o-4_5-Q4_K_M.gguf"

WORK_DIR = Path(tempfile.mkdtemp(prefix="omni_duplex_"))
OUTPUT_DIR = WORK_DIR / "output"


# ─── 全局状态 ──────────────────────────────────────────────────────

running = True
server_proc = None
round_idx = 0


def signal_handler(sig, frame):
    global running
    print("\n[INFO] 正在退出...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── 工具函数 ──────────────────────────────────────────────────────

def api_url(path: str) -> str:
    return f"http://{SERVER_HOST}:{SERVER_PORT}{path}"


def save_wav(filepath: str, audio_data: np.ndarray, sr: int = SAMPLE_RATE):
    sf.write(filepath, audio_data, sr, subtype="PCM_16")


def wait_server_ready(timeout: int = 120):
    """等待 llama-server 启动就绪"""
    print("[INFO] 等待 llama-server 启动...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(api_url("/health"), timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    print("[INFO] llama-server 就绪!")
                    return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(2)
    print("[ERROR] llama-server 启动超时")
    return False


def start_server(args):
    """启动 llama-server"""
    global server_proc
    cmd = [
        str(LLAMA_SERVER),
        "--host", "0.0.0.0",
        "--port", str(SERVER_PORT),
        "--model", str(LLM_MODEL),
        "-ngl", "99",
        "--ctx-size", str(args.ctx_size),
        "--repeat-penalty", "1.05",
        "--temp", "0.7",
    ]
    print(f"[INFO] 启动 llama-server: {' '.join(cmd)}")
    server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    threading.Thread(target=_drain_server_log, daemon=True).start()
    return wait_server_ready()


def _drain_server_log():
    """读取 server 日志并打印关键行"""
    if server_proc is None:
        return
    for line in iter(server_proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if any(k in text.lower() for k in ["error", "listen", "loaded", "omni", "ready"]):
            print(f"  [server] {text}")


def omni_init(duplex: bool = True, media_type: int = 2):
    """初始化 omni 模块"""
    ref_audio = str(BASE_DIR / "llama.cpp-omni" / "tools" / "omni" / "assets" / "default_ref_audio" / "default_ref_audio.wav")
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
    print("[INFO] 初始化 omni 模块...")
    r = requests.post(api_url("/v1/stream/omni_init"), json=body, timeout=120)
    r.raise_for_status()
    resp = r.json()
    if not resp.get("success"):
        raise RuntimeError(f"omni_init failed: {resp}")
    print(f"[INFO] omni 模块初始化成功: media_type={resp.get('media_type')}, tts={resp.get('use_tts')}")
    return resp


def stream_prefill(audio_path: str, img_path: str, cnt: int):
    """发送音频+图像给模型进行预填充"""
    body = {
        "audio_path_prefix": audio_path,
        "img_path_prefix": img_path,
        "cnt": cnt,
    }
    r = requests.post(api_url("/v1/stream/prefill"), json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def stream_decode_sse(round_id: int) -> list[str]:
    """
    调用 stream_decode，通过 SSE 接收文本流。
    返回模型生成的文本片段列表。
    """
    body = {
        "stream": True,
        "round_idx": round_id,
    }
    texts = []
    try:
        r = requests.post(
            api_url("/v1/stream/decode"),
            json=body,
            stream=True,
            timeout=60,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            content = ev.get("content", "")
            if content:
                texts.append(content)
                print(content, end="", flush=True)
            if ev.get("stop") or ev.get("end_of_turn"):
                break
    except Exception as e:
        print(f"\n[WARN] decode SSE error: {e}")
    return texts


def stream_break():
    """打断当前生成"""
    try:
        r = requests.post(api_url("/v1/stream/break"), json={}, timeout=5)
        r.raise_for_status()
    except Exception:
        pass


def stream_reset():
    """重置会话"""
    try:
        r = requests.post(api_url("/v1/stream/reset"), json={}, timeout=10)
        r.raise_for_status()
    except Exception:
        pass


# ─── 音频录制 ──────────────────────────────────────────────────────

class AudioRecorder:
    """持续录音，按 chunk 写入 WAV 文件"""

    def __init__(self, output_dir: Path, chunk_sec: float = AUDIO_CHUNK_SEC):
        self.output_dir = output_dir
        self.chunk_sec = chunk_sec
        self.chunk_samples = int(SAMPLE_RATE * chunk_sec)
        self._buffer = queue.Queue()
        self._stream = None
        self._chunk_idx = 0

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()
        print(f"[MIC] 录音已启动 (sr={SAMPLE_RATE}, chunk={self.chunk_sec}s)")

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MIC WARN] {status}")
        self._buffer.put(indata.copy())

    def record_chunk(self) -> str | None:
        """录制一个 chunk 的音频，保存为 WAV，返回文件路径"""
        frames = []
        total = 0
        deadline = time.time() + self.chunk_sec + 0.1
        while total < self.chunk_samples and time.time() < deadline:
            try:
                data = self._buffer.get(timeout=0.1)
                frames.append(data)
                total += len(data)
            except queue.Empty:
                continue

        if not frames:
            return None

        audio = np.concatenate(frames, axis=0)[:self.chunk_samples]
        if len(audio) < self.chunk_samples:
            pad = np.zeros((self.chunk_samples - len(audio), CHANNELS), dtype="float32")
            audio = np.concatenate([audio, pad], axis=0)

        filepath = str(self.output_dir / f"mic_{self._chunk_idx:04d}.wav")
        save_wav(filepath, audio.flatten(), SAMPLE_RATE)
        self._chunk_idx += 1
        return filepath

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ─── 摄像头捕获 ──────────────────────────────────────────────────────

class CameraCapture:
    """从摄像头捕获帧"""

    def __init__(self, output_dir: Path, device_id: int = 0):
        self.output_dir = output_dir
        self.device_id = device_id
        self._cap = None
        self._frame_idx = 0
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            print(f"[WARN] 无法打开摄像头 {self.device_id}，将在无视觉模式下运行")
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[CAM] 摄像头已启动 (device={self.device_id})")
        return True

    def _capture_loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(0.03)

    def capture_frame(self) -> str | None:
        """捕获当前帧，保存为 JPEG，返回文件路径"""
        with self._lock:
            frame = self._latest_frame

        if frame is None:
            return None

        filepath = str(self.output_dir / f"frame_{self._frame_idx:04d}.jpg")
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._frame_idx += 1
        return filepath

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None


# ─── TTS 音频播放 ──────────────────────────────────────────────────

class TTSPlayer:
    """监听输出目录中的 TTS WAV 文件并播放"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._running = False
        self._thread = None
        self._played = set()
        self.is_speaking = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _play_loop(self):
        while self._running:
            self._scan_and_play()
            time.sleep(0.1)

    def _scan_and_play(self):
        for round_dir in sorted(self.output_dir.glob("round_*")):
            wav_dir = round_dir / "tts_wav"
            if not wav_dir.exists():
                continue
            wav_files = sorted(wav_dir.glob("wav_*.wav"))
            for wf in wav_files:
                key = str(wf)
                if key in self._played:
                    continue
                if wf.stat().st_size < 100:
                    continue
                self._played.add(key)
                self._play_wav(wf)

    def _play_wav(self, wav_path: Path):
        try:
            self.is_speaking = True
            data, sr = sf.read(str(wav_path), dtype="float32")
            if len(data) == 0:
                return
            print(f"  [TTS] 播放 {wav_path.name} ({len(data)/sr:.2f}s)")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"  [TTS ERROR] {wav_path.name}: {e}")
        finally:
            self.is_speaking = False

    def wait_done(self, timeout: float = 30.0):
        """等待所有已知 WAV 播放完成"""
        t0 = time.time()
        while self.is_speaking and (time.time() - t0) < timeout:
            time.sleep(0.2)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)


# ─── 主循环 ──────────────────────────────────────────────────────

def run_duplex_loop(args):
    global running, round_idx

    mic_dir = WORK_DIR / "mic"
    cam_dir = WORK_DIR / "cam"
    mic_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recorder = AudioRecorder(mic_dir, chunk_sec=args.chunk_sec)
    camera = CameraCapture(cam_dir, device_id=args.camera)
    player = TTSPlayer(OUTPUT_DIR)

    has_camera = camera.start()
    recorder.start()
    player.start()

    media_type = 2 if has_camera else 1
    omni_init(duplex=True, media_type=media_type)

    cnt = 1

    print("\n" + "=" * 60)
    print("  全双工语音" + ("+摄像头" if has_camera else "") + "对话已就绪")
    print("  请对着麦克风说话，模型会实时回复")
    print("  按 Ctrl+C 退出")
    print("=" * 60 + "\n")

    try:
        while running:
            # 录制一段音频
            audio_path = recorder.record_chunk()
            if audio_path is None:
                continue

            # 捕获一帧图像
            img_path = ""
            if has_camera:
                frame_path = camera.capture_frame()
                if frame_path:
                    img_path = frame_path

            # 发送 prefill
            try:
                stream_prefill(audio_path, img_path, cnt)
                cnt += 1
            except Exception as e:
                print(f"[WARN] prefill 失败: {e}")
                continue

            # 每次 prefill 后触发 decode 获取回复
            if cnt % args.decode_interval == 0:
                print(f"\n[轮次 {round_idx}] 模型回复: ", end="")
                texts = stream_decode_sse(round_idx)
                if texts:
                    print()
                    round_idx += 1
                    player.wait_done(timeout=15)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] 清理资源...")
        recorder.stop()
        camera.stop()
        player.stop()


def run_simplex_loop(args):
    """单工模式: 录音 → 发送 → 等回复 → 播放 → 再录音"""
    global running, round_idx

    mic_dir = WORK_DIR / "mic"
    cam_dir = WORK_DIR / "cam"
    mic_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recorder = AudioRecorder(mic_dir, chunk_sec=args.chunk_sec)
    camera = CameraCapture(cam_dir, device_id=args.camera)
    player = TTSPlayer(OUTPUT_DIR)

    has_camera = camera.start()
    recorder.start()
    player.start()

    media_type = 2 if has_camera else 1
    omni_init(duplex=False, media_type=media_type)

    print("\n" + "=" * 60)
    print("  单工语音" + ("+摄像头" if has_camera else "") + "对话已就绪")
    print("  说完一段话后等待模型回复，回复完再说下一段")
    print("  按 Ctrl+C 退出")
    print("=" * 60 + "\n")

    cnt = 1
    try:
        while running:
            # 录制多个 chunk 的音频（积攒用户输入）
            print(f"\n[轮次 {round_idx}] 正在录音 ({args.listen_sec}s)...", flush=True)
            chunk_count = max(1, int(args.listen_sec / args.chunk_sec))
            last_audio = None
            last_img = ""
            for _ in range(chunk_count):
                if not running:
                    break
                audio_path = recorder.record_chunk()
                if audio_path:
                    last_audio = audio_path
                if has_camera:
                    frame_path = camera.capture_frame()
                    if frame_path:
                        last_img = frame_path

                # 每个 chunk 都 prefill
                if last_audio:
                    try:
                        stream_prefill(last_audio, last_img, cnt)
                        cnt += 1
                    except Exception as e:
                        print(f"[WARN] prefill: {e}")

            if not running:
                break

            # 触发 decode
            print(f"[轮次 {round_idx}] 模型回复: ", end="", flush=True)
            texts = stream_decode_sse(round_idx)
            print()
            round_idx += 1

            player.wait_done(timeout=30)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] 清理资源...")
        recorder.stop()
        camera.stop()
        player.stop()


# ─── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MiniCPM-o 4.5 全双工语音+摄像头对话 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动启动 server + 双工模式
  python3 omni_duplex_demo.py

  # 使用已运行的 server
  python3 omni_duplex_demo.py --no-server --port 9060

  # 单工模式（录音完等回复）
  python3 omni_duplex_demo.py --simplex

  # 不使用摄像头（纯语音）
  python3 omni_duplex_demo.py --no-camera
        """,
    )
    parser.add_argument("--no-server", action="store_true", help="不自动启动 llama-server")
    parser.add_argument("--host", default=SERVER_HOST, help="server 地址")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="server 端口")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备 ID")
    parser.add_argument("--no-camera", action="store_true", help="不使用摄像头（纯语音模式）")
    parser.add_argument("--simplex", action="store_true", help="单工模式（录完再回复）")
    parser.add_argument("--chunk-sec", type=float, default=1.0, help="音频 chunk 时长(秒)")
    parser.add_argument("--listen-sec", type=float, default=3.0, help="单工模式每轮录音时长(秒)")
    parser.add_argument("--decode-interval", type=int, default=2, help="双工模式每几个 chunk 触发一次 decode")
    parser.add_argument("--ctx-size", type=int, default=4096, help="LLM 上下文窗口大小")
    args = parser.parse_args()

    global SERVER_HOST, SERVER_PORT
    SERVER_HOST = args.host
    SERVER_PORT = args.port

    print("=" * 60)
    print("  MiniCPM-o 4.5 全双工语音+摄像头对话 Demo")
    print(f"  工作目录: {WORK_DIR}")
    print(f"  Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)

    if args.no_camera:
        CameraCapture.start = lambda self: False

    # 启动 server
    if not args.no_server:
        if not start_server(args):
            print("[ERROR] 无法启动 llama-server，退出")
            sys.exit(1)
    else:
        if not wait_server_ready(timeout=10):
            print("[ERROR] 无法连接到 llama-server，请确认服务已启动")
            sys.exit(1)

    try:
        if args.simplex:
            run_simplex_loop(args)
        else:
            run_duplex_loop(args)
    finally:
        if server_proc:
            print("[INFO] 停止 llama-server...")
            server_proc.terminate()
            server_proc.wait(timeout=10)
        print(f"[INFO] 临时文件在: {WORK_DIR}")
        print("[INFO] 再见!")


if __name__ == "__main__":
    main()
