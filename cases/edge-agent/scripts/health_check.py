#!/usr/bin/env python3
"""24/7 health check script for EdgeAgent services.

Run via cron every hour:
    crontab -e
    0 * * * * cd /home/pineapi/gy && .venv/bin/python scripts/health_check.py

Logs results to stability.jsonl for long-term monitoring.
"""

import json
import time

import httpx

OLLAMA_URL = "http://localhost:11434"
MINICPM_URL = "http://localhost:9060"
AGENT_URL = "http://localhost:8080"
LOG_FILE = "stability.jsonl"


def check_ollama() -> dict:
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "qwen3.5:27b", "prompt": "ping", "stream": False},
            timeout=30,
        )
        return {
            "service": "ollama",
            "ok": r.status_code == 200,
            "status": r.status_code,
            "ms": round(r.elapsed.total_seconds() * 1000),
        }
    except Exception as e:
        return {"service": "ollama", "ok": False, "error": str(e)}


def check_minicpm() -> dict:
    try:
        r = httpx.get(f"{MINICPM_URL}/health", timeout=5)
        return {
            "service": "minicpm",
            "ok": r.status_code == 200,
            "status": r.status_code,
        }
    except Exception as e:
        return {"service": "minicpm", "ok": False, "error": str(e)}


def check_agent() -> dict:
    try:
        r = httpx.get(f"{AGENT_URL}/api/status", timeout=5)
        data = r.json()
        return {
            "service": "agent",
            "ok": r.status_code == 200,
            "system1": data.get("system1", False),
            "system2": data.get("system2", False),
        }
    except Exception as e:
        return {"service": "agent", "ok": False, "error": str(e)}


def check_gpu() -> dict:
    """Read GPU stats if available."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "service": "gpu",
                "ok": True,
                "temp_c": int(parts[0]),
                "mem_used_mb": int(parts[1]),
                "mem_total_mb": int(parts[2]),
                "util_pct": int(parts[3]),
            }
    except Exception:
        pass

    try:
        result = subprocess.run(["tegrastats", "--interval", "1000", "--count", "1"],
                                capture_output=True, text=True, timeout=5)
        return {"service": "gpu", "ok": True, "raw": result.stdout.strip()[:200]}
    except Exception:
        return {"service": "gpu", "ok": False, "error": "no gpu monitoring tool found"}


def main():
    checks = [check_ollama(), check_minicpm(), check_agent(), check_gpu()]
    entry = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
        "all_ok": all(c.get("ok", False) for c in checks),
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(json.dumps(entry, indent=2, ensure_ascii=False))

    if not entry["all_ok"]:
        failed = [c["service"] for c in checks if not c.get("ok")]
        print(f"\nWARNING: Failed services: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
