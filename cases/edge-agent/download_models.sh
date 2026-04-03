#!/bin/bash
set -e

BASE_URL="https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/models/MiniCPM-o-4_5-gguf}"

mkdir -p "$OUT_DIR"/{audio,tts,token2wav-gguf,vision}

download() {
    local rel="$1"
    local out="$OUT_DIR/$rel"
    if [ -f "$out" ]; then
        local expected_min="$2"
        local actual=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ "$actual" -ge "$expected_min" ]; then
            echo "[SKIP] $rel ($(numfmt --to=iec $actual))"
            return 0
        fi
        rm -f "$out"
    fi
    echo "[DOWN] $rel ..."
    curl -L -C - --retry 10 --retry-delay 5 -o "$out" "$BASE_URL/$rel" 2>&1
    local size=$(stat -c%s "$out" 2>/dev/null || echo 0)
    echo "[DONE] $rel ($(numfmt --to=iec $size))"
}

echo "=== Downloading MiniCPM-o 4.5 GGUF models ==="
echo "Destination: $OUT_DIR"
echo ""

# Small files first (fast)
download "tts/MiniCPM-o-4_5-projector-F16.gguf"      14000000
download "token2wav-gguf/flow_extra.gguf"              13000000
download "token2wav-gguf/prompt_cache.gguf"            67000000
download "token2wav-gguf/hifigan2.gguf"                79000000
download "token2wav-gguf/encoder.gguf"                144000000
download "token2wav-gguf/flow_matching.gguf"          437000000

# Medium files
download "audio/MiniCPM-o-4_5-audio-F16.gguf"        629000000

# Large files
download "tts/MiniCPM-o-4_5-tts-F16.gguf"           1073000000
download "vision/MiniCPM-o-4_5-vision-F16.gguf"      2500000000
download "MiniCPM-o-4_5-Q4_K_M.gguf"                 4600000000

echo ""
echo "=== All downloads complete! ==="
find "$OUT_DIR" -name "*.gguf" -exec ls -lh {} \;
