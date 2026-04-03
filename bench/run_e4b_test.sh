#!/usr/bin/env bash
set -euo pipefail
LOG="/home/pineapi/gy/bench/r2_e4b.log"
exec > >(tee -a "$LOG") 2>&1

export OLLAMA_HOST=http://localhost:11436
OLLAMA_CMD="$HOME/bin/ollama-020/bin/ollama"

echo "=== $(date) Step 1: Check Ollama ==="
curl -s http://localhost:11436/api/version || true
echo ""

if ! curl -sf http://localhost:11436/api/version >/dev/null 2>&1; then
  echo "=== Starting Ollama 0.20 on 11436 ==="
  LD_LIBRARY_PATH=$HOME/bin/ollama-020/lib/ollama/cuda_v13:/usr/local/cuda/lib64 \
    OLLAMA_HOST=0.0.0.0:11436 \
    OLLAMA_MODELS=/home/pineapi/liuminglu/models/models \
    OLLAMA_FLASH_ATTENTION=true \
    $HOME/bin/ollama-020/bin/ollama serve &
  sleep 5
  curl -s http://localhost:11436/api/version || true
  echo ""
fi

echo "=== Step 2: List gemma models ==="
$OLLAMA_CMD list 2>&1 | grep gemma || true
echo ""

cd /home/pineapi/gy

echo "=== Step 3: gemma4:e4b quick ==="
python3 bench/bench.py --models "gemma4:e4b" --suite quick --tag r2_gemma4_e4b_quick
echo ""

echo "=== Step 4: gemma4:e4b standard ==="
python3 bench/bench.py --models "gemma4:e4b" --suite standard --tag r2_gemma4_e4b_standard
echo ""

echo "=== Step 5: gemma4:e4b toolcall ==="
python3 bench/bench.py --models "gemma4:e4b" --suite toolcall --tag r2_gemma4_e4b_toolcall
echo ""

echo "=== Step 6: Pull gemma4:26b ==="
$OLLAMA_CMD pull gemma4:26b 2>&1 | tail -3
echo ""

echo "=== Step 7: gemma4:26b quick ==="
python3 bench/bench.py --models "gemma4:26b" --suite quick --tag r2_gemma4_26b_quick
echo ""

echo "=== Step 8: gemma4:26b standard ==="
python3 bench/bench.py --models "gemma4:26b" --suite standard --tag r2_gemma4_26b_standard
echo ""

echo "=== Step 9: gemma4:26b toolcall ==="
python3 bench/bench.py --models "gemma4:26b" --suite toolcall --tag r2_gemma4_26b_toolcall
echo ""

echo "=== Step 10: Pull gemma4:31b ==="
$OLLAMA_CMD pull gemma4:31b 2>&1 | tail -3
echo ""

echo "=== Step 11: gemma4:31b quick ==="
python3 bench/bench.py --models "gemma4:31b" --suite quick --tag r2_gemma4_31b_quick
echo ""

echo "=== Step 12: gemma4:31b standard ==="
python3 bench/bench.py --models "gemma4:31b" --suite standard --tag r2_gemma4_31b_standard
echo ""

echo "=== Step 13: gemma4:31b toolcall ==="
python3 bench/bench.py --models "gemma4:31b" --suite toolcall --tag r2_gemma4_31b_toolcall
echo ""

echo "=== Step 14: qwen3.5:35b standard (control) ==="
python3 bench/bench.py --models "qwen3.5:35b" --suite standard --tag r2_qwen35_35b_standard
echo ""

echo "=== Step 15: qwen3.5:35b toolcall (control) ==="
python3 bench/bench.py --models "qwen3.5:35b" --suite toolcall --tag r2_qwen35_35b_toolcall
echo ""

echo "=== ALL DONE $(date) ==="
