#!/usr/bin/env bash
set -euo pipefail
LOG="/home/pineapi/gy/bench/r2_remaining.log"
exec > >(tee -a "$LOG") 2>&1

export OLLAMA_HOST=http://localhost:11436
cd /home/pineapi/gy

echo "=== $(date) gemma4:31b quick ==="
python3 bench/bench.py --models "gemma4:31b" --suite quick --tag r2_gemma4_31b_quick

echo "=== $(date) gemma4:31b standard ==="
python3 bench/bench.py --models "gemma4:31b" --suite standard --tag r2_gemma4_31b_standard

echo "=== $(date) gemma4:31b toolcall ==="
python3 bench/bench.py --models "gemma4:31b" --suite toolcall --tag r2_gemma4_31b_toolcall

echo "=== $(date) qwen3.5:35b standard (control) ==="
python3 bench/bench.py --models "qwen3.5:35b" --suite standard --tag r2_qwen35_35b_standard

echo "=== $(date) qwen3.5:35b toolcall (control) ==="
python3 bench/bench.py --models "qwen3.5:35b" --suite toolcall --tag r2_qwen35_35b_toolcall

echo "=== ALL DONE $(date) ==="
