#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models/MiniCPM-o-4_5-gguf"
Q8_MODEL="$MODEL_DIR/MiniCPM-o-4_5-Q8_0.gguf"
VISION_PROJ="$MODEL_DIR/vision/MiniCPM-o-4_5-vision-F16.gguf"
EVAL_SERVER="$SCRIPT_DIR/llama.cpp-eval/build/bin/llama-server"
EVAL_PORT=9061

echo "=== MiniCPM-o 4.5 Q8_0 Evaluation Pipeline ==="

# 1. Check prerequisites
if [ ! -f "$Q8_MODEL" ]; then
    echo "ERROR: Q8_0 model not found at $Q8_MODEL"
    echo "Download it first:"
    echo "  curl -L -C - --retry 10 -o $Q8_MODEL \\"
    echo "    https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/resolve/main/MiniCPM-o-4_5-Q8_0.gguf"
    exit 1
fi

if [ ! -f "$VISION_PROJ" ]; then
    echo "ERROR: Vision projector not found at $VISION_PROJ"
    exit 1
fi

if [ ! -f "$EVAL_SERVER" ]; then
    echo "ERROR: Eval llama-server not found at $EVAL_SERVER"
    echo "Build it first: cd llama.cpp-eval && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc) --target llama-server"
    exit 1
fi

echo "[1/4] Model size: $(du -h "$Q8_MODEL" | cut -f1)"

# 2. Kill any existing eval server on the same port
if lsof -ti:$EVAL_PORT > /dev/null 2>&1; then
    echo "[2/4] Killing existing server on port $EVAL_PORT..."
    kill $(lsof -ti:$EVAL_PORT) 2>/dev/null || true
    sleep 2
fi

# 3. Start llama-server with Q8_0
echo "[2/4] Starting llama-server with Q8_0 on port $EVAL_PORT..."
"$EVAL_SERVER" \
    --host 127.0.0.1 --port $EVAL_PORT \
    --model "$Q8_MODEL" \
    --mmproj "$VISION_PROJ" \
    -ngl 99 --ctx-size 4096 --temp 0.1 \
    > "$SCRIPT_DIR/eval_server_q8.log" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
echo "  Waiting for server health..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:$EVAL_PORT/health | grep -q "ok"; then
        echo "  Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died. Check eval_server_q8.log"
        exit 1
    fi
    sleep 1
done

# 4. Run evaluation
echo "[3/4] Running MMStar evaluation (300 samples, seed=42)..."
rm -f "$SCRIPT_DIR/eval_results/mmstar_results_Q8_0.jsonl"
rm -f "$SCRIPT_DIR/eval_results/mmstar_summary_Q8_0.json"
rm -f "$SCRIPT_DIR/eval_results/progress_Q8_0.log"

cd "$SCRIPT_DIR"
python3 eval_vision.py \
    --llama-port $EVAL_PORT \
    --samples 300 \
    --seed 42 \
    --tag Q8_0 \
    2>&1 | tee eval_vision_q8.log

echo "[4/4] Evaluation complete. Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Results ==="
cat eval_results/mmstar_summary_Q8_0.json | python3 -m json.tool
echo ""
echo "Comparison report: eval_results/REPORT.md"
echo "Done!"
