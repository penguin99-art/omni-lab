#!/bin/bash
# Start vLLM with Qwen3.5-35B-A3B-FP8 + prefix caching for TTFT comparison
MODEL_DIR="/home/pineapi/liuminglu/models/qwen35-35b-fp8"
CONTAINER_NAME="vllm-qwen35-fp8"
PORT=18080

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    exit 1
fi

docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "Starting vLLM with Qwen3.5-35B-A3B-FP8 on port $PORT..."
echo "  - Prefix caching: ENABLED"
echo "  - KV cache dtype: fp8"

docker run -d \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus all \
    --shm-size 16g \
    -p ${PORT}:8000 \
    -v ${MODEL_DIR}:/model \
    vllm/vllm-openai:cu130-nightly \
    --model /model \
    --served-model-name qwen3.5-35b-fp8 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --chat-template-content-format string \
    --dtype auto \
    --trust-remote-code

echo ""
echo "Container started. Checking logs..."
sleep 5
docker logs $CONTAINER_NAME 2>&1 | tail -20

echo ""
echo "To monitor: docker logs -f $CONTAINER_NAME"
echo "API endpoint: http://localhost:${PORT}/v1"
