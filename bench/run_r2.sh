#!/bin/bash
# Round 2: Gemma 4 全系列 + Qwen 3.5:35b 对比测试
# 使用 Ollama 0.20.0 @ port 11436

export OLLAMA_HOST=http://localhost:11436
OLLAMA_CMD="$HOME/bin/ollama-020/bin/ollama"
BENCH="python3 /home/pineapi/gy/bench/bench.py"
cd /home/pineapi/gy

MODELS_Q4=(
    "gemma4:e4b"
    "gemma4:26b"
    "gemma4:31b"
)

MODELS_Q8=(
    "gemma4:e2b-q8"
    "gemma4:e4b-q8"
    "gemma4:26b-q8"
    "gemma4:31b-q8"
)

MODELS_BF16=(
    "gemma4:e2b-bf16"
    "gemma4:e4b-bf16"
    "gemma4:31b-bf16"
)

MODEL_IDS_Q4=(
    "gemma4:e4b"
    "gemma4:26b"
    "gemma4:31b"
)

MODEL_IDS_Q8=(
    "gemma4:e2b-it-q8_0"
    "gemma4:e4b-it-q8_0"
    "gemma4:26b-a4b-it-q8_0"
    "gemma4:31b-it-q8_0"
)

MODEL_IDS_BF16=(
    "gemma4:e2b-it-bf16"
    "gemma4:e4b-it-bf16"
    "gemma4:31b-it-bf16"
)

pull_and_test() {
    local model_id="$1"
    local bench_name="$2"
    local suite="$3"

    echo "========================================"
    echo "[$(date +%H:%M:%S)] Pulling: $model_id"
    echo "========================================"
    $OLLAMA_CMD pull "$model_id" 2>&1 | tail -3

    echo "[$(date +%H:%M:%S)] Testing: $bench_name ($suite)"
    $BENCH --models "$bench_name" --suite "$suite" --tag "r2_${bench_name//[:.]/_}_${suite}"
    echo "[$(date +%H:%M:%S)] Done: $bench_name"
    echo ""
}

echo "========================================"
echo "Phase 2A: Q4_K_M 默认量化 - Quick筛选"
echo "========================================"

for i in "${!MODEL_IDS_Q4[@]}"; do
    pull_and_test "${MODEL_IDS_Q4[$i]}" "${MODELS_Q4[$i]}" "quick"
done

echo "========================================"
echo "Phase 2B: Q4_K_M 默认量化 - Standard深度评测"
echo "========================================"

for model in "${MODELS_Q4[@]}"; do
    echo "[$(date +%H:%M:%S)] Standard test: $model"
    $BENCH --models "$model" --suite standard --tag "r2_${model//[:.]/_}_standard"
    echo ""
done

echo "========================================"
echo "Phase 2C: Q4_K_M 默认量化 - Toolcall评测"
echo "========================================"

for model in "${MODELS_Q4[@]}"; do
    echo "[$(date +%H:%M:%S)] Toolcall test: $model"
    $BENCH --models "$model" --suite toolcall --tag "r2_${model//[:.]/_}_toolcall"
    echo ""
done

echo "========================================"
echo "Phase 3: Q8_0 高精度量化"
echo "========================================"

for i in "${!MODEL_IDS_Q8[@]}"; do
    pull_and_test "${MODEL_IDS_Q8[$i]}" "${MODELS_Q8[$i]}" "standard"
done

echo "========================================"
echo "Phase 4: BF16 全精度"
echo "========================================"

for i in "${!MODEL_IDS_BF16[@]}"; do
    pull_and_test "${MODEL_IDS_BF16[$i]}" "${MODELS_BF16[$i]}" "standard"
done

echo "========================================"
echo "Phase 5: qwen3.5:35b 对照组 (standard + toolcall)"
echo "========================================"
$BENCH --models "qwen3.5:35b" --suite standard --tag r2_qwen35_35b_standard
$BENCH --models "qwen3.5:35b" --suite toolcall --tag r2_qwen35_35b_toolcall

echo ""
echo "========================================"
echo "ALL DONE! $(date)"
echo "========================================"
