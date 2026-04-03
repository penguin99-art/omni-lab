#!/bin/bash
# DGX Spark Model Setup Script
# Downloads models needed for comprehensive benchmarking
set -e

echo "=== DGX Spark Model Setup ==="
echo ""

# -----------------------------------------------
# Phase 1: Ollama models (easiest, no setup needed)
# -----------------------------------------------
echo "--- Phase 1: Ollama Models ---"

OLLAMA_MODELS=(
    "gpt-oss:120b"       # ~65GB, NVIDIA MoE, great tool calling
    "gpt-oss:20b"        # ~13GB, fast
    "nemotron-3-nano"     # ~24GB, fastest on Spark (~69 tok/s)
    "nemotron-3-super"    # ~86GB, quality focus
)

for model in "${OLLAMA_MODELS[@]}"; do
    echo ""
    echo "Pulling $model ..."
    ollama pull "$model"
done

# Nemotron-Cascade-2 needs a custom Modelfile
echo ""
echo "Setting up Nemotron-Cascade-2 (custom Modelfile)..."
ollama pull hf.co/mradermacher/Nemotron-Cascade-2-30B-A3B-GGUF:Q4_K_M

cat > /tmp/Modelfile.cascade2 << 'EOF'
FROM hf.co/mradermacher/Nemotron-Cascade-2-30B-A3B-GGUF:Q4_K_M
TEMPLATE {{ .Prompt }}
RENDERER nemotron-3-nano
PARSER nemotron-3-nano
PARAMETER top_p 1
PARAMETER temperature 1
EOF
ollama create nemotron-cascade-2 -f /tmp/Modelfile.cascade2

echo ""
echo "=== Phase 1 Complete ==="
echo "Installed Ollama models:"
ollama list

# -----------------------------------------------
# Phase 2: vLLM + namake-taro patch
# -----------------------------------------------
echo ""
echo "--- Phase 2: vLLM + namake-taro Patch ---"
echo "This gives the best performance (MXFP4 quantization)."
echo ""

VLLM_DIR="$HOME/vllm-spark"

if [ -d "$VLLM_DIR/.vllm" ]; then
    echo "vLLM environment already exists at $VLLM_DIR"
else
    echo "Creating vLLM environment..."
    uv venv -p 3.12 "$VLLM_DIR/.vllm"
    cd "$VLLM_DIR"
    source .vllm/bin/activate

    uv pip install vllm \
        --extra-index-url https://wheels.vllm.ai/0.17.1/cu130 \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match

    uv pip install 'nvidia-nccl-cu13>=2.29.2' fastsafetensors

    echo "Applying namake-taro patches..."
    git clone https://github.com/namake-taro/vllm-custom.git "$VLLM_DIR/vllm-custom"

    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    cd "$SITE"
    patch -p0 < "$VLLM_DIR/vllm-custom/patches/vllm_all.patch"
    patch -p0 < "$VLLM_DIR/vllm-custom/patches/flashinfer_cutlass_sfb_layout_fix.patch"

    # Clear caches
    rm -rf ~/.cache/flashinfer/ ~/.cache/vllm/torch_compile_cache/ \
        ~/.cache/vllm/torch_aot_compile/ /tmp/torchinductor_$(whoami)/

    echo "vLLM setup complete."
fi

# Create startup script
cat > "$VLLM_DIR/start-vllm.sh" << 'SCRIPT'
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export VLLM_MXFP4_BACKEND=marlin
export VLLM_MARLIN_USE_ATOMIC_ADD=1
cd ~/vllm-spark
source .vllm/bin/activate
exec vllm serve "$@"
SCRIPT
chmod +x "$VLLM_DIR/start-vllm.sh"

echo ""
echo "=== Phase 2 Complete ==="
echo ""
echo "To start vLLM with Qwen3.5-35B-A3B (MXFP4):"
echo "  ~/vllm-spark/start-vllm.sh Qwen/Qwen3.5-35B-A3B \\"
echo "    --host 0.0.0.0 --port 18080 \\"
echo "    --gpu-memory-utilization 0.85 \\"
echo "    --enforce-eager \\"
echo "    --quantization mxfp4 \\"
echo "    --kv-cache-dtype fp8"

# -----------------------------------------------
# Phase 3: vLLM Docker (Qwen3.5-27B-FP8, best tool calling)
# -----------------------------------------------
echo ""
echo "--- Phase 3: vLLM Docker (Qwen3.5-27B-FP8) ---"
echo ""
echo "To start (tool calling champion):"
echo "  docker run --gpus all --network host --ipc host \\"
echo "    --name vllm-qwen35-fp8 \\"
echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
echo "    -d vllm/vllm-openai:qwen3_5-cu130 \\"
echo "    Qwen/Qwen3.5-27B-FP8 \\"
echo "    --host 0.0.0.0 --port 18080 \\"
echo "    --max-model-len 32000 \\"
echo "    --gpu-memory-utilization 0.80 \\"
echo "    --max-num-seqs 2 \\"
echo "    --enforce-eager \\"
echo "    --language-model-only \\"
echo "    --enable-auto-tool-choice \\"
echo "    --tool-call-parser qwen3_coder \\"
echo "    --reasoning-parser qwen3 \\"
echo "    --enable-prefix-caching"

echo ""
echo "=== All Setup Complete ==="
