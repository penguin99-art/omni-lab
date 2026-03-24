#!/bin/bash
#
# MiniCPM-o 4.5 全双工语音+摄像头对话 一键启动脚本
#
# 用法:
#   bash run_omni_demo.sh              # 自动启动 server + 双工模式
#   bash run_omni_demo.sh --simplex    # 单工模式
#   bash run_omni_demo.sh --no-camera  # 纯语音（不用摄像头）
#   bash run_omni_demo.sh --no-server  # 使用已启动的 server
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="$SCRIPT_DIR/models/MiniCPM-o-4_5-gguf"
SERVER_BIN="$SCRIPT_DIR/llama.cpp-omni/build/bin/llama-server"

# 检查 server 二进制
if [ ! -f "$SERVER_BIN" ]; then
    echo "[ERROR] llama-server 未编译，请先运行:"
    echo "  cd llama.cpp-omni/build && cmake --build . --target llama-server -j\$(nproc)"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_DIR/MiniCPM-o-4_5-Q4_K_M.gguf" ]; then
    echo "[ERROR] 模型文件不存在: $MODEL_DIR/MiniCPM-o-4_5-Q4_K_M.gguf"
    exit 1
fi

echo "============================================="
echo "  MiniCPM-o 4.5 Omni Demo 启动器"
echo "============================================="
echo ""

exec python3 "$SCRIPT_DIR/omni_duplex_demo.py" "$@"
