#!/usr/bin/env bash
# ============================================================
# 01_setup_environment.sh
# Sets up the Python virtual environment and installs all
# dependencies required for MLC-LLM Android deployment.
# ============================================================

set -euo pipefail

PYTHON=python3.11
VENV_DIR="$HOME/mlc_env"

echo "[INFO] Creating virtual environment at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[INFO] Installing MLC-LLM and TVM Unity..."
pip install --upgrade pip
pip install mlc-llm-nightly-cpu   # use nightly build with CUDA if on a GPU host
pip install apache-tvm             # TVM Unity runtime (CPU wheel for compilation host)

echo "[INFO] Installing auxiliary packages..."
pip install transformers huggingface_hub onnx onnxruntime torch torchvision

echo "[INFO] Verifying MLC-LLM installation..."
python -c "import mlc_llm; print('[OK] mlc_llm version:', mlc_llm.__version__)"

echo "[INFO] Checking Android NDK (expected at \$ANDROID_NDK_HOME)..."
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "[WARN] ANDROID_NDK_HOME not set. Please install Android NDK r26b or later"
    echo "       and export ANDROID_NDK_HOME=/path/to/android-ndk-r26b"
else
    echo "[OK] NDK found at $ANDROID_NDK_HOME"
fi

echo "[DONE] Environment setup complete. Activate with: source $VENV_DIR/bin/activate"
