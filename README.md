# CSCE 685 — Offline LLM Optimization and Deployment on Android

**Student:** Agastya Todi (UIN: 835009977)  
**Supervisor:** Professor Duncan Walker  
**Semester:** Spring 2026

## Overview

This repository contains the full source code, model conversion scripts, benchmark tooling,
and documentation for deploying Microsoft Phi-1.5 (1.3B parameters) natively on Android devices
using the MLC-LLM framework and Apache TVM compiler stack.

**Achieved: ~3–4 tokens/second on-device decode throughput (Phi-1.5, INT4 quantization).**

## Repository Structure

```
csce685_offline_llm/
├── scripts/
│   ├── 01_setup_environment.sh     # Install Python deps, check NDK
│   └── 02_convert_model.py         # HF → MLC conversion + Android compilation
├── benchmark/
│   └── benchmark_host.py           # Host-side inference benchmarking (TPS, TTFT)
├── android_app/
│   └── app/
│       └── src/main/
│           ├── kotlin/com/csce685/offlinellm/
│           │   ├── MainActivity.kt  # App entry point, lifecycle management
│           │   ├── MLCBridge.kt     # JNI wrapper (Kotlin)
│           │   └── ChatScreen.kt    # Jetpack Compose streaming UI
│           └── cpp/
│               ├── mlc_jni.cpp      # C++ JNI implementation
│               └── CMakeLists.txt   # NDK build configuration
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
bash scripts/01_setup_environment.sh
source ~/mlc_env/bin/activate
```

Requires:
- Python 3.11+
- Android NDK r26b (`export ANDROID_NDK_HOME=/path/to/ndk`)
- LLVM 17+ (for TVM cross-compilation)

### 2. Convert Model

```bash
# INT4 quantization (recommended for 8 GB RAM devices)
python scripts/02_convert_model.py --model phi-1_5 --quantization q4f16_1

# 3-bit variant (smaller, slightly lower quality)
python scripts/02_convert_model.py --model phi-1_5 --quantization q3f16_1
```

Outputs compiled artifacts to `./dist/`.

### 3. Run Host Benchmarks

```bash
python benchmark/benchmark_host.py \
    --model-dir ./dist/phi-1_5-q4f16_1-MLC \
    --n-runs 20 \
    --output-csv results/host_benchmark.csv
```

### 4. Build Android App

1. Copy `dist/phi-1_5-q4f16_1-MLC` to `android_app/app/src/main/assets/mlc_models/`
2. Copy pre-built `.so` files to `android_app/app/src/main/jniLibs/arm64-v8a/`
3. Open `android_app/` in Android Studio
4. Click **Build → Make Project**, then **Run**

> A physical device with ≥8 GB RAM is required (no emulator support for GPU inference).

## Key Results

| Metric | Value | Device |
|---|---|---|
| Decode TPS (Phi-1.5 q4f16_1) | ~3–4 tok/s | Mid-range Android (Snapdragon) |
| Time to First Token (TTFT) | ~800–1200 ms | Same device |
| Model size on-disk (INT4) | ~700 MB | — |
| Peak RAM during inference | ~1.5–2.0 GB | — |

## Dependencies

See `requirements.txt`. Key packages:
- `mlc-llm-nightly-cpu` — MLC-LLM Python package
- `apache-tvm` — TVM Unity compiler
- `transformers`, `huggingface_hub` — model download

## License

MIT License — see LICENSE file.
