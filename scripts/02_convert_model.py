"""
02_convert_model.py
-------------------
Downloads microsoft/phi-1_5 from HuggingFace, converts it to
MLC-LLM format with INT4 group quantization (q4f16_1), and
exports ONNX weights for cross-validation.

Usage:
    python 02_convert_model.py --model phi-1_5 --quantization q4f16_1
    python 02_convert_model.py --model phi-1_5 --quantization q3f16_1  # 3-bit variant
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

SUPPORTED_MODELS = {
    "phi-1_5": "microsoft/phi-1_5",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
}

QUANTIZATIONS = ["q4f16_1", "q3f16_1", "q4f32_1"]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to MLC-LLM format")
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS.keys()), default="phi-1_5")
    parser.add_argument("--quantization", choices=QUANTIZATIONS, default="q4f16_1")
    parser.add_argument("--output-dir", type=str, default="./dist")
    parser.add_argument("--export-onnx", action="store_true", help="Also export ONNX for cross-validation")
    return parser.parse_args()


def download_hf_model(hf_repo: str, local_dir: Path) -> None:
    """Download model weights from HuggingFace Hub."""
    print(f"[INFO] Downloading {hf_repo} to {local_dir}...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )
    print("[OK] Download complete.")


def convert_to_mlc(model_path: Path, quantization: str, output_dir: Path) -> Path:
    """
    Run mlc_llm convert_weight and gen_config to produce MLC model artifacts.
    Returns the path to the output model directory.
    """
    model_name = model_path.name
    out_path = output_dir / f"{model_name}-{quantization}-MLC"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting weights to {quantization}...")
    subprocess.run(
        [
            sys.executable, "-m", "mlc_llm", "convert_weight",
            str(model_path),
            "--quantization", quantization,
            "--output", str(out_path),
        ],
        check=True,
    )

    print("[INFO] Generating MLC config...")
    subprocess.run(
        [
            sys.executable, "-m", "mlc_llm", "gen_config",
            str(model_path),
            "--quantization", quantization,
            "--prefill-chunk-size", "512",
            "--context-window-size", "2048",
            "--output", str(out_path),
        ],
        check=True,
    )
    print(f"[OK] MLC model written to {out_path}")
    return out_path


def compile_for_android(mlc_model_path: Path, output_dir: Path) -> None:
    """
    Compile the MLC model library (.tar -> .so) for Android arm64-v8a
    targeting OpenCL and Vulkan backends.
    """
    so_name = mlc_model_path.name.replace("-MLC", "") + "-android.tar"
    lib_out = output_dir / so_name

    print("[INFO] Compiling model library for android/arm64 (OpenCL + Vulkan)...")
    subprocess.run(
        [
            sys.executable, "-m", "mlc_llm", "compile",
            str(mlc_model_path),
            "--device", "android",
            "--output", str(lib_out),
        ],
        check=True,
    )
    print(f"[OK] Android library written to {lib_out}")


def export_onnx(hf_model_path: Path, output_dir: Path) -> None:
    """Export model to ONNX format for cross-validation and benchmarking."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[INFO] Loading model for ONNX export (this may take a few minutes)...")
    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_model_path),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    dummy_input = tokenizer("Hello, world!", return_tensors="pt")
    onnx_path = output_dir / f"{hf_model_path.name}.onnx"

    print(f"[INFO] Exporting ONNX to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"],),
        str(onnx_path),
        opset_version=17,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}, "logits": {0: "batch", 1: "seq_len"}},
    )
    print(f"[OK] ONNX model exported to {onnx_path}")


def main():
    args = parse_args()
    hf_repo = SUPPORTED_MODELS[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_hf_dir = output_dir / "hf_weights" / args.model
    download_hf_model(hf_repo, local_hf_dir)

    mlc_path = convert_to_mlc(local_hf_dir, args.quantization, output_dir)
    compile_for_android(mlc_path, output_dir)

    if args.export_onnx:
        export_onnx(local_hf_dir, output_dir)

    print("\n[DONE] All artifacts ready in:", output_dir)
    print("  MLC model:  ", mlc_path)


if __name__ == "__main__":
    main()
