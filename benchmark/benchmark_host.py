"""
benchmark_host.py
-----------------
Runs inference benchmarks on the compiled MLC-LLM model on the host machine
(Linux/macOS) and generates a CSV + summary report.

Measures:
  - Time to First Token (TTFT) in milliseconds
  - Tokens Per Second (TPS) for decode phase
  - Memory footprint (RSS) in MB

Usage:
    python benchmark_host.py --model-dir ./dist/phi-1_5-q4f16_1-MLC \
        --prompts-file prompts.txt --n-runs 20
"""

import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path

# Synthetic prompts for arbitrary / automated testing
SYNTHETIC_PROMPTS = [
    "Explain the concept of neural network quantization in simple terms.",
    "What are the trade-offs between 3-bit and 4-bit weight quantization?",
    "Write a Python function that computes the Fibonacci sequence.",
    "Describe the Vulkan rendering pipeline for a mobile GPU.",
    "How does key-value cache management affect LLM inference latency?",
    "Summarize the main advantages of on-device AI over cloud inference.",
    "What is operator fusion in a compiler, and why does it help performance?",
    "Write a short poem about running AI models on mobile hardware.",
    "Explain Apache TVM's Relax IR in one paragraph.",
    "List five applications that would benefit from offline LLM inference.",
    "What is the difference between OpenCL and Vulkan compute shaders?",
    "Describe how Android's JNI bridge connects Kotlin UI to a C++ inference engine.",
    "What metrics should you measure when benchmarking an on-device LLM?",
    "Explain memory bandwidth bottlenecks in mobile LLM inference.",
    "Write pseudocode for a sliding-window KV cache eviction policy.",
    "What is the Qualcomm Adreno GPU architecture?",
    "How does group quantization differ from per-channel quantization?",
    "Describe a typical first-token latency budget for a mobile chat app.",
    "Why is perplexity important when evaluating quantized models?",
    "Explain thermal throttling and its effect on sustained LLM inference.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Path to MLC model directory")
    p.add_argument("--prompts-file", type=str, default=None, help="Optional plain-text file of prompts (one per line)")
    p.add_argument("--n-runs", type=int, default=20, help="Number of inference runs")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output-csv", type=str, default="benchmark_results.csv")
    return p.parse_args()


def get_memory_mb() -> float:
    """Return current process resident set size in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux, bytes on macOS
    if os.uname().sysname == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def run_benchmark(model_dir: str, prompts: list[str], max_new_tokens: int, output_csv: str):
    try:
        from mlc_llm import MLCEngine
    except ImportError:
        print("[ERROR] mlc_llm not installed. Run: pip install mlc-llm-nightly-cpu")
        return

    print(f"[INFO] Loading model from {model_dir}...")
    engine = MLCEngine(model_dir)

    results = []
    for i, prompt in enumerate(prompts):
        mem_before = get_memory_mb()
        t_start = time.perf_counter()

        # First token timing
        first_token_time = None
        full_output = ""
        token_count = 0

        for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            stream=True,
        ):
            delta = response.choices[0].delta.content or ""
            if delta and first_token_time is None:
                first_token_time = time.perf_counter()
            full_output += delta
            token_count += 1

        t_end = time.perf_counter()
        mem_after = get_memory_mb()

        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else None
        total_elapsed = t_end - t_start
        # Decode TPS excludes first-token latency
        decode_elapsed = (t_end - first_token_time) if first_token_time else total_elapsed
        tps = (token_count - 1) / decode_elapsed if decode_elapsed > 0 and token_count > 1 else 0.0

        row = {
            "run": i + 1,
            "prompt_tokens": len(prompt.split()),
            "output_tokens": token_count,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms else "N/A",
            "tps_decode": round(tps, 2),
            "total_time_s": round(total_elapsed, 3),
            "mem_delta_mb": round(mem_after - mem_before, 1),
            "prompt_snippet": prompt[:60].replace(",", ";"),
        }
        results.append(row)
        print(f"  Run {i+1:02d}/{len(prompts)} | TTFT={row['ttft_ms']}ms | TPS={row['tps_decode']} | tokens={token_count}")

    # Write CSV
    fieldnames = ["run", "prompt_tokens", "output_tokens", "ttft_ms", "tps_decode",
                  "total_time_s", "mem_delta_mb", "prompt_snippet"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    valid_tps = [r["tps_decode"] for r in results if isinstance(r["tps_decode"], float)]
    valid_ttft = [r["ttft_ms"] for r in results if isinstance(r["ttft_ms"], float)]
    print("\n=== SUMMARY ===")
    print(f"  Runs:         {len(results)}")
    print(f"  Avg TPS:      {sum(valid_tps)/len(valid_tps):.2f}" if valid_tps else "  Avg TPS: N/A")
    print(f"  Avg TTFT:     {sum(valid_ttft)/len(valid_ttft):.1f} ms" if valid_ttft else "  Avg TTFT: N/A")
    print(f"  Results CSV:  {output_csv}")


def main():
    args = parse_args()
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts = Path(args.prompts_file).read_text().strip().splitlines()
    else:
        prompts = SYNTHETIC_PROMPTS
    prompts = prompts[: args.n_runs]
    run_benchmark(args.model_dir, prompts, args.max_new_tokens, args.output_csv)


if __name__ == "__main__":
    main()
