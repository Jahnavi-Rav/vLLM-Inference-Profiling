# vLLM Inference Profiling — Batching and Quantization Analysis

A profiling study of generative AI inference performance on a T4 GPU using
vLLM and Mistral-7B-AWQ. The goal was to measure how batch size affects
GPU utilization, throughput, and per-request latency under a memory-bound
workload.

## Setup

- GPU: NVIDIA T4 (16GB VRAM, Turing architecture, sm75)
- Framework: vLLM (latest), PyTorch
- Model: Mistral-7B-Instruct-v0.2 (AWQ 4-bit quantized)
- Environment: Google Colab

## What This Experiment Does

Runs the same inference workload at three concurrency levels (1, 4, and 8
simultaneous requests) and measures throughput and latency at each level.
Profiling is enabled via vLLM's PyTorch profiler integration, which dumps
execution traces to /tmp/vllm_traces/ for analysis in Perfetto.

Also tests AWQ-Marlin kernel compatibility as an additional optimization path.

## Results

| Batch Size | Throughput (tok/s) | Latency per Request |
|------------|-------------------|---------------------|
| 1          | 3.1               | 64.22s              |
| 4          | 13.6              | 14.68s              |
| 8          | 20.2              | 9.91s               |

Throughput improvement from batch size 1 to 8: **6.5x**

## Key Findings

**Memory-bound decode phase**
The single-request run (3.1 tok/s) shows the GPU is severely underutilized
when serving one request at a time. The decode phase achieves approximately
2 FLOPs/byte arithmetic intensity — well below the T4's compute ridge point.
The GPU spends most of its time waiting on HBM reads, not computing.

**Batching hides memory latency**
Increasing concurrency forces the GPU scheduler to overlap memory fetches
across multiple requests. This keeps the HBM bus saturated and pushes
throughput from 3.1 to 20.2 tok/s without any model changes.

**Scaling curve behavior**
The throughput gain is superlinear from batch 1 to 4 (4.4x gain for 4x
requests), then sublinear from 4 to 8 (1.5x gain for 2x requests). This
indicates the GPU approaching memory bandwidth saturation around batch
size 8.

**AWQ-Marlin incompatibility on Turing**
AWQ-Marlin is a specialized INT4 GEMM kernel that eliminates the FP16
dequantization step during decode. Attempting to use it on the T4 resulted
in engine initialization failure — Marlin requires Ampere (sm80+) warp-level
instructions not available on Turing (sm75). This confirms that low-level
kernel optimizations are architecture-specific and cannot be assumed portable.

## How to Run

```bash
pip install vllm
python vllm_profiling.py
```

Traces are saved to /tmp/vllm_traces/. To visualize, open
https://ui.perfetto.dev and drag in the generated .json.gz file.

## Files

- vllm_profiling.py — main benchmarking script with profiler config
- README.md — this file

## Background

This experiment was done as part of a broader study on generative AI
inference optimization, documented in the paper:
"Profiling and Optimizing Generative AI Inference: A Practical Guide to
Identifying Bottlenecks and Maximizing Accelerator Performance" (2026).
