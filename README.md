# Dynamic INT8 KV Cache Lab

This is a correctness-first, Google Colab-friendly project for exploring a
dynamic INT8 KV-cache decode path. It models paged KV storage with physical
layout `[num_blocks, block_size, 2, num_kv_heads, head_dim]` and implements:

- FP32-accumulating FP reference PagedAttention;
- symmetric INT8 quantization where Q is per sequence tensor and K/V are
  separately per KV head over valid paged tokens;
- GQA-aware INT8 dequantized attention;
- an opt-in Qwen2.5 Transformers adapter for batch-one decode;
- teacher-forced WikiText-2 PPL and JSON experiment reports.

The dynamic path recomputes scales, quantizes the full FP cache, and creates a
temporary INT8 cache on every decode step. It validates numerical behavior,
but is **not expected to outperform FP16**. Static write-time quantization and
fused kernels are deliberately out of scope.

## Run in Google Colab

1. Select a GPU runtime with at least 24 GiB VRAM (L4/A100). Qwen2.5-7B in
   FP16 is intentionally rejected on smaller runtimes to avoid conflating an
   out-of-memory failure with a quantization result.
2. Open `notebooks/colab_int8_kvcache.ipynb`, set `REPO_URL`, and run it.
3. The bootstrap script installs this package, checks out vLLM at the revision
   in `configs/versions.env` for source comparison only, runs tests, the
   synthetic benchmark, and Qwen PPL evaluation.

The notebook never installs or patches vLLM. The adapter is built around the
eager Qwen2 API from the pinned dependency range and fails explicitly if that
contract changes.

## Local commands

```bash
python -m pip install -e '.[dev]'
pytest -n 4 -q
python benchmarks/benchmark_dynamic.py
python -m int8_kvcache_lab.evaluation --samples 4 --context 128
```

Tests run on CPU where possible. Benchmarking and the model evaluation require
CUDA. Results are written to ignored `artifacts/run-*.json` files and include
GPU, CUDA, package, shape, accuracy, cache-byte, and latency metadata.

## Quality gates

- Synthetic paged attention relative L2 error: `<= 5%`.
- Qwen decode PPL relative increase over the native FP16 baseline: `<= 1%` on
  the deterministic WikiText-2 subset.

The PPL evaluation uses teacher forcing, native FP16 prefill, and quantized
non-padded batch-one decode. It is deliberately restricted to
`Qwen/Qwen2.5-7B-Instruct`; unsupported batches, prefill calls, and disabled
layers delegate to the original Transformers implementation.

## Next stage

Replace the per-step full-cache `absmax + quantize + temporary cache` path with
static calibration, INT8 cache writes during prefill/decode, and an attention
kernel that consumes the resident INT8 data directly.
