"""Measure FP reference vs correctness-first dynamic INT8 paged attention."""

from __future__ import annotations

import argparse
import statistics

import torch

from int8_kvcache_lab import PagedKVCache, QuantConfig, paged_attention_dynamic_int8, paged_attention_reference
from int8_kvcache_lab.reporting import write_report


def measure(fn, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), sorted(samples)[int(0.95 * (len(samples) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    heads, kv_heads, dim = 32, 8, 128
    pages = (args.seq_len + args.block_size - 1) // args.block_size
    cache = PagedKVCache.empty(args.batch * pages, args.block_size, kv_heads, dim, dtype=torch.float16, device="cuda")
    table = torch.arange(args.batch * pages, device="cuda", dtype=torch.long).view(args.batch, pages)
    slots = torch.cat([table[row].repeat_interleave(args.block_size)[: args.seq_len] * args.block_size + torch.arange(args.seq_len, device="cuda") % args.block_size for row in range(args.batch)])
    cache.write(torch.randn(args.batch * args.seq_len, kv_heads, dim, device="cuda", dtype=torch.float16), torch.randn(args.batch * args.seq_len, kv_heads, dim, device="cuda", dtype=torch.float16), slots)
    query = torch.randn(args.batch, heads, dim, device="cuda", dtype=torch.float16)
    lengths = torch.full((args.batch,), args.seq_len, device="cuda", dtype=torch.long)
    expected = paged_attention_reference(query, cache, table, lengths)
    actual, quant_stats = paged_attention_dynamic_int8(query, cache, table, lengths, QuantConfig(args.block_size))
    relative_l2 = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8))
    fp_median, fp_p95 = measure(lambda: paged_attention_reference(query, cache, table, lengths), 20, args.repeat)
    int8_median, int8_p95 = measure(lambda: paged_attention_dynamic_int8(query, cache, table, lengths, QuantConfig(args.block_size)), 20, args.repeat)
    report = write_report({"kind": "dynamic_int8_benchmark", "shape": [args.batch, args.seq_len, heads, kv_heads, dim], "attention_relative_l2_error": relative_l2, "quantization": {key: (value.tolist() if isinstance(value, torch.Tensor) else value) for key, value in quant_stats.items()}, "fp16_ms": {"median": fp_median, "p95": fp_p95}, "dynamic_int8_ms": {"median": int8_median, "p95": int8_p95}, "note": "Dynamic quantization includes findmax and temporary cache allocation; it is not expected to be faster."}, args.output_dir)
    print(f"fp16 median={fp_median:.4f}ms p95={fp_p95:.4f}ms")
    print(f"dynamic_int8 median={int8_median:.4f}ms p95={int8_p95:.4f}ms")
    print(f"report={report}")


if __name__ == "__main__":
    main()
