"""Replay captured vLLM inputs through the two PyTorch INT8 attention paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .pytorch_ref import ref_paged_attn_quantized, relative_l2
from .reporting import default_output_dir, write_report
from .replay_capture import _output_as_heads


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    return {
        "relative_l2": relative_l2(actual, expected),
        "max_abs_error": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "nonfinite_values": int((~torch.isfinite(actual)).sum().item()),
    }


def validate_capture(capture: dict[str, Any], *, statistic: str = "absmax") -> dict[str, Any]:
    """Measure dequant and INT8-compute errors against captured native outputs."""
    steps = capture.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("capture has no steps; run Stage 2 capture first")
    comparisons: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for step_index, step in enumerate(steps):
        for layer_index, layer in enumerate(step["layers"]):
            if layer.get("causal") is not True:
                skipped += 1
                continue
            query = layer["query"]
            native = _output_as_heads(layer["output"], query)
            for kv_granularity in ("per_head", "per_channel"):
                dequant = ref_paged_attn_quantized(
                    query,
                    layer["key_cache"],
                    layer["value_cache"],
                    layer["cu_seqlens_q"],
                    layer["seqused_k"],
                    layer["block_table"],
                    implementation="dequant",
                    granularity=kv_granularity,
                    statistic=statistic,
                    softmax_scale=float(layer["softmax_scale"]),
                    causal=True,
                )
                key = f"kv_{kv_granularity}/dequant_q_fp"
                comparisons.setdefault(key, []).append({
                    "step": step_index,
                    "layer": layer_index,
                    "vs_native": _metrics(dequant, native),
                })
                for query_granularity in ("per_head", "per_channel"):
                    output = ref_paged_attn_quantized(
                        query,
                        layer["key_cache"],
                        layer["value_cache"],
                        layer["cu_seqlens_q"],
                        layer["seqused_k"],
                        layer["block_table"],
                        implementation="int8_gemm",
                        granularity=kv_granularity,
                        query_granularity=query_granularity,
                        statistic=statistic,
                        softmax_scale=float(layer["softmax_scale"]),
                        causal=True,
                    )
                    key = f"kv_{kv_granularity}/q_{query_granularity}/int8_gemm"
                    comparisons.setdefault(key, []).append({
                        "step": step_index,
                        "layer": layer_index,
                        "vs_native": _metrics(output, native),
                        "vs_dequant_q_fp": _metrics(output, dequant),
                    })
    if not comparisons:
        raise ValueError("capture contains no causal attention layers to validate")
    summary: dict[str, Any] = {}
    for key, rows in comparisons.items():
        native_errors = [row["vs_native"]["relative_l2"] for row in rows]
        compute_errors = [row["vs_dequant_q_fp"]["relative_l2"] for row in rows if "vs_dequant_q_fp" in row]
        summary[key] = {
            "layers_compared": len(rows),
            "max_relative_l2_vs_native": max(native_errors),
            "mean_relative_l2_vs_native": sum(native_errors) / len(native_errors),
            "max_relative_l2_vs_dequant_q_fp": max(compute_errors) if compute_errors else None,
            "mean_relative_l2_vs_dequant_q_fp": sum(compute_errors) / len(compute_errors) if compute_errors else None,
            "max_nonfinite_values": max(row["vs_native"]["nonfinite_values"] for row in rows),
            "layer_results": rows,
        }
    dequant_gate = all(
        result["max_relative_l2_vs_native"] <= 0.05
        for name, result in summary.items()
        if name.endswith("dequant_q_fp")
    )
    gemm_gate = all(
        result["max_relative_l2_vs_dequant_q_fp"] is not None
        and result["max_relative_l2_vs_dequant_q_fp"] <= 0.10
        for name, result in summary.items()
        if name.endswith("int8_gemm")
    )
    return {
        "capture_format": capture.get("format"),
        "model": capture.get("model"),
        "dtype": capture.get("dtype"),
        "sequence_length": capture.get("sequence_length"),
        "layer_count": len(steps[-1]["layers"]),
        "step_count": len(steps),
        "layers_skipped_noncausal": skipped,
        "scale_statistic": statistic,
        "quality_gates": {
            "dequant_vs_native_relative_l2_max": 0.05,
            "int8_gemm_vs_dequant_relative_l2_max": 0.10,
            "dequant_pass": dequant_gate,
            "int8_gemm_pass": gemm_gate,
            "overall_pass": dequant_gate and gemm_gate,
        },
        "comparisons": summary,
        "interpretation": (
            "dequant_q_fp isolates KV quantization with FP Q. INT8-GEMM groups quantize Q at the "
            "named per-head/per-channel granularity and report error versus both native output and FP-Q dequant."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analysis-report", type=Path)
    args = parser.parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    statistic = "absmax"
    recommendation = None
    if args.analysis_report:
        analysis = json.loads(args.analysis_report.read_text())
        recommendation = analysis.get("recommendation", {})
        statistic = recommendation.get("scale_statistic", "absmax")
        if statistic not in ("absmax", "p999"):
            raise ValueError("analysis report has an unsupported scale_statistic")
    report = validate_capture(capture, statistic=statistic)
    report["stage2_recommendation"] = recommendation
    report["capture_path"] = str(args.capture)
    path = write_report({"experiment": "stage4_capture_numerics", **report}, args.output_dir or default_output_dir())
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, indent=2))
    for name, result in report["comparisons"].items():
        print(f"{name}: max_rel_l2_vs_native={result['max_relative_l2_vs_native']:.6g}; "
              f"max_rel_l2_vs_dequant_q_fp={result['max_relative_l2_vs_dequant_q_fp']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
