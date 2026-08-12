"""KV-cache capture analysis and quantization-granularity calibration.

The calibration data is deliberately kept separate from the dynamic runtime
path.  It answers which *static* granularity best represents the KV values
observed during a real vLLM decode, without silently changing the current
per-head dynamic experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from .quantization import INT8_MAX
from .reporting import write_report


_CANDIDATES = ("per_tensor", "per_head", "per_token", "per_channel")


def _safe_scale(absmax: torch.Tensor, eps: float) -> torch.Tensor:
    """Turn an absmax tensor into non-zero symmetric INT8 scales."""
    return torch.where(absmax > eps, absmax / INT8_MAX, torch.ones_like(absmax))


def _scale_for(values: torch.Tensor, granularity: str, eps: float) -> torch.Tensor:
    """Return one scale per requested KV-cache quantization group.

    ``values`` has canonical logical shape ``[tokens, 2, kv_heads, head_dim]``.
    The first two granularities are per layer: K and V remain separate so that
    their scales follow their individual distributions.
    """
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("values must have shape [tokens, 2, kv_heads, head_dim]")
    absolute = values.float().abs()
    if granularity == "per_tensor":
        return _safe_scale(absolute.amax(dim=(0, 2, 3)), eps)[None, :, None, None]
    if granularity == "per_head":
        return _safe_scale(absolute.amax(dim=(0, 3)), eps)[None, :, :, None]
    if granularity == "per_token":
        return _safe_scale(absolute.amax(dim=(2, 3)), eps)[:, :, None, None]
    if granularity == "per_channel":
        # A token/head feature vector is the "channel" used by this lab.
        return _safe_scale(absolute.amax(dim=3), eps)[:, :, :, None]
    raise ValueError(f"unsupported granularity: {granularity}")


def _error_for(values: torch.Tensor, granularity: str, eps: float) -> dict[str, float | int]:
    scale = _scale_for(values, granularity, eps)
    restored = (torch.round(values.float() / scale).clamp(-127, 127) * scale)
    error = restored - values.float()
    return {
        "squared_error": float(error.square().sum().item()),
        "squared_signal": float(values.float().square().sum().item()),
        "max_abs_error": float(error.abs().max().item()),
        "scale_count": int(scale.numel()),
    }


def _percentiles(values: torch.Tensor) -> dict[str, float]:
    absolute = values.float().abs().flatten()
    if not absolute.numel():
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "p999": 0.0, "absmax": 0.0}
    return {
        "p50": float(torch.quantile(absolute, 0.50).item()),
        "p90": float(torch.quantile(absolute, 0.90).item()),
        "p99": float(torch.quantile(absolute, 0.99).item()),
        "p999": float(torch.quantile(absolute, 0.999).item()),
        "absmax": float(absolute.max().item()),
    }


def _histogram(values: torch.Tensor, bins: int = 64) -> dict[str, Any]:
    absolute = values.float().abs().flatten()
    maximum = float(absolute.max().item()) if absolute.numel() else 0.0
    if maximum == 0.0:
        return {"absmax": 0.0, "bins": [0] * bins, "bin_edges": [0.0] * (bins + 1)}
    return {
        "absmax": maximum,
        "bins": [int(value) for value in torch.histc(absolute, bins=bins, min=0, max=maximum).tolist()],
        "bin_edges": [maximum * index / bins for index in range(bins + 1)],
    }


def _iter_layers(capture: dict[str, Any]) -> Iterable[torch.Tensor]:
    layers = capture.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("capture must contain at least one logical KV-cache layer")
    for index, layer in enumerate(layers):
        if not isinstance(layer, torch.Tensor):
            raise TypeError(f"capture layer {index} is not a tensor")
        if layer.ndim != 4 or layer.shape[1] != 2:
            raise ValueError(f"capture layer {index} must have shape [tokens, 2, heads, dim]")
        yield layer.cpu()


def analyze_capture(
    capture: dict[str, Any],
    *,
    relative_l2_target: float = 0.01,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Measure distribution tails and candidate INT8 reconstruction errors.

    The recommendation is the cheapest scale layout that satisfies the target
    relative-L2 reconstruction error.  The report includes all candidates so a
    performance/accuracy choice remains auditable rather than implicit.
    """
    if not 0 < relative_l2_target < 1:
        raise ValueError("relative_l2_target must lie in (0, 1)")
    layers = list(_iter_layers(capture))
    by_kind = {"key": [], "value": []}
    totals = {name: {"squared_error": 0.0, "squared_signal": 0.0, "max_abs_error": 0.0, "scale_count": 0} for name in _CANDIDATES}
    layer_summaries: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        by_kind["key"].append(layer[:, 0])
        by_kind["value"].append(layer[:, 1])
        candidate_errors: dict[str, Any] = {}
        for name in _CANDIDATES:
            metrics = _error_for(layer, name, eps)
            candidate_errors[name] = metrics
            for metric in ("squared_error", "squared_signal", "scale_count"):
                totals[name][metric] += metrics[metric]
            totals[name]["max_abs_error"] = max(totals[name]["max_abs_error"], metrics["max_abs_error"])
        layer_summaries.append({"layer_index": index, "tokens": int(layer.shape[0]), "candidate_errors": candidate_errors})

    candidates: dict[str, Any] = {}
    for name in _CANDIDATES:
        total = totals[name]
        signal = max(float(total["squared_signal"]), 1e-20)
        candidates[name] = {
            "relative_l2": math.sqrt(float(total["squared_error"]) / signal),
            "max_abs_error": total["max_abs_error"],
            "scale_count": int(total["scale_count"]),
            "scale_bytes_fp32": int(total["scale_count"]) * 4,
        }
    eligible = [name for name in _CANDIDATES if candidates[name]["relative_l2"] <= relative_l2_target]
    recommendation = eligible[0] if eligible else min(_CANDIDATES, key=lambda name: candidates[name]["relative_l2"])
    return {
        "capture": {key: value for key, value in capture.items() if key != "layers"},
        "layer_count": len(layers),
        "relative_l2_target": relative_l2_target,
        "recommendation": {
            "kv_granularity": recommendation,
            "meets_target": bool(eligible),
            "rationale": "lowest scale-overhead candidate meeting the reconstruction-error target" if eligible else "no candidate met target; selected the lowest reconstruction error",
        },
        "candidates": candidates,
        "distribution": {
            "key": {"percentiles": _percentiles(torch.cat(by_kind["key"])), "histogram": _histogram(torch.cat(by_kind["key"]))},
            "value": {"percentiles": _percentiles(torch.cat(by_kind["value"])), "histogram": _histogram(torch.cat(by_kind["value"]))},
        },
        "layers": layer_summaries,
    }


def main() -> None:
    """Analyze a ``vllm_capture`` artifact and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help=".pt artifact written by vllm_capture")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--relative-l2-target", type=float, default=0.01)
    args = parser.parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    report = analyze_capture(capture, relative_l2_target=args.relative_l2_target)
    path = write_report({"experiment": "vllm_kv_cache_calibration", **report}, args.output_dir)
    print(json.dumps(report["recommendation"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
