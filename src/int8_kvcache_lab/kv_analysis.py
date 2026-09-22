"""KV-cache capture analysis and quantization-granularity calibration.

The calibration data is deliberately kept separate from the runtime path.
It reports reconstruction error for several scale layouts and recommends
``per_channel`` when channels inside a head differ sharply, otherwise
``per_head``. The recommendation is an input to the stage-4/5 experiments;
it does not rewrite ``QuantConfig`` by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from .quantization import INT8_MAX
from .reporting import to_jsonable, write_report


_CANDIDATES = ("per_tensor", "per_head", "per_token", "per_channel")
# A head whose largest channel is at least this many times its smallest
# channel needs a scale that follows head_dim. 8x is a starting threshold,
# not a property of the model.
_WITHIN_HEAD_CHANNEL_RATIO = 8.0
# ``layers`` is the analyzed matrix; ``steps`` is the raw per-step tape that
# stage 3 replays. Both are far too large for a JSON report, so the report
# keeps only the scalar provenance of the capture.
_TENSOR_PAYLOAD_KEYS = ("layers", "steps")


def _capture_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    """Reduce a capture's provenance to scalars a JSON report can carry.

    The capture dict holds tensors (``block_tables``) and the whole step tape
    under ``steps``; embedding either one makes the report unencodable and
    enormous. Tensors outside those keys are summarized by shape and dtype.
    """
    metadata: dict[str, Any] = {}
    for key, value in capture.items():
        if key in _TENSOR_PAYLOAD_KEYS:
            continue
        if isinstance(value, torch.Tensor):
            metadata[key] = {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
        else:
            metadata[key] = value
    steps = capture.get("steps")
    if isinstance(steps, list):
        metadata["capture_step_count"] = len(steps)
    return metadata


def _safe_scale(absmax: torch.Tensor, eps: float) -> torch.Tensor:
    """Turn an absmax tensor into non-zero symmetric INT8 scales."""
    return torch.where(absmax > eps, absmax / INT8_MAX, torch.ones_like(absmax))


def _reduce_magnitude(absolute: torch.Tensor, dims: tuple[int, ...], statistic: str) -> torch.Tensor:
    """Reduce ``absolute`` over ``dims`` by absmax or the 99.9th percentile."""
    if statistic == "absmax":
        return absolute.amax(dim=dims)
    keep = [axis for axis in range(absolute.ndim) if axis not in dims]
    moved = absolute.permute(*dims, *keep)
    flat = moved.reshape(-1, *moved.shape[len(dims) :])
    if flat.shape[0] == 1:
        return flat[0]
    return torch.quantile(flat, 0.999, dim=0)


def _scale_for(
    values: torch.Tensor,
    granularity: str,
    eps: float,
    statistic: str = "absmax",
) -> torch.Tensor:
    """Return one scale per requested KV-cache quantization group.

    ``values`` has canonical logical shape ``[tokens, 2, kv_heads, head_dim]``.
    K and V stay separate. ``per_channel`` is one scale per
    ``(K or V, kv_head, head_dim)`` reduced over tokens. It is not a
    per-token scale.
    """
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("values must have shape [tokens, 2, kv_heads, head_dim]")
    absolute = values.float().abs()
    if granularity == "per_tensor":
        return _safe_scale(_reduce_magnitude(absolute, (0, 2, 3), statistic), eps)[None, :, None, None]
    if granularity == "per_head":
        return _safe_scale(_reduce_magnitude(absolute, (0, 3), statistic), eps)[None, :, :, None]
    if granularity == "per_token":
        return _safe_scale(_reduce_magnitude(absolute, (2, 3), statistic), eps)[:, :, None, None]
    if granularity == "per_channel":
        return _safe_scale(_reduce_magnitude(absolute, (0,), statistic), eps)[None]
    raise ValueError(f"unsupported granularity: {granularity}")


def _spread(layers: list[torch.Tensor]) -> dict[str, float]:
    """Compare channel spread inside a head with spread across heads."""
    within: list[torch.Tensor] = []
    across: list[torch.Tensor] = []
    for layer in layers:
        channel_absmax = layer.float().abs().amax(dim=0)
        peak = channel_absmax.amax(dim=-1).clamp_min(1e-8)
        floor = channel_absmax.amin(dim=-1).clamp_min(1e-8)
        within.append((peak / floor).reshape(-1))
        head_peak = channel_absmax.amax(dim=-1)
        across.append((head_peak.amax(dim=-1) / head_peak.amin(dim=-1).clamp_min(1e-8)).reshape(-1))
    within_all = torch.cat(within)
    across_all = torch.cat(across)
    return {
        "within_head_max_ratio": float(within_all.max().item()),
        "within_head_median_ratio": float(within_all.median().item()),
        "across_head_median_ratio": float(across_all.median().item()),
        "within_head_threshold": _WITHIN_HEAD_CHANNEL_RATIO,
    }


def _error_for(
    values: torch.Tensor,
    granularity: str,
    eps: float,
    statistic: str = "absmax",
) -> dict[str, float | int]:
    scale = _scale_for(values, granularity, eps, statistic)
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
    p999_totals = {name: {"squared_error": 0.0, "squared_signal": 0.0} for name in _CANDIDATES}
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
            p999_metrics = _error_for(layer, name, eps, "p999")
            for metric in ("squared_error", "squared_signal"):
                p999_totals[name][metric] += p999_metrics[metric]
        layer_summaries.append({"layer_index": index, "tokens": int(layer.shape[0]), "candidate_errors": candidate_errors})

    candidates: dict[str, Any] = {}
    for name in _CANDIDATES:
        total = totals[name]
        signal = max(float(total["squared_signal"]), 1e-20)
        p999_signal = max(float(p999_totals[name]["squared_signal"]), 1e-20)
        candidates[name] = {
            "relative_l2": math.sqrt(float(total["squared_error"]) / signal),
            "p999_relative_l2": math.sqrt(float(p999_totals[name]["squared_error"]) / p999_signal),
            "max_abs_error": total["max_abs_error"],
            "scale_count": int(total["scale_count"]),
            "scale_bytes_fp32": int(total["scale_count"]) * 4,
        }
    spread = _spread(layers)
    recommendation = (
        "per_channel" if spread["within_head_max_ratio"] >= _WITHIN_HEAD_CHANNEL_RATIO else "per_head"
    )
    flat = torch.cat([layer.float().abs().reshape(-1) for layer in layers])
    absmax = flat.amax()
    percentile = flat[0] if flat.numel() == 1 else torch.quantile(flat, 0.999)
    outlier_ratio = float((absmax / percentile.clamp_min(1e-8)).item())
    return {
        "capture": _capture_metadata(capture),
        "layer_count": len(layers),
        "kv_tokens_analyzed": int(sum(layer.shape[0] for layer in layers)),
        "relative_l2_target": relative_l2_target,
        "spread": spread,
        "outlier_ratio_absmax_over_p999": outlier_ratio,
        "recommendation": {
            "kv_granularity": recommendation,
            "scale_statistic": "p999" if outlier_ratio >= 4.0 else "absmax",
            "meets_target": candidates[recommendation]["relative_l2"] <= relative_l2_target,
            "rationale": (
                "channels inside a head differ by at least the configured ratio, so the scale must follow head_dim"
                if recommendation == "per_channel"
                else "channels inside a head are comparable, so one scale per KV head is enough"
            ),
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
    # Print before writing: the recommendation is the result of a long GPU run,
    # so a reporting failure must not be the only thing the run leaves behind.
    # The print goes through to_jsonable too, or an unencodable value would
    # abort here and cost the run the write it was moved ahead of.
    print(json.dumps(to_jsonable(report["recommendation"]), indent=2))
    print(
        "relative_l2 by granularity: "
        + ", ".join(
            f"{name}={report['candidates'][name]['relative_l2']:.5f}" for name in _CANDIDATES
        )
    )
    path = write_report({"experiment": "vllm_kv_cache_calibration", **report}, args.output_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
