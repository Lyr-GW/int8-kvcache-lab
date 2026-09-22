"""Symmetric INT8 quantization primitives."""

from __future__ import annotations

import torch


INT8_MAX = 127.0


def quantize_symmetric_int8(
    values: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Quantize ``values`` using an already-computed positive scale."""
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if torch.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    return torch.round(values.float() / scale.float()).clamp(-127, 127).to(torch.int8)


def dequantize_int8(values: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize an INT8 tensor and cast it to ``dtype``."""
    if values.dtype != torch.int8:
        raise TypeError("values must have dtype torch.int8")
    return (values.float() * scale.float()).to(dtype)


def per_tensor_scale(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return one safe symmetric scale per leading batch element.

    ``values`` must have shape ``[batch, ...]``. A zero-valued slice receives
    scale ``1`` so it round-trips exactly without division by zero.
    """
    if values.ndim < 1:
        raise ValueError("values must include a batch dimension")
    amax = values.float().abs().reshape(values.shape[0], -1).amax(dim=1)
    return torch.where(amax > eps, amax / INT8_MAX, torch.ones_like(amax))


def _positive_scale(statistic: torch.Tensor, eps: float) -> torch.Tensor:
    """Convert a magnitude statistic into a strictly positive INT8 scale."""
    return torch.where(statistic > eps, statistic / INT8_MAX, torch.ones_like(statistic))


def _magnitude(values: torch.Tensor, statistic: str) -> torch.Tensor:
    """Reduce ``values`` over the token axis.

    ``values`` is ``[tokens, ...]``. ``absmax`` uses the maximum absolute
    value. ``p999`` uses the 99.9th percentile, which clips a single outlier.
    """
    absolute = values.float().abs()
    if statistic == "absmax":
        return absolute.amax(dim=0)
    if statistic == "p999":
        if absolute.shape[0] == 1:
            return absolute[0]
        return torch.quantile(absolute, 0.999, dim=0)
    raise ValueError("statistic must be absmax or p999")


def per_head_scale(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
    *,
    statistic: str = "absmax",
) -> torch.Tensor:
    """Return one safe scale per KV head over valid paged tokens.

    Args:
        values: ``[blocks, block_size, kv_heads, head_dim]`` floating tensor.
        valid_mask: Boolean ``[blocks, block_size]`` mask. Invalid page slots
            are excluded from the reduction. The result shape is ``[kv_heads]``.
    """
    if values.ndim != 4:
        raise ValueError("values must have shape [blocks, block_size, heads, dim]")
    if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [blocks, block_size]")
    selected = values.reshape(-1, values.shape[2], values.shape[3])[valid_mask.reshape(-1)]
    if selected.numel() == 0:
        return torch.ones(values.shape[2], dtype=torch.float32, device=values.device)
    return _positive_scale(_magnitude(selected, statistic).amax(dim=-1), eps)


def per_channel_scale(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
    *,
    statistic: str = "absmax",
) -> torch.Tensor:
    """Return one scale per ``(kv_head, head_dim)`` over valid tokens.

    The result shape is ``[kv_heads, head_dim]``. This scale depends on the
    axis that QK reduces, so an INT8 GEMM cannot factor it out of the dot
    product. Fold it into Q instead.
    """
    if values.ndim != 4:
        raise ValueError("values must have shape [blocks, block_size, heads, dim]")
    if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [blocks, block_size]")
    selected = values.reshape(-1, values.shape[2], values.shape[3])[valid_mask.reshape(-1)]
    if selected.numel() == 0:
        return torch.ones(values.shape[2:], dtype=torch.float32, device=values.device)
    return _positive_scale(_magnitude(selected, statistic), eps)


def logical_kv_scale(
    values: torch.Tensor,
    *,
    granularity: str,
    statistic: str = "absmax",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale a gathered logical KV tensor of shape ``[tokens, kv_heads, dim]``."""
    if values.ndim != 3:
        raise ValueError("logical KV must have shape [tokens, kv_heads, head_dim]")
    if granularity == "per_head":
        return _positive_scale(_magnitude(values, statistic).amax(dim=-1), eps)
    if granularity == "per_channel":
        return _positive_scale(_magnitude(values, statistic), eps)
    raise ValueError("granularity must be per_head or per_channel")
