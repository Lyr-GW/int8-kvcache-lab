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


def per_head_scale(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return one safe scale per KV head over valid paged tokens.

    Args:
        values: ``[blocks, block_size, kv_heads, head_dim]`` floating tensor.
        valid_mask: Boolean ``[blocks, block_size]`` mask. Invalid page slots
            are excluded from the absmax reduction.
    """
    if values.ndim != 4:
        raise ValueError("values must have shape [blocks, block_size, heads, dim]")
    if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [blocks, block_size]")
    masked = values.float().abs() * valid_mask[..., None, None]
    amax = masked.amax(dim=(0, 1, 3))
    return torch.where(amax > eps, amax / INT8_MAX, torch.ones_like(amax))
