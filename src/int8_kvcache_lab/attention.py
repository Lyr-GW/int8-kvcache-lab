"""FP and dynamically quantized paged decode attention references."""

from __future__ import annotations

import math

import torch

from .config import QuantConfig
from .paged_cache import PagedKVCache
from .quantization import per_head_scale, per_tensor_scale, quantize_symmetric_int8


def _validate_query(query: torch.Tensor, cache: PagedKVCache, block_tables: torch.Tensor) -> None:
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if query.shape[0] != block_tables.shape[0] or query.shape[2] != cache.head_dim:
        raise ValueError("query batch/head_dim must match cache metadata")
    if query.shape[1] % cache.num_kv_heads:
        raise ValueError("query_heads must be divisible by num_kv_heads for GQA")


def _attention_from_logical(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch, query_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    groups = query_heads // kv_heads
    expanded_key = key.repeat_interleave(groups, dim=2)
    expanded_value = value.repeat_interleave(groups, dim=2)
    logits = torch.einsum("bhd,bthd->bht", query.float(), expanded_key.float()) / math.sqrt(head_dim)
    logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum("bht,bthd->bhd", probabilities, expanded_value.float()).to(query.dtype)


def paged_attention_reference(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute decode attention from FP paged cache using FP32 accumulation."""
    _validate_query(query, cache, block_tables)
    key, value, valid = cache.gather(block_tables, seq_lens)
    if torch.any(seq_lens == 0):
        raise ValueError("decode attention requires at least one KV token per request")
    return _attention_from_logical(query, key, value, valid)


def paged_attention_dynamic_int8(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    config: QuantConfig = QuantConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Dynamically quantize Q/K/V then perform dequantized decode attention.

    This intentionally models the expensive correctness-first implementation:
    the full FP cache remains resident and each call creates a temporary INT8
    cache. It is not a production fast path.
    """
    if config.block_size != cache.block_size:
        raise ValueError("QuantConfig.block_size must equal the cache block size")
    _validate_query(query, cache, block_tables)
    if torch.any(seq_lens == 0):
        raise ValueError("decode attention requires at least one KV token per request")

    valid_physical = cache.valid_mask(block_tables, seq_lens)
    key_physical = cache.values[:, :, 0]
    value_physical = cache.values[:, :, 1]
    key_scale = per_head_scale(key_physical, valid_physical, config.eps)
    value_scale = per_head_scale(value_physical, valid_physical, config.eps)
    q_scale = per_tensor_scale(query, config.eps)

    q_int8 = quantize_symmetric_int8(query, q_scale[:, None, None])
    key_int8 = quantize_symmetric_int8(key_physical, key_scale[None, None, :, None])
    value_int8 = quantize_symmetric_int8(value_physical, value_scale[None, None, :, None])
    quantized = PagedKVCache(torch.stack((key_int8, value_int8), dim=2))
    key_i8, value_i8, logical_valid = quantized.gather(block_tables, seq_lens)

    query_heads = query.shape[1]
    groups = query_heads // cache.num_kv_heads
    q_float = q_int8.float() * q_scale[:, None, None]
    key_float = key_i8.float() * key_scale[None, None, :, None]
    value_float = value_i8.float() * value_scale[None, None, :, None]
    expanded_key = key_float.repeat_interleave(groups, dim=2)
    expanded_value = value_float.repeat_interleave(groups, dim=2)
    logits = torch.einsum("bhd,bthd->bht", q_float, expanded_key) / math.sqrt(cache.head_dim)
    logits = logits.masked_fill(~logical_valid[:, None, :], float("-inf"))
    output = torch.einsum("bht,bthd->bhd", torch.softmax(logits, dim=-1), expanded_value).to(query.dtype)
    return output, {
        "q_scale": q_scale,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "fp_cache_bytes": cache.fp_bytes,
        "int8_cache_bytes": cache.int8_bytes,
        "scale_bytes": (q_scale.numel() + key_scale.numel() + value_scale.numel()) * q_scale.element_size(),
        "int8_cache_and_scale_bytes": cache.int8_bytes
        + (q_scale.numel() + key_scale.numel() + value_scale.numel()) * q_scale.element_size(),
    }
