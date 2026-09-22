"""PyTorch paged-attention reference used as the stage-3 accuracy oracle.

The math follows the causal varlen semantics of
``flash_attn_varlen_func(..., block_table=...)``: K/V stay in physical pages,
``block_table`` restores logical order, and the causal mask is aligned to the
bottom-right of the QK matrix. This file does not call a CUDA kernel.
"""

from __future__ import annotations

import math

import torch

from .quantization import logical_kv_scale, per_tensor_scale, quantize_symmetric_int8


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return ``||actual - expected|| / ||expected||`` in FP32."""
    numerator = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-8)
    return float((numerator / denominator).item())


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run FP32 attention for one sequence.

    Args:
        query: ``[query_len, query_heads, head_dim]``.
        key: ``[kv_len, kv_heads, head_dim]``.
        value: ``[kv_len, kv_heads, head_dim]``.
    """
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("query/key/value must be logical per-sequence tensors")
    if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1]:
        raise ValueError("head counts must form a GQA group and share head_dim")
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    logits = torch.einsum("qhd,khd->hqk", query.float(), expanded_key.float()) * softmax_scale
    if causal:
        query_len, kv_len = query.shape[0], key.shape[0]
        query_pos = torch.arange(kv_len - query_len, kv_len, device=query.device)
        key_pos = torch.arange(kv_len, device=query.device)
        allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        if not bool(allowed.any()):
            return torch.zeros_like(query)
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities.masked_fill(~allowed.unsqueeze(0), 0)
    else:
        probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, expanded_value.float()).to(query.dtype)


def _gather_sequence(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = key_cache.shape[1]
    pages = (kv_len + block_size - 1) // block_size
    block_ids = block_table[:pages].to(dtype=torch.long)
    key = key_cache[block_ids].reshape(-1, key_cache.shape[2], key_cache.shape[3])[:kv_len]
    value = value_cache[block_ids].reshape(-1, value_cache.shape[2], value_cache.shape[3])[:kv_len]
    return key, value


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """FP reference for paged varlen attention.

    ``query`` is ``[total_q, query_heads, head_dim]``. Caches are
    ``[blocks, block_size, kv_heads, head_dim]``, which is the layout
    ``flash_attn_varlen_func`` consumes after vLLM splits its packed cache.
    """
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("unexpected query or paged-cache rank")
    if cu_seqlens_q.ndim != 1 or seqused_k.ndim != 1 or block_table.ndim != 2:
        raise ValueError("cu_seqlens_q, seqused_k, and block_table have the wrong rank")
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    outputs = []
    for index, kv_len in enumerate(seqused_k.tolist()):
        start = int(cu_seqlens_q[index])
        stop = int(cu_seqlens_q[index + 1])
        piece = query[start:stop]
        if piece.shape[0] == 0 or int(kv_len) == 0:
            outputs.append(torch.zeros_like(piece))
            continue
        key, value = _gather_sequence(key_cache, value_cache, block_table[index], int(kv_len))
        outputs.append(attend(piece, key, value, softmax_scale=scale, causal=causal))
    return torch.cat(outputs, dim=0) if outputs else query[:0]


def _quantize_logical_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    granularity: str,
    statistic: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key_scale = logical_kv_scale(key, granularity=granularity, statistic=statistic, eps=eps)
    value_scale = logical_kv_scale(value, granularity=granularity, statistic=statistic, eps=eps)
    key_view = key_scale.view(1, -1, 1) if granularity == "per_head" else key_scale.view(1, *key_scale.shape)
    value_view = value_scale.view(1, -1, 1) if granularity == "per_head" else value_scale.view(1, *value_scale.shape)
    return (
        quantize_symmetric_int8(key, key_view),
        quantize_symmetric_int8(value, value_view),
        key_scale,
        value_scale,
    )


def dequantized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    granularity: str,
    statistic: str = "absmax",
    eps: float = 1e-8,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 4A: quantize K/V, dequantize through FP32, then run unchanged FP attention.

    Q is intentionally left in its original dtype. The only new error is the
    INT8 rounding of K and V.
    """
    key_i8, value_i8, key_scale, value_scale = _quantize_logical_kv(
        key, value, granularity=granularity, statistic=statistic, eps=eps
    )
    key_view = key_scale.view(1, -1, 1) if granularity == "per_head" else key_scale.view(1, *key_scale.shape)
    value_view = value_scale.view(1, -1, 1) if granularity == "per_head" else value_scale.view(1, *value_scale.shape)
    key_deq = (key_i8.float() * key_view).to(query.dtype)
    value_deq = (value_i8.float() * value_view).to(query.dtype)
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return attend(query, key_deq, value_deq, softmax_scale=scale, causal=causal)


def int8_simulated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    granularity: str,
    statistic: str = "absmax",
    eps: float = 1e-8,
    softmax_scale: float | None = None,
    quantize_probabilities: bool = False,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 4B: simulate INT8 QK and PV in FP32, without changing the attention control flow.

    Per-channel K scales depend on ``head_dim``, which QK reduces, so they are
    folded into Q and must not be multiplied again. Per-head K scales do not
    depend on ``head_dim``, so they stay outside the dot product. V scales
    depend on ``head_dim`` rather than the token axis, so PV can multiply them
    after the token reduction. Softmax stays in FP32.
    """
    if granularity not in ("per_head", "per_channel"):
        raise ValueError("granularity must be per_head or per_channel")
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    key_i8, value_i8, key_scale, value_scale = _quantize_logical_kv(
        key, value, granularity=granularity, statistic=statistic, eps=eps
    )
    groups = query.shape[1] // key.shape[1]
    if granularity == "per_channel":
        folded_scale = key_scale.repeat_interleave(groups, dim=0)
        folded_query = query.float() * folded_scale.unsqueeze(0)
        query_scale = per_tensor_scale(folded_query.unsqueeze(0), eps)[0]
        query_i8 = quantize_symmetric_int8(folded_query, query_scale)
        key_factor = query_scale
    else:
        query_scale = per_tensor_scale(query.unsqueeze(0), eps)[0]
        query_i8 = quantize_symmetric_int8(query, query_scale)
        # [query_heads, 1, 1] broadcasts across the QK logits [heads, query, tokens].
        key_factor = (query_scale * key_scale.repeat_interleave(groups, dim=0)).view(-1, 1, 1)
    expanded_key = key_i8.float().repeat_interleave(groups, dim=1)
    logits = torch.einsum("qhd,khd->hqk", query_i8.float(), expanded_key) * key_factor * scale
    if causal:
        query_len, kv_len = query.shape[0], key.shape[0]
        query_pos = torch.arange(kv_len - query_len, kv_len, device=query.device)
        key_pos = torch.arange(kv_len, device=query.device)
        allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    if causal:
        probabilities = probabilities.masked_fill(~allowed.unsqueeze(0), 0)
    if quantize_probabilities:
        # Optional second dynamic quant, matching an INT8 tensor-core PV.
        probability_scale = per_tensor_scale(probabilities.unsqueeze(0), eps)[0]
        probabilities = quantize_symmetric_int8(probabilities, probability_scale).float() * probability_scale
    expanded_value = value_i8.float().repeat_interleave(groups, dim=1)
    output = torch.einsum("hqk,khd->qhd", probabilities, expanded_value)
    value_factor = value_scale.repeat_interleave(groups, dim=0)
    value_view = value_factor.view(1, -1, 1) if granularity == "per_head" else value_factor.view(1, *value_factor.shape)
    return (output * value_view).to(query.dtype)


def ref_paged_attn_quantized(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    *,
    implementation: str,
    granularity: str,
    statistic: str = "absmax",
    eps: float = 1e-8,
    softmax_scale: float | None = None,
    causal: bool = True,
    quantize_probabilities: bool = False,
) -> torch.Tensor:
    """Stage 4 varlen path. ``implementation`` is ``dequant`` or ``int8_gemm``."""
    if implementation not in ("dequant", "int8_gemm"):
        raise ValueError("implementation must be dequant or int8_gemm")
    outputs = []
    for index, kv_len in enumerate(seqused_k.tolist()):
        start = int(cu_seqlens_q[index])
        stop = int(cu_seqlens_q[index + 1])
        piece = query[start:stop]
        if piece.shape[0] == 0 or int(kv_len) == 0:
            outputs.append(torch.zeros_like(piece))
            continue
        key, value = _gather_sequence(key_cache, value_cache, block_table[index], int(kv_len))
        if implementation == "dequant":
            outputs.append(
                dequantized_attention(
                    piece,
                    key,
                    value,
                    granularity=granularity,
                    statistic=statistic,
                    eps=eps,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            )
        else:
            outputs.append(
                int8_simulated_attention(
                    piece,
                    key,
                    value,
                    granularity=granularity,
                    statistic=statistic,
                    eps=eps,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    quantize_probabilities=quantize_probabilities,
                )
            )
    return torch.cat(outputs, dim=0) if outputs else query[:0]


def softmax_scale_from_head_dim(head_dim: int) -> float:
    """Return the attention scale used by both the reference and FlashAttention."""
    return 1.0 / math.sqrt(head_dim)
