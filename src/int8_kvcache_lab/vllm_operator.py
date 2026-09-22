"""Call the installed vLLM FlashAttention varlen kernel on a lab cache.

This is the stage-3 kernel oracle. It targets vLLM 0.29's
``flash_attn_varlen_func`` with a paged ``block_table``, not the v0.6.6
``PagedAttention.forward_decode`` path.
"""

from __future__ import annotations

import math

import torch

from .paged_cache import PagedKVCache
from .vllm_runtime import load_flash_attention


def vllm_paged_attention_decode(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Run one decode step through ``flash_attn_varlen_func``.

    ``query`` is ``[batch, query_heads, head_dim]``. The lab cache is split
    into the FlashAttention page layout
    ``[blocks, block_size, kv_heads, head_dim]``.
    """
    _, flash_attn_varlen_func = load_flash_attention()
    if not query.is_cuda or not cache.values.is_cuda:
        raise ValueError("the FlashAttention oracle requires CUDA tensors")
    if cache.values.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the FlashAttention oracle expects an FP16 or BF16 cache")
    if query.dtype != cache.values.dtype:
        raise TypeError("query dtype must match the cache dtype")
    if query.shape[1] % cache.num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    if query.shape[0] != block_tables.shape[0] or query.shape[0] != seq_lens.shape[0]:
        raise ValueError("query, block_tables, and seq_lens must share the batch size")

    key_cache = cache.values[:, :, 0].contiguous()
    value_cache = cache.values[:, :, 1].contiguous()
    batch = query.shape[0]
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    seqused_k = seq_lens.to(device=query.device, dtype=torch.int32)
    block_table = block_tables.to(device=query.device, dtype=torch.int32)
    return flash_attn_varlen_func(
        q=query.contiguous(),
        k=key_cache,
        v=value_cache,
        max_seqlen_q=1,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=int(seqused_k.max().item()),
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(cache.head_dim),
        causal=True,
        block_table=block_table,
    )
