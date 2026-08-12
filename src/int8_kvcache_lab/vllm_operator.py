"""Direct vLLM v0.6.6 PagedAttention operator bridge for CUDA integration tests."""

from __future__ import annotations

import importlib.metadata
import math

import torch

from .paged_cache import PagedKVCache


def vllm_paged_attention_decode(
    query: torch.Tensor,
    cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Run vLLM v0.6.6's FP PagedAttention decode kernel on a lab cache.

    This bridge is intentionally only an integration-test oracle.  It creates
    vLLM's packed K layout and calls the installed CUDA custom op; the lab's
    dynamic INT8 reference is then compared with this native FP output.
    """
    try:
        version = importlib.metadata.version("vllm")
        from vllm.attention.ops.paged_attn import PagedAttention
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError("install vLLM 0.6.6 to call its PagedAttention operator") from error
    if version != "0.6.6":
        raise RuntimeError(f"vLLM operator bridge requires vLLM 0.6.6, found {version}")
    if not query.is_cuda or not cache.values.is_cuda:
        raise ValueError("vLLM PagedAttention integration requires CUDA tensors")
    if cache.values.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("vLLM FP kernel comparison expects an FP16 or BF16 cache")
    if query.shape[1] % cache.num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    native = torch.empty(
        PagedAttention.get_kv_cache_shape(cache.num_blocks, cache.block_size, cache.num_kv_heads, cache.head_dim),
        dtype=cache.values.dtype,
        device=cache.values.device,
    )
    key_cache, value_cache = PagedAttention.split_kv_cache(native, cache.num_kv_heads, cache.head_dim)
    valid = cache.valid_mask(block_tables, seq_lens)
    physical_slots = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    blocks = torch.div(physical_slots, cache.block_size, rounding_mode="floor")
    offsets = physical_slots.remainder(cache.block_size)
    key = cache.values[blocks, offsets, 0].contiguous()
    value = cache.values[blocks, offsets, 1].contiguous()
    PagedAttention.write_to_paged_cache(
        key,
        value,
        key_cache,
        value_cache,
        physical_slots,
        "auto",
        1.0,
        1.0,
    )
    return PagedAttention.forward_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        int(seq_lens.max().item()),
        "auto",
        cache.num_kv_heads,
        1.0 / math.sqrt(cache.head_dim),
        None,
        1.0,
        1.0,
    )
