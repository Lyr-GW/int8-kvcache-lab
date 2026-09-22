"""Helpers for the vLLM 0.29 V1 FlashAttention path.

The tested runtime is vLLM 0.29.0. Paged KV attention is
``FlashAttentionImpl.forward`` calling ``flash_attn_varlen_func``. On Ampere
that resolves to ``torch.ops._vllm_fa2_C.varlen_fwd``. Hopper uses FA3 and
Blackwell uses FA4. This module does not support the removed v0.6.6
``PagedAttention.forward_decode`` operator.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

TESTED_VLLM_VERSION = "0.29.0"

_OPERATOR_BY_FA_VERSION = {
    2: "torch.ops._vllm_fa2_C.varlen_fwd",
    3: "torch.ops._vllm_fa3_C.fwd",
    4: "vllm.vllm_flash_attn.cute._flash_attn_fwd",
}


def installed_version() -> str:
    """Return the installed vLLM version, or raise when the package is absent."""
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "vLLM is not installed. In a fresh Colab GPU runtime run "
            "`bash scripts/install_vllm_capture.sh`. The tested release is "
            f"vLLM {TESTED_VLLM_VERSION}."
        ) from error


def load_flash_attention() -> tuple[Any, Any]:
    """Import V1 FlashAttention and the varlen entry point."""
    version = installed_version()
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError as error:
        raise RuntimeError(
            f"vLLM {version} has no V1 FlashAttention varlen operator. "
            f"Install vLLM {TESTED_VLLM_VERSION}. The v0.6.6 PagedAttention "
            "decode kernel is not the oracle for this lab."
        ) from error
    return FlashAttentionImpl, flash_attn_varlen_func


def describe_call(fa_version: int | None, module_file: str | None = None) -> dict[str, Any]:
    """Name the Python call and the CUDA operator selected for this GPU."""
    operator = _OPERATOR_BY_FA_VERSION.get(fa_version, "flash_attn_varlen_func")
    return {
        "vllm_version": installed_version(),
        "tested_vllm_version": TESTED_VLLM_VERSION,
        "backend": "FLASH_ATTN",
        "python_file": module_file or "vllm/v1/attention/backends/flash_attn.py",
        "python_call": "FlashAttentionImpl.forward -> flash_attn_varlen_func",
        "fa_version": fa_version,
        "operator": operator,
        "kv_cache_layout": "[num_blocks, num_kv_heads, block_size, 2 * head_size]",
        "paged_kv_layout": "[num_blocks, block_size, num_kv_heads, head_size]",
        "rank": int(os.environ.get("RANK", "0")),
    }


def format_call_site(description: dict[str, Any]) -> str:
    """One line suitable for the stage-1 acceptance log."""
    return (
        f"Paged KV attention: {description['python_call']} "
        f"fa_version={description['fa_version']} operator={description['operator']} "
        f"vllm={description['vllm_version']} rank={description['rank']}"
    )


def make_llm(model: str, *, dtype: str, max_model_len: int, gpu_memory_utilization: float = 0.75) -> Any:
    """Construct an eager, single-GPU engine with prefix caching and chunked prefill off."""
    from vllm import LLM

    return LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_config={"backend": "FLASH_ATTN"},
    )


def split_packed_kv_cache(kv_cache: Any, head_size: int) -> tuple[Any, Any]:
    """Split vLLM 0.29's packed cache into FlashAttention K and V pages."""
    import torch

    if kv_cache.ndim != 4 or kv_cache.shape[-1] != 2 * head_size:
        raise ValueError(
            "expected vLLM 0.29 KV cache [num_blocks, num_kv_heads, block_size, 2 * head_size], "
            f"got {tuple(kv_cache.shape)} for head_size={head_size}"
        )
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    return key_cache.contiguous(), value_cache.contiguous()


def pack_referenced_pages(
    key_cache: Any,
    value_cache: Any,
    block_table: Any,
    seqused_k: Any,
) -> tuple[Any, Any, Any]:
    """Copy only the pages a step actually addresses, and remap the block table."""
    import torch

    block_size = key_cache.shape[1]
    ordered: list[int] = []
    seen: set[int] = set()
    for row, length in enumerate(seqused_k.tolist()):
        pages = (int(length) + block_size - 1) // block_size
        for block_id in block_table[row, :pages].tolist():
            block_id = int(block_id)
            if block_id not in seen:
                seen.add(block_id)
                ordered.append(block_id)
    if not ordered:
        empty = key_cache[:0].detach().cpu()
        return empty, value_cache[:0].detach().cpu(), block_table.detach().cpu()
    remap = {old: new for new, old in enumerate(ordered)}
    local = torch.zeros_like(block_table, device="cpu")
    for row, length in enumerate(seqused_k.tolist()):
        pages = (int(length) + block_size - 1) // block_size
        local[row, :pages] = torch.tensor(
            [remap[int(block_id)] for block_id in block_table[row, :pages].tolist()],
            dtype=block_table.dtype,
        )
    index = torch.tensor(ordered, device=key_cache.device, dtype=torch.long)
    return key_cache.index_select(0, index).detach().cpu(), value_cache.index_select(0, index).detach().cpu(), local


@contextmanager
def observe_flash_forward(callback: Callable[..., None]) -> Iterator[None]:
    """Call ``callback`` after every eager ``FlashAttentionImpl.forward``."""
    flash_impl, _ = load_flash_attention()
    original = flash_impl.forward

    def wrapped(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        result = original(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        if attn_metadata is not None:
            callback(self, query, key, value, kv_cache, attn_metadata, output)
        return result

    flash_impl.forward = wrapped
    try:
        yield
    finally:
        flash_impl.forward = original


@contextmanager
def replace_varlen_with_pytorch(
    implementation: str,
    *,
    granularity: str = "per_channel",
    statistic: str = "absmax",
) -> Iterator[None]:
    """Replace the FlashAttention varlen call with the stage-3/4 PyTorch path.

    Metadata setup and the KV write stay in vLLM. Only the attention math
    changes. Calls without a block table, or with a non-boolean causal flag,
    keep the native kernel.
    """
    import vllm.v1.attention.backends.flash_attn as flash_module

    from .pytorch_ref import ref_paged_attn, ref_paged_attn_quantized

    if implementation not in ("fp", "dequant", "int8_gemm"):
        raise ValueError("implementation must be fp, dequant, or int8_gemm")
    original = flash_module.flash_attn_varlen_func

    def wrapped(**kwargs):
        causal = kwargs.get("causal")
        block_table = kwargs.get("block_table")
        window = kwargs.get("window_size")
        uses_window = window not in (None, (-1, -1))
        if block_table is None or not isinstance(causal, bool) or uses_window:
            return original(**kwargs)
        common = dict(
            query=kwargs["q"],
            key_cache=kwargs["k"],
            value_cache=kwargs["v"],
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            seqused_k=kwargs["seqused_k"],
            block_table=block_table,
            softmax_scale=kwargs.get("softmax_scale"),
            causal=causal,
        )
        if implementation == "fp":
            computed = ref_paged_attn(**common)
        else:
            computed = ref_paged_attn_quantized(
                **common,
                implementation=implementation,
                granularity=granularity,
                statistic=statistic,
            )
        destination = kwargs.get("out")
        if destination is not None:
            destination.copy_(computed.to(dtype=destination.dtype))
            return destination
        return computed

    flash_module.flash_attn_varlen_func = wrapped
    try:
        yield
    finally:
        flash_module.flash_attn_varlen_func = original
