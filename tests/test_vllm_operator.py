"""Opt-in CUDA parity test for the real vLLM v0.6.6 PagedAttention operator."""

import importlib.metadata

import pytest
import torch

from int8_kvcache_lab import PagedKVCache, QuantConfig, paged_attention_dynamic_int8, paged_attention_reference


vllm = pytest.importorskip("vllm", reason="vLLM integration is installed only in the capture Colab runtime")
if not torch.cuda.is_available():
    pytest.skip("vLLM PagedAttention integration requires CUDA", allow_module_level=True)
if importlib.metadata.version("vllm") != "0.6.6":
    pytest.skip("test is pinned to the vLLM 0.6.6 cache/operator contract", allow_module_level=True)

from int8_kvcache_lab.vllm_operator import vllm_paged_attention_decode


def test_vllm_paged_attention_matches_fp_reference_then_dynamic_int8():
    torch.manual_seed(11)
    device = "cuda"
    block_size, kv_heads, query_heads, head_dim = 16, 2, 4, 64
    lengths = torch.tensor([15, 17], dtype=torch.int32, device=device)
    table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device=device)
    cache = PagedKVCache.empty(4, block_size, kv_heads, head_dim, dtype=torch.float16, device=device)
    slots, keys, values = [], [], []
    for request, length in enumerate(lengths.tolist()):
        for token in range(length):
            block = table[request, token // block_size].item()
            slots.append(block * block_size + token % block_size)
            keys.append(torch.randn(kv_heads, head_dim, device=device))
            values.append(torch.randn(kv_heads, head_dim, device=device))
    cache.write(torch.stack(keys).half(), torch.stack(values).half(), torch.tensor(slots, device=device))
    query = torch.randn(2, query_heads, head_dim, dtype=torch.float16, device=device)
    native = vllm_paged_attention_decode(query, cache, table, lengths)
    fp_reference = paged_attention_reference(query, cache, table, lengths)
    dynamic, _ = paged_attention_dynamic_int8(query, cache, table, lengths, QuantConfig(block_size=block_size))
    native_error = (native - fp_reference).float().norm() / fp_reference.float().norm().clamp_min(1e-8)
    int8_error = (dynamic - native).float().norm() / native.float().norm().clamp_min(1e-8)
    assert native_error.item() <= 0.02
    assert int8_error.item() <= 0.05
