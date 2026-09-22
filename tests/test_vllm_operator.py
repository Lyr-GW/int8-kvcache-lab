"""CUDA parity test for vLLM's FlashAttention varlen paged kernel.

The test runs only when a current vLLM build exposing
``flash_attn_varlen_func`` is installed. vLLM 0.6.6 is not accepted.
"""

import pytest
import torch

from int8_kvcache_lab import PagedKVCache, QuantConfig, attention_dequantized, paged_attention_reference
from int8_kvcache_lab.pytorch_ref import relative_l2


vllm = pytest.importorskip("vllm", reason="vLLM is installed only in the capture Colab runtime")
if not torch.cuda.is_available():
    pytest.skip("FlashAttention oracle requires CUDA", allow_module_level=True)
pytest.importorskip("vllm.v1.attention.backends.flash_attn", reason="requires vLLM V1 FlashAttention, tested at 0.29.0")
pytest.importorskip("vllm.vllm_flash_attn", reason="requires the vLLM flash-attn package")

from int8_kvcache_lab.vllm_operator import vllm_paged_attention_decode


def test_flash_attn_varlen_matches_fp_reference_and_dequant_int8():
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
    dequant, _ = attention_dequantized(query, cache, table, lengths, QuantConfig(block_size=block_size, kv_granularity="per_head"))
    assert relative_l2(native, fp_reference) <= 0.02
    assert relative_l2(dequant, native) <= 0.05
