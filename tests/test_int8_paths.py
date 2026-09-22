import torch

from int8_kvcache_lab import (
    PagedKVCache,
    QuantConfig,
    attention_dequantized,
    attention_int8_simulated,
    paged_attention_reference,
)
from int8_kvcache_lab.pytorch_ref import relative_l2


def _case(seed: int, scale_first_channel: float = 1.0):
    torch.manual_seed(seed)
    block_size, kv_heads, query_heads, dim, length = 16, 2, 4, 16, 17
    pages = 2
    cache = PagedKVCache.empty(pages + 1, block_size, kv_heads, dim, dtype=torch.float32, device="cpu")
    table = torch.tensor([[2, 0]], dtype=torch.int64)
    slots, keys, values = [], [], []
    for token in range(length):
        block = int(table[0, token // block_size])
        slots.append(block * block_size + token % block_size)
        key = torch.randn(kv_heads, dim)
        key[:, 0] *= scale_first_channel
        keys.append(key)
        values.append(torch.randn(kv_heads, dim))
    cache.write(torch.stack(keys), torch.stack(values), torch.tensor(slots))
    query = torch.randn(1, query_heads, dim)
    return query, cache, table, torch.tensor([length])


def test_dequant_and_int8_gemm_stay_near_the_fp_reference():
    query, cache, table, lengths = _case(4)
    reference = paged_attention_reference(query, cache, table, lengths)
    for granularity in ("per_head", "per_channel"):
        config = QuantConfig(block_size=16, kv_granularity=granularity)
        dequant, _ = attention_dequantized(query, cache, table, lengths, config)
        simulated, _ = attention_int8_simulated(query, cache, table, lengths, config)
        assert relative_l2(dequant, reference) <= 0.05
        assert relative_l2(simulated, dequant) <= 0.10


def test_per_channel_int8_gemm_survives_a_large_channel_imbalance():
    query, cache, table, lengths = _case(5, scale_first_channel=40.0)
    reference = paged_attention_reference(query, cache, table, lengths)
    config = QuantConfig(block_size=16, kv_granularity="per_channel")
    dequant, _ = attention_dequantized(query, cache, table, lengths, config)
    simulated, _ = attention_int8_simulated(query, cache, table, lengths, config)
    assert relative_l2(dequant, reference) <= 0.05
    assert relative_l2(simulated, dequant) <= 0.10
