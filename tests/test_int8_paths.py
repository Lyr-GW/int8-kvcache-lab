import torch
import pytest

from int8_kvcache_lab import (
    PagedKVCache,
    QuantConfig,
    attention_dequantized,
    attention_int8_simulated,
    paged_attention_reference,
)
from int8_kvcache_lab.pytorch_ref import dequantized_attention, int8_simulated_attention, relative_l2
from int8_kvcache_lab.quantization import logical_query_scale, quantize_symmetric_int8


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


@pytest.mark.parametrize("query_granularity", ["per_head", "per_channel"])
def test_stage5_quantizes_query_by_selected_granularity_for_both_compute_paths(query_granularity):
    query, cache, table, lengths = _case(21)
    gathered_key, gathered_value, _ = cache.gather(table, lengths)
    key = gathered_key[0, : int(lengths[0])]
    value = gathered_value[0, : int(lengths[0])]

    q_scale = logical_query_scale(query, granularity=query_granularity)
    q_view = q_scale.view(1, -1, 1) if query_granularity == "per_head" else q_scale.unsqueeze(0)
    q_i8 = quantize_symmetric_int8(query, q_view)
    q_dequant = (q_i8.float() * q_view).to(query.dtype)
    expected_dequant = dequantized_attention(
        q_dequant, key, value, granularity="per_channel"
    )
    dequant = dequantized_attention(
        query, key, value, granularity="per_channel", query_granularity=query_granularity
    )
    simulated = int8_simulated_attention(
        query, key, value, granularity="per_channel", query_granularity=query_granularity
    )

    assert dequant.shape == query.shape
    assert simulated.shape == query.shape
    torch.testing.assert_close(dequant, expected_dequant)
    assert torch.isfinite(dequant).all()
    assert torch.isfinite(simulated).all()
