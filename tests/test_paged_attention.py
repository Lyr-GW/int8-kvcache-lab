import pytest
import torch

from int8_kvcache_lab import PagedKVCache, QuantConfig, paged_attention_dynamic_int8, paged_attention_reference


def _make_case(batch: int, length: int, block_size: int, dtype: torch.dtype):
    torch.manual_seed(7)
    kv_heads, q_heads, dim = 2, 8, 16
    pages = (length + block_size - 1) // block_size
    cache = PagedKVCache.empty(batch * pages + 2, block_size, kv_heads, dim, dtype=dtype, device="cpu")
    table = torch.full((batch, pages), -1, dtype=torch.int64)
    slots, keys, values = [], [], []
    for request in range(batch):
        ids = torch.arange(request * pages, (request + 1) * pages)
        table[request] = ids
        for token in range(length):
            slots.append(int(ids[token // block_size]) * block_size + token % block_size)
            keys.append(torch.randn(kv_heads, dim))
            values.append(torch.randn(kv_heads, dim))
    cache.write(torch.stack(keys).to(dtype), torch.stack(values).to(dtype), torch.tensor(slots))
    return torch.randn(batch, q_heads, dim, dtype=dtype), cache, table, torch.full((batch,), length, dtype=torch.int64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [1, 15, 16, 17, 257])
@pytest.mark.parametrize("block_size", [16, 32])
def test_dynamic_paged_attention_matches_fp_reference(dtype, batch, length, block_size):
    query, cache, table, seq_lens = _make_case(batch, length, block_size, dtype)
    expected = paged_attention_reference(query, cache, table, seq_lens)
    actual, stats = paged_attention_dynamic_int8(query, cache, table, seq_lens, QuantConfig(block_size=block_size))
    relative_l2 = (actual - expected).float().norm() / expected.float().norm().clamp_min(1e-8)
    assert relative_l2.item() <= 0.05
    assert stats["int8_cache_bytes"] == cache.values.numel()
    assert stats["fp_cache_bytes"] == cache.values.numel() * cache.values.element_size()
    if dtype in (torch.float16, torch.bfloat16):
        assert stats["int8_cache_bytes"] * 2 == stats["fp_cache_bytes"]
    assert stats["int8_cache_and_scale_bytes"] > stats["int8_cache_bytes"]


def test_invalid_referenced_block_is_rejected():
    cache = PagedKVCache.empty(1, 16, 1, 8, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="invalid referenced block"):
        cache.gather(torch.tensor([[1]]), torch.tensor([1]))
