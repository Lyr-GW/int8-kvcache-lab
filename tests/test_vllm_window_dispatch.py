"""Verify that the vLLM interception recognizes unbounded window forms."""

import pytest
import torch

from int8_kvcache_lab import pytorch_ref
from int8_kvcache_lab.vllm_runtime import replace_varlen_with_pytorch


@pytest.mark.parametrize(
    ("window", "intercepted"),
    [(None, True), ((-1, -1), True), ([-1, -1], True), ([4, 0], False)],
)
def test_window_dispatch(monkeypatch, window, intercepted):
    flash = pytest.importorskip("vllm.v1.attention.backends.flash_attn")
    query = torch.ones(1, 1, 2)
    monkeypatch.setattr(flash, "flash_attn_varlen_func", lambda **kwargs: query * 17)
    monkeypatch.setattr(pytorch_ref, "ref_paged_attn_quantized", lambda **kwargs: query * 3)
    counters = {}
    with replace_varlen_with_pytorch("dequant", call_stats=counters):
        result = flash.flash_attn_varlen_func(
            q=query, k=query, v=query, cu_seqlens_q=torch.tensor([0, 1]),
            seqused_k=torch.tensor([1]), block_table=torch.tensor([[0]]),
            causal=True, window_size=window,
        )
    assert result[0, 0, 0].item() == (3 if intercepted else 17)
    assert counters.get("quantized_pytorch_calls", 0) == int(intercepted)
    assert counters.get("native_fallback_calls", 0) == int(not intercepted)
