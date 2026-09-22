import torch

from int8_kvcache_lab.pytorch_ref import ref_paged_attn
from int8_kvcache_lab.replay_capture import replay


def test_causal_varlen_hides_future_keys():
    query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    key_cache = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    value_cache = torch.tensor([[[[3.0, 0.0]], [[0.0, 5.0]]]])
    output = ref_paged_attn(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seqused_k=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        softmax_scale=1.0,
        causal=True,
    )
    torch.testing.assert_close(output[0, 0], torch.tensor([3.0, 0.0]))
    assert output[1, 0, 1] > output[1, 0, 0]


def test_replay_accepts_a_capture_produced_by_the_reference():
    query = torch.randn(3, 2, 4)
    key_cache = torch.randn(2, 2, 1, 4)
    value_cache = torch.randn(2, 2, 1, 4)
    cu = torch.tensor([0, 3], dtype=torch.int32)
    seqused = torch.tensor([3], dtype=torch.int32)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    output = ref_paged_attn(query, key_cache, value_cache, cu, seqused, table, causal=True)
    capture = {
        "steps": [
            {
                "layers": [
                    {
                        "query": query,
                        "key_cache": key_cache,
                        "value_cache": value_cache,
                        "cu_seqlens_q": cu,
                        "seqused_k": seqused,
                        "block_table": table,
                        "output": output,
                        "causal": True,
                        "softmax_scale": 4 ** -0.5,
                    }
                ]
            }
        ]
    }
    report = replay(capture)
    assert report["passes_stage3_gate"]
    assert report["max_relative_l2"] < 1e-5
