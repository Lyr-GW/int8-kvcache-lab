import torch

from int8_kvcache_lab.quantization import (
    dequantize_int8,
    logical_kv_scale,
    per_channel_scale,
    per_head_scale,
    per_tensor_scale,
    quantize_symmetric_int8,
)


def test_per_tensor_round_trip_and_zero_slice():
    values = torch.tensor([[[0.0, 0.0]], [[-2.0, 1.0]]])
    scale = per_tensor_scale(values)
    actual = dequantize_int8(quantize_symmetric_int8(values, scale[:, None, None]), scale[:, None, None], values.dtype)
    assert scale.tolist()[0] == 1.0
    torch.testing.assert_close(actual, values, atol=2e-2, rtol=2e-2)


def test_per_head_ignores_invalid_outlier():
    values = torch.tensor([[[[1.0], [2.0]], [[999.0], [999.0]]]])
    mask = torch.tensor([[True, False]])
    scale = per_head_scale(values, mask)
    torch.testing.assert_close(scale, torch.tensor([1 / 127, 2 / 127]))


def test_per_channel_scale_follows_head_dim_and_ignores_invalid_slots():
    values = torch.zeros(1, 2, 1, 2)
    values[0, 0, 0, 0] = 127.0
    values[0, 0, 0, 1] = 1.0
    values[0, 1, 0, :] = 1000.0
    mask = torch.tensor([[True, False]])
    scale = per_channel_scale(values, mask)
    torch.testing.assert_close(scale, torch.tensor([[1.0, 1.0 / 127]]))


def test_per_token_head_and_per_channel_keep_different_axes():
    values = torch.tensor([[[1.0, 127.0]], [[64.0, 2.0]]])
    token_head = logical_kv_scale(values, granularity="per_token_head")
    channel = logical_kv_scale(values, granularity="per_channel")
    torch.testing.assert_close(token_head, torch.tensor([[1.0], [64.0 / 127]]))
    torch.testing.assert_close(channel, torch.tensor([[64.0 / 127, 1.0]]))
