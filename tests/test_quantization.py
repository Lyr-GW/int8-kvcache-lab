import torch

from int8_kvcache_lab.quantization import dequantize_int8, per_head_scale, per_tensor_scale, quantize_symmetric_int8


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
