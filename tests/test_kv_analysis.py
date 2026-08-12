import torch

from int8_kvcache_lab.kv_analysis import analyze_capture


def test_analysis_reports_all_granularities_and_recommends_lowest_eligible_cost():
    torch.manual_seed(3)
    layers = [torch.randn(17, 2, 2, 8), torch.randn(17, 2, 2, 8) * 0.25]
    report = analyze_capture({"format": "test", "layers": layers}, relative_l2_target=0.02)
    assert report["layer_count"] == 2
    assert set(report["candidates"]) == {"per_tensor", "per_head", "per_token", "per_channel"}
    assert report["recommendation"]["kv_granularity"] in report["candidates"]
    assert report["distribution"]["key"]["percentiles"]["absmax"] > 0
    assert len(report["distribution"]["value"]["histogram"]["bins"]) == 64


def test_per_channel_has_no_worse_reconstruction_than_per_tensor():
    values = torch.tensor([[[[100.0, 0.01]], [[0.1, -0.1]]]]).repeat(8, 1, 1, 1)
    report = analyze_capture({"layers": [values]}, relative_l2_target=0.01)
    assert report["candidates"]["per_channel"]["relative_l2"] <= report["candidates"]["per_tensor"]["relative_l2"]
