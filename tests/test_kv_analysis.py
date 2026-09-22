import json

import torch

from int8_kvcache_lab.kv_analysis import analyze_capture


def test_analysis_reports_all_granularities_and_a_supported_recommendation():
    torch.manual_seed(3)
    layers = [torch.randn(17, 2, 2, 8), torch.randn(17, 2, 2, 8) * 0.25]
    report = analyze_capture({"format": "test", "layers": layers}, relative_l2_target=0.02)
    assert report["layer_count"] == 2
    assert set(report["candidates"]) == {"per_tensor", "per_head", "per_token", "per_channel"}
    assert report["recommendation"]["kv_granularity"] in report["candidates"]
    assert report["distribution"]["key"]["percentiles"]["absmax"] > 0
    assert len(report["distribution"]["value"]["histogram"]["bins"]) == 64


def test_channel_spread_selects_per_channel_and_beats_per_tensor():
    values = torch.tensor([[[[100.0, 0.01]], [[0.1, -0.1]]]]).repeat(8, 1, 1, 1)
    report = analyze_capture({"layers": [values]}, relative_l2_target=0.01)
    assert report["recommendation"]["kv_granularity"] == "per_channel"
    assert report["spread"]["within_head_max_ratio"] >= 8
    assert report["candidates"]["per_channel"]["relative_l2"] <= report["candidates"]["per_tensor"]["relative_l2"]


def test_the_report_stays_json_encodable_when_the_capture_carries_tensors():
    """Regression: a real capture embeds tensors and a step tape, which used to
    make the report raise 'Object of type Tensor is not JSON serializable' after
    the whole analysis had already run."""
    capture = {
        "format": "test",
        "block_tables": torch.zeros(1, 2, dtype=torch.long),
        "steps": [{"layers": [{"query": torch.zeros(1, 2, 4)}]}],
        "layers": [torch.randn(5, 2, 2, 4)],
    }
    report = analyze_capture(capture, relative_l2_target=0.5)
    # No tensor may survive into the report, so plain json.dumps must work.
    encoded = json.loads(json.dumps(report))
    assert encoded["capture"]["block_tables"]["tensor_shape"] == [1, 2]
    assert encoded["capture"]["capture_step_count"] == 1
    assert "steps" not in encoded["capture"]
    assert "layers" not in encoded["capture"]
    assert encoded["kv_tokens_analyzed"] == 5
