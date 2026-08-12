from types import SimpleNamespace

import pytest
import torch

from int8_kvcache_lab.evaluation import _losses, _perplexity, _select_model_dtype


class _RecordingModel:
    """CPU model double that exposes cache-call metadata to the test."""

    def __init__(self, *, nan_on_call: int | None = None):
        self.calls: list[dict[str, object]] = []
        self.nan_on_call = nan_on_call

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        call_index = len(self.calls)
        logits = torch.zeros((1, 1, 16), dtype=torch.float32)
        if self.nan_on_call == call_index:
            logits.fill_(float("nan"))
        else:
            logits[..., 0] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=f"cache-{call_index}")


def test_losses_passes_full_prefix_masks_and_exact_cache_positions():
    model = _RecordingModel()

    losses = _losses(model, torch.tensor([1, 2, 3, 4]), torch.device("cpu"), context=2, sample_index=7)

    assert len(losses) == 2
    assert len(model.calls) == 2
    assert model.calls[0]["attention_mask"].tolist() == [[1, 1]]
    assert model.calls[0]["cache_position"].tolist() == [0, 1]
    assert model.calls[1]["attention_mask"].tolist() == [[1, 1, 1]]
    assert model.calls[1]["cache_position"].tolist() == [2]


def test_losses_fails_with_sample_and_token_context_for_nonfinite_logits():
    model = _RecordingModel(nan_on_call=2)

    with pytest.raises(RuntimeError, match=r"non-finite logits: sample_index=3, token_index=3"):
        _losses(model, torch.tensor([1, 2, 3, 4]), torch.device("cpu"), context=2, sample_index=3)


def test_perplexity_rejects_nonfinite_inputs_and_empty_inputs():
    with pytest.raises(RuntimeError, match="no token losses"):
        _perplexity([], label="baseline")
    with pytest.raises(RuntimeError, match="mean loss is non-finite"):
        _perplexity([float("nan")], label="baseline")


def test_select_model_dtype_prefers_bf16_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _select_model_dtype() is torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _select_model_dtype() is torch.float16
