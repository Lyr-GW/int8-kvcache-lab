"""Tests for Colab-safe vLLM runtime setup."""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from int8_kvcache_lab import vllm_capture, vllm_probe
from int8_kvcache_lab.vllm_runtime import VLLM_WORKER_MULTIPROC_METHOD, configure_worker_multiprocessing


def test_configure_worker_multiprocessing_forces_spawn(monkeypatch):
    """A pre-existing fork choice cannot poison CUDA worker startup."""
    monkeypatch.setenv(VLLM_WORKER_MULTIPROC_METHOD, "fork")

    configure_worker_multiprocessing()

    assert os.environ[VLLM_WORKER_MULTIPROC_METHOD] == "spawn"


def test_configure_worker_multiprocessing_allows_fork_for_capture(monkeypatch):
    """Capture needs its wrapper inherited by the EngineCore child."""
    monkeypatch.setenv(VLLM_WORKER_MULTIPROC_METHOD, "spawn")

    configure_worker_multiprocessing("fork")

    assert os.environ[VLLM_WORKER_MULTIPROC_METHOD] == "fork"


def test_probe_uses_fork_without_parent_cuda_probe(monkeypatch):
    """The Stage-1 callback must be present when EngineCore forks."""
    calls: list[str] = []

    class StopAfterEngineStart(Exception):
        pass

    class FakeEngine:
        def generate(self, *args, **kwargs):
            raise StopAfterEngineStart

    @contextmanager
    def fake_observer(callback):
        calls.append("observer")
        yield

    def fake_make_llm(*args, **kwargs):
        calls.append("engine")
        assert kwargs["worker_multiprocessing_method"] == "fork"
        assert os.environ[VLLM_WORKER_MULTIPROC_METHOD] == "fork"
        return FakeEngine()

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda *args, **kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(vllm_probe, "observe_flash_forward", fake_observer)
    monkeypatch.setattr(vllm_probe, "make_llm", fake_make_llm)
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: pytest.fail("probe must not initialize CUDA in its parent"),
    )

    with pytest.raises(StopAfterEngineStart):
        vllm_probe.probe(model="unused", prompt="unused")

    assert calls == ["observer", "engine"]


def test_capture_uses_fork_without_parent_cuda_probe(monkeypatch):
    """The capture wrapper must be present when EngineCore forks."""
    calls: list[str] = []

    class StopAfterEngineStart(Exception):
        pass

    class FakeEngine:
        def generate(self, *args, **kwargs):
            raise StopAfterEngineStart

    @contextmanager
    def fake_observer(callback):
        calls.append("observer")
        yield

    def fake_make_llm(*args, **kwargs):
        calls.append("engine")
        assert kwargs["worker_multiprocessing_method"] == "fork"
        assert os.environ[VLLM_WORKER_MULTIPROC_METHOD] == "fork"
        return FakeEngine()

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda *args, **kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(vllm_capture, "observe_flash_forward", fake_observer)
    monkeypatch.setattr(vllm_capture, "make_llm", fake_make_llm)
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: pytest.fail("capture must not initialize CUDA in its parent"),
    )

    with pytest.raises(StopAfterEngineStart):
        vllm_capture.capture_decode(model="unused", prompt="unused", output=None)

    assert calls == ["observer", "engine"]
