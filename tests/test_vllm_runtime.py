"""Tests for Colab-safe vLLM runtime setup."""

import os

import pytest

from int8_kvcache_lab import vllm_capture, vllm_compare, vllm_probe
from int8_kvcache_lab.vllm_runtime import VLLM_WORKER_MULTIPROC_METHOD, configure_worker_multiprocessing


def test_configure_worker_multiprocessing_forces_spawn(monkeypatch):
    """A pre-existing fork choice cannot poison CUDA worker startup."""
    monkeypatch.setenv(VLLM_WORKER_MULTIPROC_METHOD, "fork")

    configure_worker_multiprocessing()

    assert os.environ[VLLM_WORKER_MULTIPROC_METHOD] == "spawn"


@pytest.mark.parametrize(
    ("entrypoint", "kwargs", "message"),
    [
        (vllm_probe.probe, {"model": "unused", "prompt": "unused"}, "CUDA GPU"),
        (vllm_capture.capture_decode, {"model": "unused", "prompt": "unused", "output": None}, "CUDA GPU"),
        (vllm_compare.compare, {"model": "unused", "prompt": "unused"}, "CUDA GPU"),
    ],
)
def test_vllm_entrypoints_choose_spawn_before_cuda_probe(monkeypatch, entrypoint, kwargs, message):
    """CLI entrypoints must configure workers before touching CUDA."""
    monkeypatch.setenv(VLLM_WORKER_MULTIPROC_METHOD, "fork")

    def no_cuda():
        assert os.environ[VLLM_WORKER_MULTIPROC_METHOD] == "spawn"
        return False

    monkeypatch.setattr("torch.cuda.is_available", no_cuda)
    with pytest.raises(RuntimeError, match=message):
        entrypoint(**kwargs)
