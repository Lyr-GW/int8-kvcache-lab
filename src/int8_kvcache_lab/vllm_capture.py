"""Version-pinned, opt-in vLLM decode KV-cache capture.

This module supports vLLM ``v0.6.6`` only.  It observes the cache immediately
after a real eager decode forward, gathers only the logical tokens addressed by
``block_tables``/``seq_lens_tensor``, and writes a CPU artifact for calibration.
No vLLM source file is modified.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

import torch

from .paged_cache import PagedKVCache


SUPPORTED_VLLM_VERSION = "0.6.6"


def _require_vllm() -> tuple[Any, Any, Any]:
    """Import the pinned optional runtime with a clear incompatibility error."""
    try:
        version = importlib.metadata.version("vllm")
        from vllm import LLM, SamplingParams
        from vllm.worker.model_runner import ModelRunner
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            "vLLM capture is optional. Install the pinned runtime with "
            "`bash scripts/install_vllm_capture.sh` in a fresh CUDA Colab runtime."
        ) from error
    if version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"vLLM capture requires vLLM {SUPPORTED_VLLM_VERSION}, found {version}")
    return LLM, SamplingParams, ModelRunner


def _canonicalize_layer(kv_cache: torch.Tensor, *, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Convert vLLM's v0.6.6 physical cache into this lab's canonical layout."""
    if kv_cache.ndim == 5 and kv_cache.shape[0] == 2:
        # FlashAttention backend: [K/V, blocks, block_size, heads, dim].
        if kv_cache.shape[3:] != (num_kv_heads, head_dim):
            raise ValueError(f"unexpected 5D vLLM KV shape: {tuple(kv_cache.shape)}")
        return kv_cache.permute(1, 2, 0, 3, 4).contiguous()
    if kv_cache.ndim == 3 and kv_cache.shape[0] == 2:
        # PagedAttention/XFormers backend packs K as [head_dim/x, block, x].
        element_group = 16 // kv_cache.element_size()
        if head_dim % element_group:
            raise ValueError("vLLM packed key cache requires head_dim divisible by 16 / element_size")
        blocks = kv_cache.shape[1]
        key = kv_cache[0].view(blocks, num_kv_heads, head_dim // element_group, -1, element_group)
        value = kv_cache[1].view(blocks, num_kv_heads, head_dim, -1)
        key = key.permute(0, 3, 1, 2, 4).reshape(blocks, key.shape[3], num_kv_heads, head_dim)
        value = value.permute(0, 3, 1, 2).contiguous()
        return torch.stack((key, value), dim=2)
    raise ValueError(f"unsupported vLLM v0.6.6 KV-cache shape: {tuple(kv_cache.shape)}")


def _logical_layers(
    kv_caches: list[torch.Tensor],
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> list[torch.Tensor]:
    """Gather only the valid logical K/V tokens, preserving all transformer layers."""
    if block_tables.ndim != 2 or seq_lens.ndim != 1 or seq_lens.numel() != 1:
        raise ValueError("initial capture supports exactly one decode sequence")
    logical: list[torch.Tensor] = []
    for layer in kv_caches:
        physical = _canonicalize_layer(layer, num_kv_heads=num_kv_heads, head_dim=head_dim)
        cache = PagedKVCache(physical)
        key, value, mask = cache.gather(block_tables, seq_lens)
        logical.append(torch.stack((key[0, mask[0]], value[0, mask[0]]), dim=1).detach().cpu())
    return logical


class _DecodeCapture:
    """Small in-process observer installed around the vLLM v0.6.6 ModelRunner."""

    def __init__(self, *, num_kv_heads: int, head_dim: int) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.snapshot: dict[str, Any] | None = None

    def observe(self, model_input: Any, kv_caches: list[torch.Tensor]) -> None:
        metadata = getattr(model_input, "attn_metadata", None)
        decode = getattr(metadata, "decode_metadata", None)
        if decode is None:
            return
        block_tables = getattr(decode, "block_tables", None)
        seq_lens = getattr(decode, "seq_lens_tensor", None)
        if block_tables is None or seq_lens is None or not int(seq_lens.numel()):
            return
        # CUDA graphs pad metadata rows; eager single-sequence execution has one.
        block_tables = block_tables[:1].detach()
        seq_lens = seq_lens[:1].detach().to(dtype=torch.long)
        self.snapshot = {
            "format": "int8-kvcache-lab.vllm-capture.v1",
            "vllm_version": SUPPORTED_VLLM_VERSION,
            "block_size": int(
                kv_caches[0].shape[2]
                if kv_caches[0].ndim == 5
                else kv_caches[0].shape[-1] // (self.num_kv_heads * self.head_dim)
            ),
            "sequence_length": int(seq_lens[0].item()),
            "block_tables": block_tables.cpu(),
            "layers": _logical_layers(
                kv_caches,
                block_tables=block_tables,
                seq_lens=seq_lens,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            ),
        }


def capture_decode(
    *,
    model: str,
    prompt: str,
    output: Path,
    max_tokens: int = 16,
    max_model_len: int = 1024,
    dtype: str = "bfloat16",
) -> Path:
    """Run one Qwen decode request under vLLM and save its latest cache state."""
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM capture requires an NVIDIA CUDA GPU")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2 so a decode forward occurs")
    LLM, SamplingParams, ModelRunner = _require_vllm()
    config = __import__("transformers").AutoConfig.from_pretrained(model)
    capture = _DecodeCapture(num_kv_heads=config.num_key_value_heads, head_dim=config.hidden_size // config.num_attention_heads)
    original = ModelRunner.execute_model

    def wrapped(runner: Any, model_input: Any, kv_caches: list[torch.Tensor], *args: Any, **kwargs: Any) -> Any:
        result = original(runner, model_input, kv_caches, *args, **kwargs)
        capture.observe(model_input, kv_caches)
        return result

    ModelRunner.execute_model = wrapped
    try:
        engine = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enforce_eager=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.75,
            disable_log_stats=True,
        )
        engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=7))
    finally:
        ModelRunner.execute_model = original
    if capture.snapshot is None:
        raise RuntimeError("vLLM completed without an observable eager decode cache snapshot")
    capture.snapshot.update({"model": model, "dtype": dtype, "prompt_character_count": len(prompt)})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(capture.snapshot, output)
    return output


def main() -> None:
    """CLI entry point used by the vLLM calibration Colab notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", default="Explain why paged KV caches improve decode throughput.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm-qwen-kv-cache.pt"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=("half", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    path = capture_decode(**vars(args))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
