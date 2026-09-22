"""Capture real Q/K/V/KV-cache tensors from vLLM 0.29 FlashAttention.

The observer wraps ``FlashAttentionImpl.forward`` after the native kernel
returns. It does not edit the vLLM checkout. Each recorded step keeps the
query, the newly projected K/V, the paged K/V pages addressed by
``block_table``, ``slot_mapping``, ``cu_seqlens``, ``seqused_k``, and the
kernel output. Unused preallocated pages are not stored.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from .vllm_runtime import (
    describe_call,
    format_call_site,
    make_llm,
    observe_flash_forward,
    pack_referenced_pages,
    split_packed_kv_cache,
)


def _metadata_tensor(metadata: Any, *names: str) -> torch.Tensor | None:
    for name in names:
        value = getattr(metadata, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _logical_layer(key_cache: torch.Tensor, value_cache: torch.Tensor, block_table: torch.Tensor, seqused_k: torch.Tensor) -> torch.Tensor:
    """Gather valid K/V into ``[tokens, 2, kv_heads, head_dim]``."""
    pieces: list[torch.Tensor] = []
    block_size = key_cache.shape[1]
    for row, length in enumerate(seqused_k.tolist()):
        length = int(length)
        if length <= 0:
            continue
        pages = (length + block_size - 1) // block_size
        block_ids = block_table[row, :pages].to(dtype=torch.long)
        key = key_cache[block_ids].reshape(-1, key_cache.shape[2], key_cache.shape[3])[:length]
        value = value_cache[block_ids].reshape(-1, value_cache.shape[2], value_cache.shape[3])[:length]
        pieces.append(torch.stack((key, value), dim=1))
    if not pieces:
        raise ValueError("capture step did not address any KV tokens")
    return torch.cat(pieces, dim=0)


class _ForwardRecorder:
    """Group per-layer forward taps into decoder steps."""

    def __init__(self) -> None:
        self.call_site: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self._open_key: tuple[Any, ...] | None = None
        self._open_layers: list[dict[str, Any]] = []

    def __call__(self, impl: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, metadata: Any, output: torch.Tensor) -> None:
        block_table = _metadata_tensor(metadata, "block_table", "block_table_tensor")
        seqused_k = _metadata_tensor(metadata, "seq_lens", "seqused_k")
        cu_seqlens_q = _metadata_tensor(metadata, "query_start_loc", "cu_seqlens_q")
        if block_table is None or seqused_k is None or cu_seqlens_q is None:
            return
        num_tokens = int(getattr(metadata, "num_actual_tokens", query.shape[0]))
        key_cache, value_cache = split_packed_kv_cache(kv_cache, impl.head_size)
        packed_key, packed_value, local_table = pack_referenced_pages(
            key_cache, value_cache, block_table, seqused_k
        )
        slot_mapping = _metadata_tensor(metadata, "slot_mapping")
        if self.call_site is None:
            module_file = None
            try:
                import vllm.v1.attention.backends.flash_attn as flash_module

                module_file = flash_module.__file__
            except ImportError:
                module_file = None
            self.call_site = describe_call(getattr(impl, "vllm_flash_attn_version", None), module_file)
            print(format_call_site(self.call_site))
        causal_flag = getattr(metadata, "causal", True)
        layer = {
            "query": query[:num_tokens].detach().cpu(),
            "key": key[:num_tokens].detach().cpu(),
            "value": value[:num_tokens].detach().cpu(),
            "key_cache": packed_key,
            "value_cache": packed_value,
            "block_table": local_table,
            "slot_mapping": None if slot_mapping is None else slot_mapping.detach().cpu(),
            "cu_seqlens_q": cu_seqlens_q.detach().cpu(),
            "seqused_k": seqused_k.detach().cpu(),
            "output": output[:num_tokens].detach().cpu(),
            "causal": causal_flag if isinstance(causal_flag, bool) else "tensor",
            "softmax_scale": float(impl.scale),
        }
        step_key = (tuple(seqused_k.detach().cpu().tolist()), tuple(cu_seqlens_q.detach().cpu().tolist()))
        if step_key != self._open_key:
            self._flush()
            self._open_key = step_key
        self._open_layers.append(layer)

    def _flush(self) -> None:
        if self._open_layers:
            self.steps.append({"layers": self._open_layers})
            self._open_layers = []

    def finish(self) -> dict[str, Any]:
        self._flush()
        if not self.steps or self.call_site is None:
            raise RuntimeError("vLLM finished without a FlashAttention forward that carried a block table")
        last = self.steps[-1]["layers"]
        logical_layers = [
            _logical_layer(layer["key_cache"], layer["value_cache"], layer["block_table"], layer["seqused_k"])
            for layer in last
        ]
        return {
            "format": "int8-kvcache-lab.vllm-capture.v2",
            "call_site": self.call_site,
            "rank": int(os.environ.get("RANK", "0")),
            "block_size": int(last[0]["key_cache"].shape[1]),
            "sequence_length": int(last[0]["seqused_k"].max().item()),
            "block_tables": last[0]["block_table"],
            "layers": logical_layers,
            "steps": self.steps,
        }


def capture_decode(
    *,
    model: str,
    prompt: str,
    output: Path,
    max_tokens: int = 16,
    max_model_len: int = 256,
    dtype: str = "bfloat16",
) -> Path:
    """Run one eager generate and save the FlashAttention inputs and outputs."""
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM capture requires an NVIDIA CUDA GPU")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    from vllm import SamplingParams

    recorder = _ForwardRecorder()
    # Build the engine before installing the observer so startup profiling
    # does not get mixed into the captured generate steps.
    engine = make_llm(model, dtype=dtype, max_model_len=max_model_len)
    with observe_flash_forward(recorder):
        engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=7))
    snapshot = recorder.finish()
    snapshot.update({"model": model, "dtype": dtype, "prompt": prompt, "max_model_len": max_model_len})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, output)
    return output


def main() -> None:
    """CLI entry point used by the Colab stage-2 cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="What is 17 * 23? Reply with the integer only.")
    parser.add_argument("--dataset", choices=("prompt", "aime"), default="prompt")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm-qwen-kv-cache.pt"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--dtype", choices=("half", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    prompt = args.prompt
    if args.dataset == "aime":
        prompt = _aime_prompt(prompt)
    path = capture_decode(
        model=args.model,
        prompt=prompt,
        output=args.output,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    print(f"wrote {path}")


def _aime_prompt(fallback: str) -> str:
    """Load one AIME problem when the dataset is reachable, else keep the fallback."""
    try:
        from datasets import load_dataset

        row = next(iter(load_dataset("HuggingFaceH4/aime_2024", split="train")))
        problem = row.get("problem") or row.get("question") or fallback
        return str(problem)
    except Exception as error:
        print(f"AIME dataset is unavailable ({error}); using the built-in prompt.")
        return fallback


if __name__ == "__main__":
    main()
