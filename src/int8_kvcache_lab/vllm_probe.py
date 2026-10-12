"""Stage 1: run one eager generate and print the paged-attention call site."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch

from .vllm_runtime import configure_worker_multiprocessing, describe_call, format_call_site, make_llm, observe_flash_forward


def probe(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 8,
    max_model_len: int = 256,
    dtype: str = "bfloat16",
) -> dict[str, object]:
    """Generate one prompt and return the first FlashAttention call description."""
    # The callback runs in vLLM's EngineCore.  It must be inherited by that
    # process, and the parent must stay CUDA-uninitialized before ``fork``.
    configure_worker_multiprocessing("fork")
    from vllm import SamplingParams

    seen: dict[str, object] = {}
    descriptor_fd, descriptor_name = tempfile.mkstemp(prefix="int8-kvcache-probe-", suffix=".pt")
    os.close(descriptor_fd)
    descriptor_file = Path(descriptor_name)

    def callback(impl, query, key, value, kv_cache, metadata, output) -> None:
        if seen:
            return
        block_table = getattr(metadata, "block_table", None)
        seqused_k = getattr(metadata, "seq_lens", None)
        description = describe_call(getattr(impl, "vllm_flash_attn_version", None))
        description.update(
            {
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "kv_cache_shape": list(kv_cache.shape),
                "output_shape": list(output.shape),
                "block_table_shape": None if block_table is None else list(block_table.shape),
                "seqused_k_shape": None if seqused_k is None else list(seqused_k.shape),
            }
        )
        seen.update(description)
        torch.save(seen, descriptor_file)
        print(format_call_site(description))
        print(
            "shapes "
            f"q={description['query_shape']} k={description['key_shape']} "
            f"v={description['value_shape']} kv_cache={description['kv_cache_shape']} "
            f"block_table={description['block_table_shape']} seqused_k={description['seqused_k_shape']} "
            f"output={description['output_shape']}"
        )

    try:
        with observe_flash_forward(callback):
            engine = make_llm(
                model,
                dtype=dtype,
                max_model_len=max_model_len,
                worker_multiprocessing_method="fork",
            )
            result = engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=7))
        if not seen and descriptor_file.stat().st_size:
            seen.update(torch.load(descriptor_file, map_location="cpu", weights_only=False))
    finally:
        descriptor_file.unlink(missing_ok=True)
    if not seen:
        raise RuntimeError("generate finished without entering FlashAttentionImpl.forward")
    text = result[0].outputs[0].text if result and result[0].outputs else ""
    seen["generated_text"] = text
    print(f"generated: {text!r}")
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="Explain paged KV cache in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--dtype", choices=("half", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    probe(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
