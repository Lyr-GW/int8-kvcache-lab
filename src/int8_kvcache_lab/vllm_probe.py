"""Stage 1: run one eager generate and print the paged-attention call site."""

from __future__ import annotations

import argparse

import torch

from .vllm_runtime import describe_call, format_call_site, make_llm, observe_flash_forward


def probe(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 8,
    max_model_len: int = 256,
    dtype: str = "bfloat16",
) -> dict[str, object]:
    """Generate one prompt and return the first FlashAttention call description."""
    if not torch.cuda.is_available():
        raise RuntimeError("the vLLM probe requires an NVIDIA CUDA GPU")
    from vllm import SamplingParams

    seen: dict[str, object] = {}

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
        print(format_call_site(description))
        print(
            "shapes "
            f"q={description['query_shape']} k={description['key_shape']} "
            f"v={description['value_shape']} kv_cache={description['kv_cache_shape']} "
            f"block_table={description['block_table_shape']} seqused_k={description['seqused_k_shape']} "
            f"output={description['output_shape']}"
        )

    engine = make_llm(model, dtype=dtype, max_model_len=max_model_len)
    with observe_flash_forward(callback):
        result = engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=7))
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
