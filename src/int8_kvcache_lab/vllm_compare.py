"""Stage 3/5 check: swap ``flash_attn_varlen_func`` for the PyTorch implementations.

The engine is built with the native kernel. Each comparison then generates
the same prompt again while the varlen call is replaced. Startup profiling
stays on the native kernel.
"""

from __future__ import annotations

import argparse

import torch

from .vllm_runtime import make_llm, replace_varlen_with_pytorch


def _generate(engine, prompt: str, max_tokens: int) -> str:
    from vllm import SamplingParams

    result = engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=7))
    return result[0].outputs[0].text


def compare(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 8,
    max_model_len: int = 256,
    dtype: str = "bfloat16",
    granularity: str = "per_channel",
) -> dict[str, str]:
    """Return native, FP-reference, dequant, and INT8-GEMM generations."""
    if not torch.cuda.is_available():
        raise RuntimeError("the vLLM replacement comparison requires a CUDA GPU")
    if granularity not in ("per_head", "per_channel"):
        raise ValueError("granularity must be per_head or per_channel")
    engine = make_llm(model, dtype=dtype, max_model_len=max_model_len)
    texts = {"native_fp": _generate(engine, prompt, max_tokens)}
    with replace_varlen_with_pytorch("fp"):
        texts["pytorch_fp"] = _generate(engine, prompt, max_tokens)
    with replace_varlen_with_pytorch("dequant", granularity=granularity):
        texts["dequant"] = _generate(engine, prompt, max_tokens)
    with replace_varlen_with_pytorch("int8_gemm", granularity=granularity):
        texts["int8_gemm"] = _generate(engine, prompt, max_tokens)
    for name, text in texts.items():
        print(f"{name}: {text!r}")
    print(f"pytorch_fp_matches_native={texts['pytorch_fp'] == texts['native_fp']}")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="What is 17 * 23? Reply with the integer only.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--dtype", choices=("half", "bfloat16"), default="bfloat16")
    parser.add_argument("--granularity", choices=("per_head", "per_channel"), default="per_channel")
    args = parser.parse_args()
    compare(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        granularity=args.granularity,
    )


if __name__ == "__main__":
    main()
