"""Compare a stage-2 capture with the stage-3 PyTorch reference. No GPU is required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .pytorch_ref import ref_paged_attn, relative_l2
from .reporting import write_report


def _output_as_heads(output: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if output.shape == query.shape:
        return output
    if output.ndim == 2 and output.shape[0] == query.shape[0] and output.shape[1] == query.shape[1] * query.shape[2]:
        return output.view_as(query)
    raise ValueError(f"cannot align captured output {tuple(output.shape)} with query {tuple(query.shape)}")


def replay(capture: dict[str, Any]) -> dict[str, Any]:
    """Replay every boolean-causal layer in ``capture`` and summarize the error."""
    steps = capture.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("capture has no FlashAttention steps; rerun vllm_capture against vLLM 0.29")
    errors: list[float] = []
    skipped = 0
    for step in steps:
        for layer in step["layers"]:
            if layer.get("causal") is not True:
                skipped += 1
                continue
            predicted = ref_paged_attn(
                layer["query"],
                layer["key_cache"],
                layer["value_cache"],
                layer["cu_seqlens_q"],
                layer["seqused_k"],
                layer["block_table"],
                softmax_scale=float(layer["softmax_scale"]),
                causal=True,
            )
            errors.append(relative_l2(predicted, _output_as_heads(layer["output"], layer["query"])))
    if not errors:
        raise ValueError("capture did not contain a causal paged-attention layer")
    return {
        "layers_compared": len(errors),
        "layers_skipped": skipped,
        "max_relative_l2": max(errors),
        "mean_relative_l2": sum(errors) / len(errors),
        "passes_stage3_gate": max(errors) <= 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    report = replay(capture)
    path = write_report({"experiment": "stage3_replay", **report}, args.output_dir)
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")
    raise SystemExit(0 if report["passes_stage3_gate"] else 2)


if __name__ == "__main__":
    main()
