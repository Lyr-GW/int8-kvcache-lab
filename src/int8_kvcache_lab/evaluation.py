"""Teacher-forced Qwen decode PPL evaluation for Colab."""

from __future__ import annotations

import argparse
import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .qwen_adapter import QwenDynamicKVAdapter
from .reporting import write_report


DIAGNOSTIC_PROMPTS = (
    "Explain why KV cache helps autoregressive decoding in one sentence.",
    "Compute 17 * 23 and show only the answer.",
    "Write a Python function that returns the maximum of two integers.",
)


def _token_stream(tokenizer, samples: int) -> Iterable[torch.Tensor]:
    """Yield a deterministic WikiText-2 token stream without hidden shuffling."""
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    emitted = 0
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() > 2:
            yield ids
            emitted += 1
            if emitted >= samples:
                return


@torch.inference_mode()
def _losses(model, tokens: torch.Tensor, device: torch.device, context: int) -> list[float]:
    """Score native prefill followed by teacher-forced decode tokens."""
    losses: list[float] = []
    start = min(context, tokens.numel() - 1)
    prefill = tokens[:start].unsqueeze(0).to(device)
    prefill_output = model(input_ids=prefill, use_cache=True)
    cache = prefill_output.past_key_values
    losses.append(float(F.cross_entropy(prefill_output.logits[:, -1].float(), tokens[start : start + 1].to(device)).item()))
    for token_index in range(start, tokens.numel() - 1):
        input_id = tokens[token_index : token_index + 1].view(1, 1).to(device)
        output = model(input_ids=input_id, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        target = tokens[token_index + 1 : token_index + 2].to(device)
        losses.append(float(F.cross_entropy(output.logits[:, -1].float(), target).item()))
    return losses


@torch.inference_mode()
def _decode_diagnostics(model, tokenizer, device: torch.device, adapter: QwenDynamicKVAdapter) -> list[dict[str, object]]:
    """Compare one forced decode logit and greedy output for fixed prompts."""
    diagnostics = []
    previous_enabled = adapter.enabled
    try:
        for prompt in DIAGNOSTIC_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            adapter.enabled = False
            baseline_prefill = model(input_ids=input_ids, use_cache=True)
            forced_token = baseline_prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
            baseline_step = model(input_ids=forced_token, past_key_values=baseline_prefill.past_key_values, use_cache=True)
            baseline_generated = model.generate(input_ids, do_sample=False, max_new_tokens=16)

            adapter.enabled = True
            candidate_prefill = model(input_ids=input_ids, use_cache=True)
            candidate_step = model(input_ids=forced_token, past_key_values=candidate_prefill.past_key_values, use_cache=True)
            candidate_generated = model.generate(input_ids, do_sample=False, max_new_tokens=16)
            delta = candidate_step.logits[:, -1].float() - baseline_step.logits[:, -1].float()
            baseline_text = tokenizer.decode(baseline_generated[0], skip_special_tokens=True)
            candidate_text = tokenizer.decode(candidate_generated[0], skip_special_tokens=True)
            diagnostics.append(
                {
                    "prompt": prompt,
                    "logit_max_abs_error": float(delta.abs().max().item()),
                    "logit_relative_l2_error": float(delta.norm().item() / baseline_step.logits[:, -1].float().norm().clamp_min(1e-8).item()),
                    "greedy_exact_match": baseline_text == candidate_text,
                    "baseline_generation": baseline_text,
                    "candidate_generation": candidate_text,
                }
            )
    finally:
        adapter.enabled = previous_enabled
    return diagnostics


def evaluate(model_name: str, samples: int, context: int, output_dir: str) -> int:
    """Run baseline then dynamic decode PPL and return a process status."""
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen evaluation requires a CUDA runtime")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager").to(device).eval()
    streams = list(_token_stream(tokenizer, samples))
    if not streams:
        raise RuntimeError("WikiText-2 did not produce any evaluation token streams")
    baseline = [loss for stream in streams for loss in _losses(model, stream, device, context)]
    adapter = QwenDynamicKVAdapter()
    adapter.install(model)
    candidate = [loss for stream in streams for loss in _losses(model, stream, device, context)]
    diagnostics = _decode_diagnostics(model, tokenizer, device, adapter)
    baseline_ppl = math.exp(sum(baseline) / len(baseline))
    candidate_ppl = math.exp(sum(candidate) / len(candidate))
    relative_change = candidate_ppl / baseline_ppl - 1.0
    report = write_report(
        {
            "kind": "qwen_dynamic_int8_ppl",
            "model": model_name,
            "samples": samples,
            "context": context,
            "baseline_ppl": baseline_ppl,
            "candidate_ppl": candidate_ppl,
            "relative_ppl_change": relative_change,
            "passes_strict_gate": relative_change <= 0.01,
            "decode_diagnostics": diagnostics,
        },
        output_dir,
    )
    print(f"baseline_ppl={baseline_ppl:.5f} candidate_ppl={candidate_ppl:.5f} delta={relative_change:.3%}")
    print(f"report={report}")
    return 0 if relative_change <= 0.01 else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    raise SystemExit(evaluate(args.model, args.samples, args.context, args.output_dir))


if __name__ == "__main__":
    main()
