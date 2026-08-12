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


def _select_model_dtype() -> torch.dtype:
    """Prefer the checkpoint's BF16-friendly execution dtype when CUDA supports it."""
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _assert_finite(value: torch.Tensor, *, kind: str, sample_index: int, token_index: int) -> None:
    """Reject invalid model outputs before they become a meaningless PPL."""
    if bool(torch.isfinite(value).all()):
        return
    invalid = int((~torch.isfinite(value)).sum().item())
    raise RuntimeError(
        f"non-finite {kind}: sample_index={sample_index}, token_index={token_index}, "
        f"shape={tuple(value.shape)}, invalid_values={invalid}"
    )


def _next_token_loss(
    logits: torch.Tensor, target: torch.Tensor, *, sample_index: int, token_index: int
) -> float:
    """Compute one teacher-forced loss with actionable finite-value checks."""
    next_logits = logits[:, -1].float()
    _assert_finite(next_logits, kind="logits", sample_index=sample_index, token_index=token_index)
    loss = F.cross_entropy(next_logits, target)
    _assert_finite(loss, kind="loss", sample_index=sample_index, token_index=token_index)
    return float(loss.item())


def _perplexity(losses: list[float], *, label: str) -> float:
    """Convert checked token losses to finite perplexity."""
    if not losses:
        raise RuntimeError(f"cannot calculate {label} PPL: evaluation produced no token losses")
    mean_loss = sum(losses) / len(losses)
    if not math.isfinite(mean_loss):
        raise RuntimeError(f"cannot calculate {label} PPL: mean loss is non-finite ({mean_loss!r})")
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError as error:
        raise RuntimeError(f"cannot calculate {label} PPL: exp(mean_loss={mean_loss!r}) overflowed") from error
    if not math.isfinite(perplexity):
        raise RuntimeError(f"cannot calculate {label} PPL: result is non-finite ({perplexity!r})")
    return perplexity


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
def _losses(
    model, tokens: torch.Tensor, device: torch.device, context: int, sample_index: int = 0
) -> list[float]:
    """Score prefill then cached teacher-forced decode with explicit cache metadata."""
    if context < 1:
        raise ValueError(f"context must be at least 1, got {context}")
    if tokens.numel() < 2:
        raise ValueError(f"sample_index={sample_index} needs at least two tokens, got {tokens.numel()}")
    losses: list[float] = []
    start = min(context, tokens.numel() - 1)
    prefill = tokens[:start].unsqueeze(0).to(device)
    prefill_mask = torch.ones((1, start), dtype=torch.long, device=device)
    prefill_position = torch.arange(start, dtype=torch.long, device=device)
    prefill_output = model(
        input_ids=prefill,
        attention_mask=prefill_mask,
        cache_position=prefill_position,
        use_cache=True,
    )
    cache = prefill_output.past_key_values
    losses.append(
        _next_token_loss(
            prefill_output.logits,
            tokens[start : start + 1].to(device),
            sample_index=sample_index,
            token_index=start,
        )
    )
    for token_index in range(start, tokens.numel() - 1):
        input_id = tokens[token_index : token_index + 1].view(1, 1).to(device)
        # Transformers 4.48 Qwen2 uses the mask length as the causal-mask
        # target length.  Supplying the complete prefix prevents it from
        # inferring an extra future cache slot during one-token decode.
        decode_mask = torch.ones((1, token_index + 1), dtype=torch.long, device=device)
        decode_position = torch.tensor([token_index], dtype=torch.long, device=device)
        output = model(
            input_ids=input_id,
            attention_mask=decode_mask,
            past_key_values=cache,
            cache_position=decode_position,
            use_cache=True,
        )
        cache = output.past_key_values
        target = tokens[token_index + 1 : token_index + 2].to(device)
        losses.append(
            _next_token_loss(output.logits, target, sample_index=sample_index, token_index=token_index + 1)
        )
    return losses


@torch.inference_mode()
def _decode_diagnostics(model, tokenizer, device: torch.device, adapter: QwenDynamicKVAdapter) -> list[dict[str, object]]:
    """Compare one forced decode logit and greedy output for fixed prompts."""
    diagnostics = []
    previous_enabled = adapter.enabled
    try:
        for sample_index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_length = input_ids.shape[1]
            prompt_mask = torch.ones((1, prompt_length), dtype=torch.long, device=device)
            prompt_position = torch.arange(prompt_length, dtype=torch.long, device=device)
            decode_mask = torch.ones((1, prompt_length + 1), dtype=torch.long, device=device)
            decode_position = torch.tensor([prompt_length], dtype=torch.long, device=device)
            adapter.enabled = False
            baseline_prefill = model(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                cache_position=prompt_position,
                use_cache=True,
            )
            _assert_finite(baseline_prefill.logits, kind="diagnostic baseline prefill logits", sample_index=sample_index, token_index=prompt_length)
            forced_token = baseline_prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
            baseline_step = model(
                input_ids=forced_token,
                attention_mask=decode_mask,
                past_key_values=baseline_prefill.past_key_values,
                cache_position=decode_position,
                use_cache=True,
            )
            _assert_finite(baseline_step.logits, kind="diagnostic baseline decode logits", sample_index=sample_index, token_index=prompt_length + 1)
            baseline_generated = model.generate(input_ids, attention_mask=prompt_mask, do_sample=False, max_new_tokens=16)

            adapter.enabled = True
            candidate_prefill = model(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                cache_position=prompt_position,
                use_cache=True,
            )
            _assert_finite(candidate_prefill.logits, kind="diagnostic candidate prefill logits", sample_index=sample_index, token_index=prompt_length)
            candidate_step = model(
                input_ids=forced_token,
                attention_mask=decode_mask,
                past_key_values=candidate_prefill.past_key_values,
                cache_position=decode_position,
                use_cache=True,
            )
            _assert_finite(candidate_step.logits, kind="diagnostic candidate decode logits", sample_index=sample_index, token_index=prompt_length + 1)
            candidate_generated = model.generate(input_ids, attention_mask=prompt_mask, do_sample=False, max_new_tokens=16)
            delta = candidate_step.logits[:, -1].float() - baseline_step.logits[:, -1].float()
            _assert_finite(delta, kind="diagnostic logit delta", sample_index=sample_index, token_index=prompt_length + 1)
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
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    if context < 1:
        raise ValueError(f"context must be at least 1, got {context}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen evaluation requires a CUDA runtime")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_dtype = _select_model_dtype()
    print(f"loading {model_name} with {model_dtype}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, attn_implementation="eager").to(device).eval()
    streams = list(_token_stream(tokenizer, samples))
    if not streams:
        raise RuntimeError("WikiText-2 did not produce any evaluation token streams")
    baseline = [
        loss
        for sample_index, stream in enumerate(streams)
        for loss in _losses(model, stream, device, context, sample_index)
    ]
    adapter = QwenDynamicKVAdapter()
    adapter.install(model)
    candidate = [
        loss
        for sample_index, stream in enumerate(streams)
        for loss in _losses(model, stream, device, context, sample_index)
    ]
    diagnostics = _decode_diagnostics(model, tokenizer, device, adapter)
    baseline_ppl = _perplexity(baseline, label="baseline")
    candidate_ppl = _perplexity(candidate, label="candidate")
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
