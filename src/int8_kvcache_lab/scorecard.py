"""Stage-5 score table: native FP versus per-channel dequant and INT8 GEMM.

The default task list is a four-item smoke set, one item each for AIME-style
integer answers, a tiny code function, GPQA-style letters, and MMLU-style
letters. It is small enough to finish on a Colab GPU after the PyTorch
attention replacement. Pass ``--dataset huggingface`` to pull the public AIME, MMLU, and GPQA splits
and score the first ``--limit`` rows of each. HumanEval completions are kept
in the report and left unscored; the micro function item is executed against
its declared calls.

Prefill stays on the original FP attention. Decode steps use the selected
INT8 implementation. This measures dynamic quantization, not a resident INT8
cache.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Callable

import torch

from .config import QuantConfig
from .qwen_adapter import QwenDynamicKVAdapter
from .reporting import write_report


MICRO_TASKS: tuple[dict[str, Any], ...] = (
    {
        "suite": "aime",
        "prompt": "What is 17 * 23? Reply with the integer only.",
        "answer": "391",
        "kind": "integer",
    },
    {
        "suite": "humaneval",
        "prompt": "Define a Python function max2(a, b) that returns the larger of two integers. Output only the function.",
        "answer": "max2",
        "kind": "function",
        "calls": ((3, 9, 9), (1, 0, 1), (-4, -1, -1)),
    },
    {
        "suite": "gpqa",
        "prompt": "Which particle has a negative electric charge?\nA. proton\nB. electron\nC. neutron\nD. photon\nReply with the letter only.",
        "answer": "B",
        "kind": "letter",
    },
    {
        "suite": "mmlu",
        "prompt": "The capital of France is\nA. Berlin\nB. Paris\nC. Madrid\nD. Rome\nReply with the letter only.",
        "answer": "B",
        "kind": "letter",
    },
)

GROUPS: tuple[tuple[str, str | None, str], ...] = (
    ("native_fp", None, "per_head"),
    ("per_channel_dequant", "dequant", "per_channel"),
    ("per_channel_int8gemm", "int8_gemm", "per_channel"),
    ("per_head_dequant", "dequant", "per_head"),
)


def grade(item: dict[str, Any], text: str) -> bool | None:
    """Score one completion. A function item without declared calls is left unscored."""
    kind = item["kind"]
    if kind == "integer":
        numbers = re.findall(r"-?\d+", text)
        return bool(numbers) and numbers[-1] == str(item["answer"])
    if kind == "letter":
        match = re.search(r"\b([ABCD])\b", text.upper())
        return match is not None and match.group(1) == str(item["answer"]).upper()
    if kind == "contains":
        return str(item["answer"]).lower() in text.lower()
    if kind == "function":
        calls = tuple(item.get("calls") or ())
        if not calls:
            return None
        return _function_passes(text, str(item["answer"]), calls)
    raise ValueError(f"unsupported task kind: {kind}")


def _function_passes(source: str, name: str, calls: tuple[tuple[int, ...], ...]) -> bool:
    match = re.search(rf"def {re.escape(name)}\b[\s\S]*", source)
    if match is None:
        return False
    namespace: dict[str, Any] = {}
    try:
        exec(match.group(0), namespace)  # noqa: S102 - grading a function the prompt asked for
    except Exception:
        return False
    function = namespace.get(name)
    if not isinstance(function, Callable):
        return False
    for *args, expected in calls:
        try:
            actual = function(*args)
        except Exception:
            return False
        if actual != expected:
            return False
    return True


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse scored items into one accuracy per suite and group."""
    table: dict[tuple[str, str], list[bool]] = {}
    unscored: dict[tuple[str, str], int] = {}
    for row in results:
        key = (row["suite"], row["group"])
        if row["correct"] is None:
            unscored[key] = unscored.get(key, 0) + 1
            continue
        table.setdefault(key, []).append(bool(row["correct"]))
    keys = sorted(set(table) | set(unscored))
    summary = []
    for suite, group in keys:
        grades = table.get((suite, group), [])
        summary.append(
            {
                "suite": suite,
                "group": group,
                "correct": sum(grades),
                "total": len(grades),
                "unscored": unscored.get((suite, group), 0),
                "accuracy": None if not grades else sum(grades) / len(grades),
            }
        )
    return summary


def _generate(model: Any, tokenizer: Any, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


def run_scorecard(
    model_name: str,
    tasks: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    output_dir: str,
) -> dict[str, Any]:
    """Load Qwen once and score native FP plus the INT8 decode replacements."""
    if not torch.cuda.is_available():
        raise RuntimeError("the stage-5 scorecard requires a CUDA GPU")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    adapter = QwenDynamicKVAdapter()
    adapter.install(model)
    rows: list[dict[str, Any]] = []
    try:
        for group, implementation, granularity in GROUPS:
            adapter.enabled = implementation is not None
            if implementation is not None:
                adapter.implementation = implementation
                adapter.config = QuantConfig(kv_granularity=granularity)
            for item in tasks:
                completion = _generate(model, tokenizer, item["prompt"], device, max_new_tokens)
                rows.append(
                    {
                        "suite": item["suite"],
                        "group": group,
                        "prompt": item["prompt"],
                        "completion": completion,
                        "correct": grade(item, completion),
                    }
                )
    finally:
        adapter.uninstall(model)
    summary = summarize(rows)
    report = {
        "kind": "stage5_scorecard",
        "model": model_name,
        "note": "Prefill is native FP. Only batch-1 decode attention is quantized.",
        "summary": summary,
        "rows": rows,
    }
    path = write_report(report, output_dir)
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    return report


def load_huggingface_tasks(limit: int) -> list[dict[str, Any]]:
    """Load the first ``limit`` public rows from each stage-5 suite."""
    from datasets import load_dataset

    tasks: list[dict[str, Any]] = []
    aime = load_dataset("HuggingFaceH4/aime_2024", split="train")
    for row in list(aime)[:limit]:
        tasks.append(
            {
                "suite": "aime",
                "prompt": f"{row['problem']}\nReply with the integer only.",
                "answer": str(row["answer"]),
                "kind": "integer",
            }
        )
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    for row in list(humaneval)[:limit]:
        tasks.append(
            {
                "suite": "humaneval",
                "prompt": row["prompt"],
                "answer": row["entry_point"],
                "kind": "function",
                "calls": (),
                "test": row["test"],
            }
        )
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    for row in list(mmlu)[:limit]:
        choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
        tasks.append(
            {
                "suite": "mmlu",
                "prompt": f"{row['question']}\n{choices}\nReply with the letter only.",
                "answer": "ABCD"[int(row["answer"])],
                "kind": "letter",
            }
        )
    gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    for row in list(gpqa)[:limit]:
        tasks.append(
            {
                "suite": "gpqa",
                "prompt": f"{row['Question']}\nReply with the correct choice text.",
                "answer": row["Correct Answer"],
                "kind": "contains",
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", choices=("micro", "huggingface"), default="micro")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    tasks = list(MICRO_TASKS) if args.dataset == "micro" else load_huggingface_tasks(args.limit)
    run_scorecard(args.model, tasks, max_new_tokens=args.max_new_tokens, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
