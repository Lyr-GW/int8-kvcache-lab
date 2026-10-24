"""Stage-5 vLLM scorecard for dynamic INT8 attention experiments.

Each group gets a fresh vLLM engine and evaluates the same prompts. Native FP
is the baseline; quantized groups patch the vLLM FlashAttention varlen call and
use the PyTorch reference implementations. This is an accuracy experiment, not
a resident INT8 cache or a performance benchmark.
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from typing import Any

from .reporting import default_output_dir, write_report
from .vllm_runtime import make_llm, replace_varlen_with_pytorch


def evaluation_groups(
    granularity: str, include_per_head: bool
) -> list[tuple[str, str | None, str]]:
    """Return the requested native/dequant/INT8-compute comparison groups."""
    if granularity not in ("per_head", "per_channel", "per_token_head"):
        raise ValueError("granularity must be per_head, per_channel, or per_token_head")
    groups = [
        ("native_fp", None, granularity),
        (f"{granularity}_int8_dequant", "dequant", granularity),
        (f"{granularity}_int8_gemm", "int8_gemm", granularity),
    ]
    if include_per_head and granularity != "per_head":
        groups.append(("per_head_int8_dequant", "dequant", "per_head"))
    return groups


def _last_integer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}", text)
    candidates = boxed or re.findall(r"(?<![\w.])-?\d[\d,]*(?![\w.])", text)
    return candidates[-1].replace(",", "") if candidates else None


def _letter(text: str) -> str | None:
    answers = re.findall(
        r"\b(?:answer|option|choice)\s*(?:(?:is|:)\s*)?(?:option\s*)?([ABCD])\b",
        text.upper(),
    )
    if answers:
        return answers[-1]
    tokens = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", text.upper())
    return tokens[-1] if tokens else None


_HUMANEVAL_WORKER = r'''import ast, json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1000000, 1000000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
except (ImportError, OSError, ValueError):
    pass
p = json.load(sys.stdin)
tree = ast.parse(p["source"])
allowed_modules = {"math", "itertools", "collections", "functools", "operator", "heapq", "bisect", "re", "string", "typing", "copy", "decimal", "fractions", "statistics"}
blocked_calls = {"exec", "eval", "compile", "open", "input", "breakpoint", "globals", "locals", "vars", "getattr", "setattr", "delattr", "__import__"}
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in blocked_calls):
        raise ValueError("unsafe name")
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise ValueError("unsafe attribute")
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = (
            [item.name.split(".", 1)[0] for item in node.names]
            if isinstance(node, ast.Import)
            else [node.module.split(".", 1)[0]]
        )
        if any(name not in allowed_modules for name in names if name):
            raise ValueError("module not allowed")
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] not in allowed_modules:
        raise ImportError("module not allowed")
    return __import__(name, globals, locals, fromlist, level)
safe = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "pow": pow, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "AssertionError": AssertionError,
    "ValueError": ValueError, "TypeError": TypeError, "IndexError": IndexError,
    "KeyError": KeyError, "ZeroDivisionError": ZeroDivisionError,
    "object": object, "property": property, "staticmethod": staticmethod,
    "classmethod": classmethod, "super": super,
    "__build_class__": __build_class__, "__import__": safe_import,
}
scope = {"__builtins__": safe, "__name__": "__main__"}
exec(compile(tree, "<humaneval>", "exec"), scope, scope)
scope["check"](scope[p["entry_point"]])
print("PASS")
'''


def _run_humaneval(source: str, entry_point: str, timeout_seconds: float = 6.0) -> bool:
    """Run a HumanEval completion in a constrained subprocess, never in-process."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    ):
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _HUMANEVAL_WORKER],
            input=json.dumps({"source": source, "entry_point": entry_point}),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip().endswith("PASS")


def grade(item: dict[str, Any], text: str) -> bool | None:
    """Grade a generated answer; return None only for explicitly unscored items."""
    kind = item["kind"]
    if kind == "integer":
        return _last_integer(text) == str(item["answer"]).replace(",", "")
    if kind == "letter":
        return _letter(text) == str(item["answer"]).upper()
    if kind == "humaneval":
        completion = re.sub(r"^\s*```(?:python)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE)
        has_full_function = re.search(
            rf"\b(?:def|class)\s+{re.escape(item['entry_point'])}\b", completion
        )
        source = completion if has_full_function else item["prompt"] + completion
        source += "\n\n" + item["test"]
        source += f"\n\ncheck({item['entry_point']})\n"
        return _run_humaneval(source, item["entry_point"])
    raise ValueError(f"unsupported task kind: {kind}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute accuracy and explicit denominators for every dataset/group."""
    buckets: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        buckets.setdefault((row["suite"], row["group"]), []).append(bool(row["correct"]))
    return [
        {
            "suite": suite,
            "group": group,
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values) if values else None,
        }
        for (suite, group), values in sorted(buckets.items())
    ]


def format_score_table(summary: list[dict[str, Any]], groups: list[str]) -> str:
    """Render the required native-vs-quantized dataset accuracy table."""
    by_key = {(row["suite"], row["group"]): row for row in summary}
    suites = sorted({row["suite"] for row in summary})
    headers = ["Dataset", *groups]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for suite in suites:
        cells = [suite.upper()]
        for group in groups:
            result = by_key.get((suite, group))
            if result is None or result["accuracy"] is None:
                cells.append("—")
            else:
                cells.append(f"{result['correct']}/{result['total']} ({result['accuracy']:.1%})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_prompts(engine: Any, tasks: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Use the model chat template for instruction tasks, raw completion for code stubs."""
    tokenizer_getter = getattr(engine, "get_tokenizer", None)
    tokenizer = tokenizer_getter() if tokenizer_getter is not None else None
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return [task["prompt"] for task in tasks], "raw"
    formatted = []
    for task in tasks:
        if task["suite"] == "humaneval":
            content = (
                "Complete the Python function below. Return the full function implementation only, "
                "without markdown fences or explanation.\n\n" + task["prompt"]
            )
        else:
            content = task["prompt"]
        formatted.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted, "tokenizer_chat_template"


def _new_shared_call_stats() -> dict[str, Any]:
    """Create counters visible from the forked vLLM EngineCore worker."""
    context = multiprocessing.get_context("fork")
    return {
        "quantized_pytorch_calls": context.Value("i", 0),
        "native_fallback_calls": context.Value("i", 0),
    }


def _read_shared_call_stats(stats: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(counter.value) if hasattr(counter, "value") else int(counter)
        for name, counter in stats.items()
    }


def load_huggingface_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    """Load comparable AIME, HumanEval, GPQA-Diamond, and MMLU tasks."""
    import random

    from datasets import load_dataset

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    def limited(dataset: Any, seed: int) -> list[tuple[int, Any]]:
        rows = list(dataset)
        indices = list(range(len(rows)))
        if limit is not None and limit < len(rows):
            indices = sorted(random.Random(seed).sample(indices, limit))
        return [(index, rows[index]) for index in indices]

    def provenance(dataset_id: str, split: str, dataset: Any, index: int) -> dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "split": split,
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "row_index": index,
        }

    tasks: list[dict[str, Any]] = []
    aime = load_dataset("HuggingFaceH4/aime_2024", split="train")
    for index, row in limited(aime, 7):
        tasks.append({
            "suite": "aime",
            "prompt": f"{row['problem']}\nGive a concise solution and put the final integer in \\boxed{{}}.",
            "answer": str(row["answer"]),
            "kind": "integer",
            "dataset": provenance("HuggingFaceH4/aime_2024", "train", aime, index),
        })

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    for index, row in limited(humaneval, 17):
        prompt = row["prompt"]
        tasks.append({
            "suite": "humaneval", "prompt": prompt,
            "answer": row["entry_point"], "entry_point": row["entry_point"],
            "test": row["test"], "kind": "humaneval",
            "dataset": provenance("openai/openai_humaneval", "test", humaneval, index),
        })

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    for index, row in limited(mmlu, 27):
        choices = list(zip("ABCD", row["choices"]))
        rendered = "\n".join(f"{letter}. {choice}" for letter, choice in choices)
        tasks.append({
            "suite": "mmlu",
            "prompt": f"{row['question']}\n{rendered}\nReply with the answer letter only.",
            "answer": "ABCD"[int(row["answer"])], "kind": "letter",
            "dataset": provenance("cais/mmlu", "all/test", mmlu, index),
        })

    gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    for index, row in limited(gpqa, 37):
        choices = [
            row["Correct Answer"], row["Incorrect Answer 1"],
            row["Incorrect Answer 2"], row["Incorrect Answer 3"],
        ]
        random.Random(20261013 + index).shuffle(choices)
        correct_letter = "ABCD"[choices.index(row["Correct Answer"])]
        rendered = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
        tasks.append({
            "suite": "gpqa",
            "prompt": f"{row['Question']}\n{rendered}\nReply with the answer letter only.",
            "answer": correct_letter, "kind": "letter",
            "dataset": provenance("Idavidrein/gpqa", "gpqa_diamond/train", gpqa, index),
        })
    return tasks


def _read_analysis_recommendation(path: str | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    with open(path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    recommendation = report.get("recommendation", {})
    granularity = recommendation.get("kv_granularity")
    statistic = recommendation.get("scale_statistic")
    if granularity not in ("per_head", "per_channel", "per_token_head") or statistic not in ("absmax", "p999"):
        raise ValueError(f"{path} is not a valid Stage-2 recommendation report")
    return granularity, statistic


def run_scorecard(
    *,
    model: str,
    tasks: list[dict[str, Any]],
    dtype: str,
    max_model_len: int,
    max_new_tokens: int,
    granularity: str,
    statistic: str,
    include_per_head: bool,
    output_dir: str,
    query_granularity: str = "per_head",
    dataset_sampling: str = "caller-provided task list",
) -> dict[str, Any]:
    """Run the same task set through native and dynamically quantized vLLM paths."""
    from .vllm_runtime import configure_worker_multiprocessing

    if not tasks:
        raise ValueError("scorecard requires at least one task")
    if granularity not in ("per_head", "per_channel", "per_token_head"):
        raise ValueError("granularity must be per_head, per_channel, or per_token_head")
    if statistic not in ("absmax", "p999"):
        raise ValueError("statistic must be absmax or p999")
    if query_granularity not in ("per_head", "per_channel"):
        raise ValueError("query_granularity must be per_head or per_channel")
    configure_worker_multiprocessing("fork")
    try:
        from vllm import SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "Stage 5 requires the vLLM capture runtime; use a fresh Colab runtime"
        ) from error

    groups = evaluation_groups(granularity, include_per_head)
    rows: list[dict[str, Any]] = []
    call_stats_by_group: dict[str, dict[str, int]] = {}
    prompt_format: str | None = None
    for group, implementation, group_granularity in groups:
        call_stats = _new_shared_call_stats() if implementation is not None else {}
        wrapper = (
            nullcontext()
            if implementation is None
            else replace_varlen_with_pytorch(
                implementation,
                granularity=group_granularity,
                statistic=statistic,
                query_granularity=query_granularity,
                call_stats=call_stats,
            )
        )
        with wrapper:
            engine = make_llm(
                model,
                dtype=dtype,
                max_model_len=max_model_len,
                worker_multiprocessing_method="fork",
            )
            params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=7)
            prompts, current_prompt_format = _format_prompts(engine, tasks)
            if prompt_format is None:
                prompt_format = current_prompt_format
            elif prompt_format != current_prompt_format:
                raise RuntimeError("vLLM groups returned inconsistent tokenizer prompt formats")
            outputs = engine.generate(prompts, params)
            completions = [output.outputs[0].text for output in outputs]
            del engine
        if implementation is not None:
            observed_call_stats = _read_shared_call_stats(call_stats)
            if observed_call_stats.get("quantized_pytorch_calls", 0) == 0:
                raise RuntimeError(
                    f"{group} did not intercept any vLLM attention calls; "
                    "refusing to emit a misleading scorecard"
                )
            call_stats_by_group[group] = observed_call_stats
        for task, completion in zip(tasks, completions, strict=True):
            rows.append({
                "suite": task["suite"], "group": group, "prompt": task["prompt"],
                "completion": completion, "correct": grade(task, completion),
            })

    summary = summarize(rows)
    group_names = [name for name, _, _ in groups]
    score_table = format_score_table(summary, group_names)
    report = {
        "kind": "stage5_vllm_dynamic_int8_scorecard",
        "model": model,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "sampling": {"temperature": 0.0, "seed": 7},
        "attention_backend": "FLASH_ATTN",
        "prompt_format": prompt_format,
        "kv_granularity": granularity,
        "scale_statistic": statistic,
        "dataset_sampling": dataset_sampling,
        "dataset_sample_seeds": {"aime": 7, "humaneval": 17, "mmlu": 27, "gpqa": 37},
        "task_definitions": tasks,
        "gpqa_choice_shuffle_seed": 20261013,
        "query_quantization": f"{query_granularity}_dynamic",
        "query_granularity": query_granularity,
        "groups": group_names,
        "attention_call_counts": call_stats_by_group,
        "task_counts": {
            suite: sum(task["suite"] == suite for task in tasks)
            for suite in sorted({task["suite"] for task in tasks})
        },
        "note": (
            "Eligible prefill/decode calls use the vLLM attention path; scales are recomputed "
            "dynamically from each call's Q and logical KV. PyTorch INT8-GEMM is simulated in "
            "FP32, not a CUDA INT8 kernel or resident INT8 cache."
        ),
        "summary": summary,
        "score_table_markdown": score_table,
        "rows": rows,
    }
    path = write_report(report, output_dir)
    print(score_table)
    print(f"wrote {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", choices=("micro", "huggingface"), default="huggingface")
    parser.add_argument(
        "--limit", type=int, default=100,
        help="rows per public dataset; omit limit with --all",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="evaluate every row in the selected public splits",
    )
    parser.add_argument("--dtype", choices=("half", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--query-granularity", choices=("per_head", "per_channel"), default="per_head")
    parser.add_argument("--granularity", choices=("per_head", "per_channel", "per_token_head"))
    parser.add_argument("--statistic", choices=("absmax", "p999"))
    parser.add_argument(
        "--analysis-report",
        help="Stage-2 JSON; uses its granularity/statistic recommendation",
    )
    parser.add_argument("--include-per-head", action="store_true")
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    recommended_granularity, recommended_statistic = _read_analysis_recommendation(
        args.analysis_report
    )
    granularity = args.granularity or recommended_granularity or "per_channel"
    statistic = args.statistic or recommended_statistic or "absmax"
    if args.dataset == "micro":
        tasks = [
            {"suite": "aime", "prompt": "What is 17 * 23? Return the integer in \\boxed{}.", "answer": "391", "kind": "integer"},
            {"suite": "humaneval", "prompt": "def max2(a, b):\n    \"\"\"Return the larger integer.\"\"\"\n", "answer": "max2", "entry_point": "max2", "test": "def check(candidate):\n    assert candidate(3, 9) == 9\n    assert candidate(-4, -1) == -1\n", "kind": "humaneval"},
            {"suite": "gpqa", "prompt": "Which particle has negative charge? A. proton B. electron C. neutron D. photon. Reply with the letter only.", "answer": "B", "kind": "letter"},
            {"suite": "mmlu", "prompt": "Capital of France? A. Berlin B. Paris C. Madrid D. Rome. Reply with the letter only.", "answer": "B", "kind": "letter"},
        ]
    else:
        tasks = load_huggingface_tasks(None if args.all else args.limit)
    run_scorecard(
        model=args.model, tasks=tasks, dtype=args.dtype, max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens, granularity=granularity, statistic=statistic,
        include_per_head=args.include_per_head, output_dir=args.output_dir or str(default_output_dir()),
        query_granularity=args.query_granularity,
        dataset_sampling="all rows" if args.all else f"seeded random sample, up to {args.limit} rows per suite",
    )


if __name__ == "__main__":
    main()
