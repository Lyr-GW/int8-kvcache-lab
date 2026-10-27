"""Run a small reproducible MMLU sample through the vLLM scorecard."""

from __future__ import annotations

import argparse
import random

from datasets import load_dataset

from int8_kvcache_lab.vllm_scorecard import run_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--granularity", choices=("per_channel", "per_token_head"), required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    dataset = load_dataset("cais/mmlu", "all", split="test")
    if args.limit < 1 or args.limit > len(dataset):
        raise ValueError("limit must be between 1 and the dataset size")
    indices = sorted(random.Random(args.seed).sample(range(len(dataset)), args.limit))
    tasks = []
    for index in indices:
        row = dataset[index]
        choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
        tasks.append({
            "suite": "mmlu",
            "prompt": f"{row['question']}\n{choices}\nReply with the answer letter only.",
            "answer": "ABCD"[int(row["answer"])],
            "kind": "letter",
            "dataset": {
                "dataset_id": "cais/mmlu", "split": "all/test",
                "fingerprint": getattr(dataset, "_fingerprint", None), "row_index": index,
            },
        })
    run_scorecard(
        model=args.model, tasks=tasks, dtype="bfloat16", max_model_len=1024,
        max_new_tokens=8, granularity=args.granularity, statistic="absmax",
        include_per_head=False, output_dir=args.output_dir, query_granularity="per_tensor",
        dataset_sampling=f"MMLU test: {args.limit} rows sampled without replacement, seed {args.seed}",
    )


if __name__ == "__main__":
    main()
